import os
import json
import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from io import BytesIO
from extrato_parser import extrair_texto_pdf, processar_extrato_principal 

# ==================== FUNÇÕES DE CÁLCULO DFC (CORRIGIDA, ESTRUTURADA E ANALÍTICA) ====================

def formatar_moeda(val):
    """Formata um valor float ou int para string no formato R$."""
    if isinstance(val, (int, float)):
        # Formatação robusta: R$ 1.234,56 ou -R$ 1.234,56
        if pd.isna(val) or val == 0:
            return ""
        if val < 0:
            val_abs = abs(val)
            # Substitui ponto por vírgula e vírgula por ponto para formato BR
            return f"-R$ {val_abs:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return val

def calcular_demonstracao_fluxo_caixa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o DFC (Demonstração do Fluxo de Caixa) detalhado por Receita/Despesa e Natureza Analítica,
    agrupando por Mês/Ano.
    """
    if df.empty:
        return pd.DataFrame()

    df_calculo = df.copy()

    # 1. Pré-processamento e Limpeza
    try:
        # Filtra a classificação "Pessoal" (conforme a nova regra)
        df_calculo = df_calculo[df_calculo['subgrupo'].isin(["Operacional", "Investimento", "Financiamento"])]

        # Converte Data e extrai Mês/Ano
        df_calculo['data'] = pd.to_datetime(df_calculo['data'], format='%d/%m/%Y', errors='coerce')
        df_calculo.dropna(subset=['data'], inplace=True)
        df_calculo['Mes_Ano'] = df_calculo['data'].dt.strftime('%Y-%m')
        
        # Converte 'valor' para float de forma robusta
        df_calculo['valor'] = df_calculo['valor'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    except Exception as e:
        st.error(f"Erro na conversão inicial de Data/Valor para cálculo do DFC: {e}")
        return pd.DataFrame()

    # 2. Determinar o sinal do Fluxo (Receita = positivo, Despesa = negativo)
    df_calculo['Fluxo'] = df_calculo.apply(
        lambda row: row['valor'] if row['natureza_geral'].upper() == 'RECEITA' else -row['valor'], 
        axis=1
    )

    # 3. Agrupamento Detalhado
    # Agrupamos por subgrupo, natureza_geral, e natureza_analitica para o detalhe
    df_agregado = df_calculo.groupby(['Mes_Ano', 'subgrupo', 'natureza_geral', 'natureza_analitica'])['Fluxo'].sum().reset_index()

    # Pivotagem: Colunas de Meses e Índice Multi-nível
    df_pivot_detalhe = df_agregado.pivot_table(
        index=['subgrupo', 'natureza_geral', 'natureza_analitica'], 
        columns='Mes_Ano', 
        values='Fluxo', 
        fill_value=0
    )
    
    colunas_meses = df_pivot_detalhe.columns.tolist()

    # 4. Construção do Relatório Hierárquico
    
    ordem_atividades = ["Operacional", "Investimento", "Financiamento"]
    ordem_geral = ["RECEITA", "DESPESA"]
    df_final_report = []
    
    for atividade in ordem_atividades:
        # Linha de Título principal (Caixa Operacional, Investimento, Financiamento)
        titulo_atividade = f"Caixa de {atividade}" if atividade != "Operacional" else "Caixa Operacional"
        df_final_report.append(pd.Series([titulo_atividade, ''], index=['Atividade', 'Detalhe'] + colunas_meses))
        
        # DataFrame filtrado apenas para a Atividade atual
        if atividade not in df_pivot_detalhe.index.get_level_values('subgrupo'):
            df_final_report.append(pd.Series(['---', 'Nenhuma movimentação neste período'], index=['Atividade', 'Detalhe'] + colunas_meses))
            continue

        for natureza in ordem_geral:
            
            # Tenta obter os dados da natureza (Receita/Despesa)
            try:
                # Filtrar o MultiIndex para a atividade e a natureza
                detalhes_natureza = df_pivot_detalhe.loc[(atividade, natureza)]
                detalhes_natureza = detalhes_natureza.reset_index()
                detalhes_natureza.rename(columns={'natureza_analitica': 'Detalhe'}, inplace=True)
                
                # Linha de Título da Natureza (Receitas/Despesas)
                df_final_report.append(pd.Series(['', natureza.capitalize()], index=['Atividade', 'Detalhe'] + colunas_meses))
                
                # Adicionar as linhas de Detalhe (Natureza Analítica)
                for index, row in detalhes_natureza.iterrows():
                    # O valor detalhado para despesas deve ser mostrado em absoluto (positivo)
                    linha = pd.Series({'Atividade': '', 'Detalhe': f"  {row['Detalhe']}"}, index=['Atividade', 'Detalhe'] + colunas_meses)
                    for mes in colunas_meses:
                        linha[mes] = abs(row[mes]) if natureza == 'DESPESA' else row[mes]
                    df_final_report.append(linha)
                    
            except KeyError:
                # Não há receitas ou despesas para esta atividade neste período
                continue

        # 5. Cálculo e Adição do Subtotal da Atividade
        subtotal_series = df_pivot_detalhe.loc[atividade].sum(axis=0)
        subtotal_row = pd.Series({
            'Atividade': f"Total do {titulo_atividade}", 
            'Detalhe': ''
        }, index=['Atividade', 'Detalhe'] + colunas_meses)
        
        for mes in colunas_meses:
            subtotal_row[mes] = subtotal_series[mes]
            
        df_final_report.append(subtotal_row)
        df_final_report.append(pd.Series('', index=['Atividade', 'Detalhe'] + colunas_meses)) # Linha Vazia

    # Concatena todas as Series em um DataFrame
    df_final = pd.DataFrame(df_final_report).reset_index(drop=True)
    df_final = df_final[df_final['Atividade'] != '---'].reset_index(drop=True) # Remove linhas de "Nenhuma movimentação"

    # 6. Adicionar a linha de Geração de Caixa Total
    
    # Filtra apenas as linhas de Total do Caixa para somar
    df_subtotais_apenas = df_final[df_final['Atividade'].str.startswith('Total do Caixa', na=False)]
    
    # Converte para float antes de somar e soma
    total_caixa_series = df_subtotais_apenas[colunas_meses].apply(lambda x: pd.to_numeric(x.apply(lambda v: v.replace('R$ ', '').replace('.', '').replace(',', '.').replace('-', '') if isinstance(v, str) else v), errors='coerce')).sum(axis=0)
    
    total_caixa_row = pd.Series({'Atividade': 'Geração de Caixa do Período', 'Detalhe': ''}, index=['Atividade', 'Detalhe'] + colunas_meses)
    for mes in colunas_meses:
        total_caixa_row[mes] = total_caixa_series[mes]
        
    df_final.loc[len(df_final)] = total_caixa_row


    # 7. Formatação Numérica Final para R$
    for col in colunas_meses:
        # Garante que a coluna seja float antes de formatar. 
        # NOTA: O cálculo já foi feito, aqui formatamos os resultados do total
        df_final[col] = df_final[col].apply(formatar_moeda)

    # Limpeza de linhas de espaçamento
    df_final.loc[df_final['Atividade'] == '', colunas_meses] = ''
    
    return df_final


# ==================== CONFIGURAÇÃO DO STREAMLIT ====================

# -------------------------------------------------
# Configuração do Streamlit
# -------------------------------------------------
st.set_page_config(page_title="Hedgewise - Extrato Profissional", layout="wide")
st.title("Análise de Extrato Bancário com IA")

# Inicializa o session_state para armazenar o DF classificado
if 'df_classificado_final' not in st.session_state:
    st.session_state['df_classificado_final'] = pd.DataFrame()

# Definição da configuração de colunas para o editor de dados
COLUMN_CONFIG_EDITOR = {
    "subgrupo": st.column_config.SelectboxColumn(
        "Subgrupo (DFC/CPC 03)",
        help="Classificação DFC/CPC 03",
        # REMOVIDO "Pessoal"
        options=["Operacional", "Investimento", "Financiamento"],
        required=True,
    ),
    "natureza_juridica": st.column_config.SelectboxColumn(
        "Natureza Jurídica",
        help="Classificação Pessoal ou Empresarial",
        options=["Empresarial", "Pessoal"],
        required=True,
    ),
    "natureza_geral": st.column_config.SelectboxColumn(
        "Natureza Geral",
        help="Classificação Principal (Receita ou Despesa)",
        options=["Receita", "Despesa"],
        required=True,
    ),
    "natureza_analitica": st.column_config.TextColumn(
        "Natureza Analítica",
        help="Classificação detalhada (Ex: Salário, Aluguel)",
        required=True,
    ),
    # Desabilita a edição das colunas originais do extrato (somente leitura)
    "data": st.column_config.Column(disabled=True),
    "historico": st.column_config.Column(disabled=True),
    "valor": st.column_config.Column(disabled=True),
    "tipo": st.column_config.Column(disabled=True),
    "arquivo_origem": st.column_config.Column(disabled=True),
}


uploaded_files = st.file_uploader(
    "📎 Envie os extratos bancários em PDF (múltiplos arquivos permitidos)",
    type=["pdf"],
    accept_multiple_files=True
)

# -------------------------------------------------
# Lógica de processamento (Bloco de COMPUTATIONAL / PESADO)
# -------------------------------------------------
if uploaded_files:
    
    if st.button("🚀 Iniciar Classificação Automática das Transações"):
        
        # Limpa o estado anterior
        st.session_state['df_classificado_final'] = pd.DataFrame() 

        API_KEY = os.getenv("GEMINI_API_KEY")
        if not API_KEY:
            st.error("Chave da API do Gemini (GEMINI_API_KEY) não configurada.")
            st.stop()
        
        try:
            client = genai.Client(api_key=API_KEY)
        except Exception as e:
            st.error(f"Erro ao inicializar o cliente Gemini: {e}")
            st.stop()
            
        todos_dados_classificados = []

        # LOOP SOBRE CADA ARQUIVO ENVIADO
        for i, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            st.subheader(f"📂 Processando Arquivo {i+1} de {len(uploaded_files)}: {file_name}")
            
            pdf_bytes = uploaded_file.read()
            pdf_stream = BytesIO(pdf_bytes)
            
            # --- EXTRAÇÃO E NORMALIZAÇÃO ---
            try:
                pdf_stream.seek(0)
                texto = extrair_texto_pdf(pdf_stream)
            except Exception as e:
                st.error(f"Erro ao ler PDF de {file_name}: {e}. Pulando.")
                continue

            if not texto or len(texto.strip()) < 50:
                st.warning(f"Nenhum texto legível foi extraído de {file_name}. Pulando.")
                continue

            with st.expander(f"📄 Texto extraído de {file_name}"):
                st.text_area(f"Conteúdo do extrato {file_name}:", texto, height=200)

            st.info("Iniciando processamento universal de transações...")

            pdf_stream.seek(0) 
            df_transacoes = processar_extrato_principal(pdf_stream)

            if df_transacoes.empty or 'Tipo' not in df_transacoes.columns:
                st.warning(f"Não foi possível identificar movimentações financeiras válidas em {file_name}.")
                continue
                
            st.success(f"Transações Extraídas de {file_name}: {len(df_transacoes)}")


            # --- CLASSIFICAÇÃO GEMINI ---
            TAMANHO_DO_LOTE = 50 
            dados_classificados_lote = []
            
            # JSON SCHEMA e Configurações (Mantidas)
            json_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "A data da transação."},
                        "historico": {"type": "string", "description": "O histórico ou descrição original da transação."},
                        "valor": {"type": "string", "description": "O valor original da transação."},
                        "tipo": {"type": "string", "description": "O tipo original da transação ('D' para débito, 'C' para crédito)."},
                        "natureza_geral": {"type": "string", "description": "Classificação PRINCIPAL em 'Despesa' ou 'Receita'."},
                        # Subgrupo Pessoal é aceito aqui, mas filtrado no DFC
                        "subgrupo": {"type": "string", "description": "Classificação DFC/CPC 03: 'Operacional', 'Investimento', 'Financiamento' ou 'Pessoal'."}, 
                        "natureza_analitica": {"type": "string", "description": "Classificação detalhada e linear da transação (Ex: 'Salário', 'Aluguel', 'Fornecedores')."},
                        "natureza_juridica": {"type": "string", "description": "Classificação 'Pessoal' ou 'Empresarial'."}
                    },
                    "required": ["data", "historico", "valor", "tipo", "natureza_geral", "subgrupo", "natureza_analitica", "natureza_juridica"]
                }
            }
            
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=json_schema,
                temperature=0.1, 
                thinking_config=types.ThinkingConfig(thinking_budget=0) 
            )

            n_batches = len(df_transacoes) // TAMANHO_DO_LOTE + (1 if len(df_transacoes) % TAMANHO_DO_LOTE > 0 else 0)
            progress_bar = st.progress(0, text=f"Iniciando a classificação de {file_name}...")
            
            for j in range(n_batches):
                start_index = j * TAMANHO_DO_LOTE
                end_index = start_index + TAMANHO_DO_LOTE
                lote_df = df_transacoes.iloc[start_index:end_index]
                
                texto_formatado_lote = "\n".join(
                    f"{row.Data} | {row['Histórico']} | {row['Valor']} | Tipo: {row['Tipo']}"
                    for _, row in lote_df.iterrows()
                )

                prompt_lote = f"""
Você é um analista financeiro sênior da Hedgewise, especializado na composição da Demonstração de Fluxo de Caixa (DFC) conforme o CPC 03 (IAS 7).
Sua tarefa é analisar AS {len(lote_df)} MOVIMENTAÇÕES BANCÁRIAS extraídas de "{file_name}" e retornar um JSON estritamente conforme o schema fornecido.
... (Resto do prompt mantido) ...
Movimentações extraídas:
{texto_formatado_lote}
                """
                
                progress_bar.progress((j + 1) / n_batches, text=f"Lote {j+1} de {n_batches} para {file_name}...")
                
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt_lote,
                        config=config,
                    )
                    resposta_texto = response.text
                    dados_lote = json.loads(resposta_texto)
                    
                    if isinstance(dados_lote, list):
                        for transacao in dados_lote:
                            transacao['arquivo_origem'] = file_name
                        dados_classificados_lote.extend(dados_lote)
                    else:
                        st.warning(f"Lote {j+1} de {file_name}: Retorno JSON inesperado. Ignorado.")

                except Exception as e:
                    st.error(f"Erro no Lote {j+1} de {file_name}: {e}")
                    
            progress_bar.empty()
            st.success(f"✅ Classificação de {file_name} concluída.")
            
            todos_dados_classificados.extend(dados_classificados_lote)

        # SALVA O DATAFRAME CONSOLIDADO NO SESSION STATE
        if todos_dados_classificados:
            df_classificado = pd.DataFrame(todos_dados_classificados)
            colunas_ordenadas = [
                'arquivo_origem', 'data', 'historico', 'valor', 'tipo', 
                'natureza_geral', 'subgrupo', 'natureza_analitica', 'natureza_juridica'
            ]
            df_classificado = df_classificado[colunas_ordenadas]
            st.session_state['df_classificado_final'] = df_classificado.copy()
            st.balloons()
            st.success("🎉 Processamento concluído! Edite a tabela abaixo para ajustes finais.")
        else:
            st.warning("Nenhum dado classificado foi retornado após o processamento.")


# -------------------------------------------------
# Bloco de EDIÇÃO, VISUALIZAÇÃO DFC E DOWNLOAD
# -------------------------------------------------

if not st.session_state['df_classificado_final'].empty:
    st.subheader("🛠️ Ajuste Manual e Validação dos Dados Classificados")
    
    # 1. Editor de Dados (Retorna o DF editado pelo usuário)
    df_editado = st.data_editor(
        st.session_state['df_classificado_final'],
        column_config=COLUMN_CONFIG_EDITOR,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic"
    )
    
    # 2. VISUALIZAÇÃO DFC (Calculado a partir do DF EDITADO)
    st.subheader("📈 Demonstração do Fluxo de Caixa (DFC/CPC 03) Detalhada por Mês")

    # Calcula e exibe o DFC
    df_fluxo = calcular_demonstracao_fluxo_caixa(df_editado)
    
    if not df_fluxo.empty:
        # Define um estilo de exibição para destacar os totais
        def highlight_total(row):
            is_subtotal = row['Atividade'].startswith('Total do Caixa')
            is_grand_total = row['Atividade'].startswith('Geração de Caixa do Período')
            is_header = row['Detalhe'] == '' and not is_subtotal and not is_grand_total
            is_sub_header = row['Detalhe'].startswith('Receitas') or row['Detalhe'].startswith('Despesas')
            
            if is_grand_total:
                return ['background-color: #a0c0e0; font-weight: bold; border-top: 2px solid black'] * len(row)
            if is_subtotal:
                return ['background-color: #e0e0f0; font-weight: bold'] * len(row)
            if is_header:
                return ['font-weight: bold; background-color: #f0f0f5'] * len(row)
            if is_sub_header:
                return ['font-style: italic'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_fluxo.style.apply(highlight_total, axis=1),
            use_container_width=True, 
            hide_index=True
        )
        st.caption("Valores agrupados por Atividade (Fluxo) e detalhados pela Natureza Analítica (subcategoria). Despesas são apresentadas em valores absolutos (positivos) dentro de seus grupos.")
    else:
        st.warning("Não foi possível calcular o DFC. Verifique se as colunas 'data', 'valor' e as classificações estão válidas e se há transações que não são classificadas como 'Pessoal'.")


    # 3. DOWNLOAD (a partir do DF EDITADO)
    st.markdown("---")
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(df_editado)
    
    st.download_button(
        label="⬇️ Baixar Tabela Classificada (CSV) - Versão Editada",
        data=csv_data,
        file_name='extratos_classificados_editados.csv',
        mime='text/csv',
        key='download_csv_button'
    )
