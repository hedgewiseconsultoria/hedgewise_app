import os
import json
import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from io import BytesIO
from extrato_parser import extrair_texto_pdf, processar_extrato_principal # Certifique-se que extrato_parser.py está atualizado

# ==================== FUNÇÕES DE CÁLCULO DFC ====================

def calcular_demonstracao_fluxo_caixa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o DFC (Demonstração do Fluxo de Caixa) com base no CPC 03,
    agrupando por Mês/Ano e Subgrupo.
    """
    if df.empty:
        return pd.DataFrame()

    df_calculo = df.copy()

    # 1. Pré-processamento e Limpeza
    # Converte 'data' para datetime e extrai Mês/Ano
    try:
        df_calculo['data'] = pd.to_datetime(df_calculo['data'], format='%d/%m/%Y', errors='coerce')
        df_calculo.dropna(subset=['data'], inplace=True)
        df_calculo['Mes_Ano'] = df_calculo['data'].dt.strftime('%Y-%m')
        
        # Converte 'valor' para float, tratando possíveis strings remanescentes (embora a normalização deva resolver)
        df_calculo['valor'] = df_calculo['valor'].astype(str).str.replace('.', '').str.replace(',', '.', regex=False).astype(float)
    except Exception as e:
        st.error(f"Erro ao converter dados para cálculo do DFC: {e}")
        return pd.DataFrame()

    # 2. Determinar o sinal do Fluxo (Débito vs. Crédito, ajustado por natureza_geral)
    # Receita (C) é positivo, Despesa (D) é negativo.
    df_calculo['Fluxo'] = df_calculo.apply(
        lambda row: row['valor'] if row['natureza_geral'].upper() == 'RECEITA' else -row['valor'], 
        axis=1
    )

    # 3. Agrupamento e Soma
    # O foco é no 'subgrupo' (Operacional, Investimento, Financiamento, Pessoal)
    df_agrupado = df_calculo.groupby(['Mes_Ano', 'subgrupo'])['Fluxo'].sum().reset_index()
    df_agrupado.rename(columns={'subgrupo': 'Atividade', 'Fluxo': 'Valor_Fluxo'}, inplace=True)
    
    # 4. Pivotagem para formato de relatório (Meses em colunas)
    df_pivot = df_agrupado.pivot_table(
        index='Atividade', 
        columns='Mes_Ano', 
        values='Valor_Fluxo', 
        fill_value=0
    )
    
    # 5. Adicionar Linhas de Totalização
    total_por_mes = df_pivot.sum(axis=0)
    
    # Dicionário de reordenação de linhas
    order = {
        "Operacional": 1, 
        "Investimento": 2, 
        "Financiamento": 3, 
        "Pessoal": 4
    }
    
    # 6. Reestruturação Final
    df_final = df_pivot.reset_index()
    df_final['Ordem'] = df_final['Atividade'].map(order).fillna(5) # Pessoal é 4, outros (se houver) 5
    df_final = df_final.sort_values(by='Ordem').drop(columns='Ordem')
    
    # Adiciona a linha de Geração de Caixa Total
    df_total = pd.DataFrame(total_por_mes).T
    df_total['Atividade'] = 'GERAÇÃO DE CAIXA TOTAL'
    df_final = pd.concat([df_final, df_total], ignore_index=True)
    
    # Formatação (opcional, mas melhora a visualização)
    colunas_mes = [col for col in df_final.columns if col != 'Atividade']
    df_final[colunas_mes] = df_final[colunas_mes].applymap(lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    
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
        options=["Operacional", "Investimento", "Financiamento", "Pessoal"],
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
    st.subheader("📈 Demonstração do Fluxo de Caixa (DFC/CPC 03) por Mês")

    # Calcula e exibe o DFC
    df_fluxo = calcular_demonstracao_fluxo_caixa(df_editado)
    
    if not df_fluxo.empty:
        st.dataframe(df_fluxo, use_container_width=True, hide_index=True)
        st.caption("Valores apresentados conforme a Demonstração do Fluxo de Caixa (DFC), no método direto, agrupados por mês. Valores positivos representam Geração de Caixa, e negativos, Uso de Caixa.")
    else:
        st.warning("Não foi possível calcular o DFC. Verifique a coluna 'data' e 'valor'.")


    # 3. DOWNLOAD (a partir do DF EDITADO)
    st.markdown("---")
    
    @st.cache_data
    def convert_df_to_csv(df):
        # Converte o DF editado em CSV
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(df_editado)
    
    st.download_button(
        label="⬇️ Baixar Tabela Classificada (CSV) - Versão Editada",
        data=csv_data,
        file_name='extratos_classificados_editados.csv',
        mime='text/csv',
        key='download_csv_button'
    )
