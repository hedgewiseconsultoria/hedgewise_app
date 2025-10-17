import os
import json
import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from io import BytesIO
from extrato_parser import extrair_texto_pdf, processar_extrato_principal

# -------------------------------------------------
# Configuração do Streamlit
# -------------------------------------------------
st.set_page_config(page_title="Hedgewise - Extrato Profissional", layout="wide")
st.title("Análise de Extrato Bancário com IA")

# 1. Inicializa o session_state para armazenar o DF classificado
if 'df_classificado_final' not in st.session_state:
    st.session_state['df_classificado_final'] = pd.DataFrame()

# Definição da configuração de colunas para o editor de dados
# Isso cria os dropdowns de seleção
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
            # ... (Lógica de extração e classificação da IA, como antes) ...

            file_name = uploaded_file.name
            st.subheader(f"📂 Processando Arquivo {i+1} de {len(uploaded_files)}: {file_name}")
            
            pdf_bytes = uploaded_file.read()
            pdf_stream = BytesIO(pdf_bytes)

            # --- PARTE 1: EXTRAÇÃO E NORMALIZAÇÃO (extrato_parser.py) ---
            
            try:
                pdf_stream.seek(0)
                texto = extrair_texto_pdf(pdf_stream)
            except Exception as e:
                st.error(f"Erro ao ler PDF de {file_name}: {e}. Pulando para o próximo.")
                continue

            if not texto or len(texto.strip()) < 50:
                st.warning(f"Nenhum texto legível foi extraído de {file_name}. Pulando.")
                continue

            with st.expander(f"📄 Texto extraído de {file_name}"):
                st.text_area(f"Conteúdo do extrato {file_name}:", texto, height=200)

            st.info("Iniciando processamento universal (Best-Effort) de transações...")

            pdf_stream.seek(0) 
            df_transacoes = processar_extrato_principal(pdf_stream)

            if df_transacoes.empty or 'Tipo' not in df_transacoes.columns:
                st.warning(f"Não foi possível identificar movimentações financeiras válidas em {file_name}.")
                continue
                
            st.success(f"Transações Extraídas de {file_name}: {len(df_transacoes)}")


            # --- PARTE 2: CLASSIFICAÇÃO GEMINI (LOOP DE LOTE) ---

            TAMANHO_DO_LOTE = 50 
            dados_classificados_lote = []
            
            # JSON SCHEMA ATUALIZADO (mantido)
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

            st.info(f"Analisando {len(df_transacoes)} transações de {file_name} com taxonomia CPC 03...")

            n_batches = len(df_transacoes) // TAMANHO_DO_LOTE + (1 if len(df_transacoes) % TAMANHO_DO_LOTE > 0 else 0)
            progress_bar = st.progress(0, text=f"Iniciando o processamento dos lotes de {file_name}...")
            
            for j in range(n_batches):
                start_index = j * TAMANHO_DO_LOTE
                end_index = start_index + TAMANHO_DO_LOTE
                lote_df = df_transacoes.iloc[start_index:end_index]
                
                texto_formatado_lote = "\n".join(
                    f"{row.Data} | {row['Histórico']} | {row['Valor']} | Tipo: {row['Tipo']}"
                    for _, row in lote_df.iterrows()
                )

                # O prompt completo é mantido
                prompt_lote = f"""
Você é um analista financeiro sênior da Hedgewise, especializado na composição da Demonstração de Fluxo de Caixa (DFC) conforme o CPC 03 (IAS 7).
... (Instruções completas omitidas por brevidade, mas o texto do prompt permanece o mesmo) ...
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
                        # Adiciona o nome do arquivo a cada transação para identificar a origem
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

        # CRIA O DATAFRAME CONSOLIDADO E SALVA NO SESSION STATE
        if todos_dados_classificados:
            df_classificado = pd.DataFrame(todos_dados_classificados)
            # Reorganiza as colunas (necessário antes de salvar para o editor)
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
# Bloco de EDIÇÃO E DOWNLOAD (Executa em toda interação)
# -------------------------------------------------

if not st.session_state['df_classificado_final'].empty:
    st.subheader("🛠️ Ajuste Manual e Validação dos Dados Classificados")
    
    # Exibe o editor de dados. O resultado editado é retornado a cada interação.
    df_editado = st.data_editor(
        st.session_state['df_classificado_final'],
        column_config=COLUMN_CONFIG_EDITOR,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic" # Permite adicionar/remover linhas se necessário
    )

    st.caption(f"Total de Transações: {len(df_editado)}")
    
    # Gera o CSV para download a partir do DataFrame EDITADO
    @st.cache_data
    def convert_df_to_csv(df):
        # Remove a coluna de arquivo_origem se o usuário preferir um CSV mais limpo,
        # mas por padrão vamos mantê-la.
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(df_editado)
    
    # O botão de download usa o DF editado
    st.download_button(
        label="⬇️ Baixar Tabela Classificada (CSV)",
        data=csv_data,
        file_name='extratos_classificados_editados.csv',
        mime='text/csv',
        key='download_csv_button'
    )
