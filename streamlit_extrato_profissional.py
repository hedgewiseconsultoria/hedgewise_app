import os
import json
import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from io import BytesIO

# IMPORTAÇÕES ESSENCIAIS DO SEU CÓDIGO ORIGINAL
from extrato_parser import extrair_texto_pdf, processar_extrato_principal

# -------------------------------------------------
# Configuração do Streamlit
# -------------------------------------------------
st.set_page_config(page_title="Hedgewise - Extrato Profissional", layout="wide")
st.title("Análise de Extrato Bancário com IA")

# 1. AJUSTE: Permite múltiplos arquivos
uploaded_files = st.file_uploader(
    "📎 Envie os extratos bancários em PDF (múltiplos arquivos permitidos)",
    type=["pdf"],
    accept_multiple_files=True  # <--- MUDANÇA PRINCIPAL
)

# Inicializa uma lista para armazenar todos os resultados classificados
todos_dados_classificados = []

# -------------------------------------------------
# Lógica de processamento (Início)
# -------------------------------------------------
if uploaded_files:
    
    # Adiciona o botão de processamento
    if st.button("🚀 Iniciar Processamento Universal das Transações"):
        
        API_KEY = os.getenv("GEMINI_API_KEY")
        if not API_KEY:
            st.error("Chave da API do Gemini (GEMINI_API_KEY) não configurada.")
            st.stop()
        
        try:
            client = genai.Client(api_key=API_KEY)
        except Exception as e:
            st.error(f"Erro ao inicializar o cliente Gemini: {e}")
            st.stop()

        
        # 2. AJUSTE: LOOP SOBRE CADA ARQUIVO ENVIADO
        for i, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            st.subheader(f"📂 Processando Arquivo {i+1} de {len(uploaded_files)}: {file_name}")
            
            pdf_bytes = uploaded_file.read()
            pdf_stream = BytesIO(pdf_bytes)

            # --- PARTE 1: EXTRAÇÃO E NORMALIZAÇÃO (extrato_parser.py) ---
            
            # A extração de texto ainda é importante para debug e inspeção
            try:
                # O seek(0) é crucial para garantir que o stream comece do início
                # A função extrair_texto_pdf fará a leitura inicial
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

            # Chama a função mestra que executa todos os processadores e escolhe o melhor DF
            # Precisa garantir que o stream esteja no início novamente antes de chamar
            pdf_stream.seek(0) 
            df_transacoes = processar_extrato_principal(pdf_stream)

            if df_transacoes.empty:
                st.warning(f"Não foi possível identificar movimentações financeiras válidas em {file_name}.")
                continue
                
            if 'Tipo' not in df_transacoes.columns:
                st.error(f"A coluna 'Tipo' (D/C) não foi gerada para {file_name}. Pulando.")
                continue

            st.success(f"Transações Extraídas de {file_name}: {len(df_transacoes)}")
            # Opcional: mostrar as transações extraídas por arquivo
            # st.dataframe(df_transacoes, use_container_width=True)


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

                prompt_lote = f"""
Você é um analista financeiro sênior da Hedgewise, especializado na composição da Demonstração de Fluxo de Caixa (DFC) conforme o CPC 03 (IAS 7).

Sua tarefa é analisar AS {len(lote_df)} MOVIMENTAÇÕES BANCÁRIAS extraídas de "{file_name}" e retornar um JSON estritamente conforme o schema fornecido.

**Instruções de Classificação (Obrigatórias):**

1.  **natureza_geral** (Grupo): Classifique estritamente como **"Receita"** ou **"Despesa"**. (Observar se o Tipo original é 'C'rédito ou 'D'ébito, mas sempre priorizar o significado da transação).
2.  **subgrupo** (DFC/CPC 03): Classifique estritamente em uma das quatro opções:
    * **"Operacional"**: Transações que afetam o resultado e o capital de giro (vendas, compras, salários, aluguéis, impostos, fornecedores, etc.).
    * **"Investimento"**: Aquisição ou venda de ativos não circulantes (imóveis, máquinas, participações societárias), desembolsos com aplicações financeiras, resgates de aplicações financeiras, rendimentos de aplicações financeiras.
    * **"Financiamento"**: Transações com capital de terceiros ou próprio (empréstimos, integralização/distribuição de capital, dividendos), pagamentos de juros, tarifas bancárias, pagamentos de empréstimos, recebimento de empréstimos.
    * **"Pessoal"**: Despesas pessoais do sócio/empreendedor pagas pela conta da empresa (retiradas, despesas particulares, etc.), gastos que fujam da lógica do contexto empresarial.
3.  **natureza_analitica** (Subgrupo Detalhado):
    * Identifique o destino/origem de forma detalhada e linear.
    * **REGRA DE PREENCHIMENTO:** Se o histórico for genérico (ex: "Pagamento de Boleto", "Transferência TED", "Pix") e não houver informação clara, assuma **"Fornecedores"** ou **"Despesas Gerais Operacionais"** se for um débito, e **"Vendas/Serviços"** se for um crédito, pois a premissa é que a conta é empresarial.
4.  **natureza_juridica**: Classifique estritamente como **"Empresarial"** ou **"Pessoal"**.

Responda APENAS com o JSON.

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

                except json.JSONDecodeError:
                    st.error(f"Lote {j+1} de {file_name} falhou. Verifique a resposta bruta abaixo.")
                    # st.text_area(f"Resposta bruta do Lote {j+1} de {file_name}:", resposta_texto, height=150)
                except Exception as e:
                    st.error(f"Erro na chamada da API para {file_name}, lote {j+1}: {e}")
                    
            progress_bar.empty()
            st.success(f"✅ Classificação de {file_name} concluída.")
            
            # Adiciona os resultados deste arquivo à lista total
            todos_dados_classificados.extend(dados_classificados_lote)

        # -------------------------------------------------
        # Exibir Resultado Final GLOBAL
        # -------------------------------------------------
        st.subheader("📊 Resultado Final da IA (Extrato Classificado) - Consolidação")

        if todos_dados_classificados:
            
            st.success(f"Processamento de todos os arquivos concluído! Total de {len(todos_dados_classificados)} transações classificadas.")

            st.subheader("Tabela Classificada Completa")
            df_classificado = pd.DataFrame(todos_dados_classificados)
            
            # Reorganiza as colunas e inclui a nova coluna de origem
            colunas_ordenadas = [
                'arquivo_origem', 'data', 'historico', 'valor', 'tipo', 
                'natureza_geral', 'subgrupo', 'natureza_analitica', 'natureza_juridica'
            ]
            df_classificado = df_classificado[colunas_ordenadas]
            
            st.dataframe(df_classificado, use_container_width=True)
            
            # Opcional: Adicionar um botão de download
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(df_classificado)
            st.download_button(
                label="⬇️ Baixar Tabela Completa (CSV)",
                data=csv,
                file_name='extratos_classificados_consolidado.csv',
                mime='text/csv',
            )
            
        else:
            st.warning("Nenhum dado classificado foi retornado após o processamento de todos os arquivos. Verifique os erros acima.")
