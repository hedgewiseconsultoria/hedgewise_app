import os
import json
import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from io import BytesIO

# IMPORTAÇÕES ESSENCIAIS DO SEU CÓDIGO ORIGINAL
# Garanta que extrato_parser.py e suas funções estão disponíveis
from extrato_parser import extrair_texto_pdf, detectar_banco, PROCESSADORES, normalizar_transacoes 

# -------------------------------------------------
# Configuração do Streamlit
# -------------------------------------------------
st.set_page_config(page_title="Hedgewise - Extrato Profissional", layout="wide")
st.title("💼 Análise de Extrato Bancário com Google Gemini (DFK/CPC 03)")

uploaded_file = st.file_uploader("📎 Envie o extrato bancário em PDF", type=["pdf"])

# -------------------------------------------------
# Lógica de processamento (Início)
# -------------------------------------------------
if uploaded_file:
    st.info("Extraindo texto do PDF...")

    pdf_bytes = uploaded_file.read()
    pdf_stream = BytesIO(pdf_bytes)

    try:
        texto = extrair_texto_pdf(pdf_stream)
    except Exception as e:
        st.error(f"Erro ao ler PDF: {e}")
        st.stop()

    if not texto or len(texto.strip()) < 50:
        st.error("Nenhum texto legível foi extraído. O arquivo pode estar protegido ou ilegível.")
        st.stop()

    with st.expander("📄 Texto extraído do PDF"):
        st.text_area("Conteúdo do extrato:", texto, height=300)
    
    # CONFIRMAÇÃO DO BANCO
    banco_detectado = detectar_banco(texto)
    bancos_disponiveis = list(PROCESSADORES.keys())
    try:
        index_selecionado = bancos_disponiveis.index(banco_detectado)
    except ValueError:
        index_selecionado = 0 

    st.success(f"🏦 Banco detectado automaticamente: {banco_detectado}")

    banco_confirmado = st.selectbox(
        "**Confirme ou ajuste o banco para o processamento das transações:**",
        options=bancos_disponiveis,
        index=index_selecionado,
        key="bank_selector"
    )

    if not st.button(f"Processar Transações ({banco_confirmado})"):
        st.stop()
        
    processador = PROCESSADORES.get(banco_confirmado, PROCESSADORES["DESCONHECIDO"])
    transacoes = processador(texto)
    
    try:
        df_transacoes = normalizar_transacoes(transacoes)
    except Exception as e:
        st.error(f"Erro ao normalizar transações: {e}")
        st.stop()

    if df_transacoes.empty:
        st.warning("Não foi possível identificar movimentações financeiras válidas neste PDF.")
        st.stop()
        
    if 'Tipo' not in df_transacoes.columns:
        st.error("A coluna 'Tipo' (D/C) não foi gerada na normalização. O processamento da IA não funcionará corretamente.")
        st.stop()

    st.subheader(f"Transações Extraídas ({len(df_transacoes)})")
    st.dataframe(df_transacoes, use_container_width=True)

    # -------------------------------------------------
    # 3. CONFIGURAÇÕES DE LOTE E ESTRUTURA PARA GEMINI (CPC 03)
    # -------------------------------------------------
    
    TAMANHO_DO_LOTE = 50 
    
    # JSON SCHEMA ATUALIZADO com as novas colunas
    json_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "A data da transação."},
                "historico": {"type": "string", "description": "O histórico ou descrição original da transação."},
                "valor": {"type": "string", "description": "O valor original da transação."},
                "tipo": {"type": "string", "description": "O tipo original da transação ('D' para débito, 'C' para crédito)."},
                "natureza_geral": {"type": "string", "description": "Classificação PRINCIPAL em 'Despesa' ou 'Receita'."}, # Nova Coluna
                "subgrupo": {"type": "string", "description": "Classificação DFC/CPC 03: 'Operacional', 'Investimento', 'Financiamento' ou 'Pessoal'."}, # Nova Coluna
                "natureza_analitica": {"type": "string", "description": "Classificação detalhada e linear da transação (Ex: 'Salário', 'Aluguel', 'Fornecedores')."}, # Categoria anterior
                "natureza_juridica": {"type": "string", "description": "Classificação 'Pessoal' ou 'Empresarial'."} # Natureza anterior
            },
            "required": ["data", "historico", "valor", "tipo", "natureza_geral", "subgrupo", "natureza_analitica", "natureza_juridica"]
        }
    }
    
    dados_classificados_totais = []
    API_KEY = os.getenv("GEMINI_API_KEY")

    if not API_KEY:
        st.error("Chave da API do Gemini (GEMINI_API_KEY) não configurada.")
        st.stop()
        
    st.info(f"Analisando o extrato em {len(df_transacoes) // TAMANHO_DO_LOTE + 1} lotes com taxonomia CPC 03...")

    try:
        client = genai.Client(api_key=API_KEY)

        # Configurações para a geração (Forçando JSON e desabilitando Thinking)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=json_schema,
            temperature=0.1, 
            thinking_config=types.ThinkingConfig(thinking_budget=0) 
        )

        n_batches = len(df_transacoes) // TAMANHO_DO_LOTE + (1 if len(df_transacoes) % TAMANHO_DO_LOTE > 0 else 0)
        progress_bar = st.progress(0, text="Iniciando o processamento dos lotes...")
        
        for i in range(n_batches):
            start_index = i * TAMANHO_DO_LOTE
            end_index = start_index + TAMANHO_DO_LOTE
            lote_df = df_transacoes.iloc[start_index:end_index]
            
            # Prepara o texto para o lote atual
            # NOTA: O campo 'Tipo' aqui ainda é 'D' ou 'C' do extrato original.
            texto_formatado_lote = "\n".join(
                f"{row.Data} | {row['Histórico']} | {row['Valor']} | Tipo: {row['Tipo']}"
                for _, row in lote_df.iterrows()
            )

            # -------------------------------------------------
            # NOVO PROMPT ALTAMENTE ESTRUTURADO PARA CLASSIFICAÇÃO DFC/CPC 03
            # -------------------------------------------------
            prompt_lote = f"""
Você é um analista financeiro sênior da Hedgewise, especializado na composição da Demonstração de Fluxo de Caixa (DFC) conforme o CPC 03 (IAS 7).

Sua tarefa é analisar AS {len(lote_df)} MOVIMENTAÇÕES BANCÁRIAS extraídas e retornar um JSON estritamente conforme o schema fornecido.

**Instruções de Classificação (Obrigatórias):**

1.  **natureza_geral** (Grupo): Classifique estritamente como **"Receita"** ou **"Despesa"**. (Observar se o Tipo original é 'C'rédito ou 'D'ébito, mas sempre priorizar o significado da transação).
2.  **subgrupo** (DFC/CPC 03): Classifique estritamente em uma das quatro opções:
    * **"Operacional"**: Transações que afetam o resultado e o capital de giro (vendas, compras, salários, aluguéis, impostos, fornecedores, etc.).
    * **"Investimento"**: Aquisição ou venda de ativos não circulantes (imóveis, máquinas, participações societárias).
    * **"Financiamento"**: Transações com capital de terceiros ou próprio (empréstimos, integralização/distribuição de capital, dividendos).
    * **"Pessoal"**: Despesas pessoais do sócio/empreendedor pagas pela conta da empresa (retiradas, despesas particulares, etc.).
3.  **natureza_analitica** (Subgrupo Detalhado):
    * Identifique o destino/origem de forma detalhada e linear.
    * **REGRA DE PREENCHIMENTO:** Se o histórico for genérico (ex: "Pagamento de Boleto", "Transferência TED", "Pix") e não houver informação clara, assuma **"Fornecedores"** ou **"Despesas Gerais Operacionais"** se for um débito, e **"Vendas/Serviços"** se for um crédito, pois a premissa é que a conta é empresarial.
4.  **natureza_juridica**: Classifique estritamente como **"Empresarial"** ou **"Pessoal"**.

Responda APENAS com o JSON.

Movimentações extraídas:
{texto_formatado_lote}
            """
            
            # Atualiza o progresso e envia o lote
            progress_bar.progress((i + 1) / n_batches, text=f"Processando lote {i+1} de {n_batches} ({len(lote_df)} transações)...")
            
            # Chamada à API
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_lote,
                config=config,
            )

            resposta_texto = response.text

            # Processar e armazenar o JSON do lote
            try:
                dados_lote = json.loads(resposta_texto)
                if isinstance(dados_lote, list):
                    dados_classificados_totais.extend(dados_lote)
                else:
                    st.warning(f"Lote {i+1}: Retorno JSON inesperado. O resultado deste lote foi ignorado.")
            except json.JSONDecodeError as e:
                st.error(f"Lote {i+1} falhou ao decodificar JSON. Certifique-se de que o modelo produziu um JSON válido.")
                st.text_area(f"Resposta bruta do Lote {i+1}:", resposta_texto, height=150)
        
        progress_bar.empty()
        st.success("✅ Classificação de todas as movimentações concluída com sucesso!")

    except Exception as e:
        st.error(f"Erro ao conectar com a API do Google Gemini ou durante o processamento: {e}")
        st.stop()

    # -------------------------------------------------
    # Exibir Resultado Final
    # -------------------------------------------------
    st.subheader("📊 Resultado Final da IA (Extrato Classificado - DFK/CPC 03)")

    if dados_classificados_totais:
        st.json(dados_classificados_totais)

        st.subheader("Tabela Classificada Completa")
        df_classificado = pd.DataFrame(dados_classificados_totais)
        
        # Reorganiza as colunas para melhor visualização
        colunas_ordenadas = [
            'data', 'historico', 'valor', 'tipo', 
            'natureza_geral', 'subgrupo', 'natureza_analitica', 'natureza_juridica'
        ]
        df_classificado = df_classificado[colunas_ordenadas]
        
        st.dataframe(df_classificado, use_container_width=True)
    else:
        st.warning("Nenhum dado classificado foi retornado. Verifique os erros acima.")
