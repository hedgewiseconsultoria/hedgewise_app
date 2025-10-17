import os
import json
import streamlit as st
# Importar a biblioteca Google GenAI
from google import genai
from google.genai import types # Para configurar o Request
from extrato_parser import extrair_texto_pdf, detectar_banco, PROCESSADORES, normalizar_transacoes
from io import BytesIO
import pandas as pd # Adicionado importação do pandas, caso não esteja no seu ambiente

# -------------------------------------------------
# Configuração do Streamlit
# -------------------------------------------------
st.set_page_config(page_title="Hedgewise - Extrato Profissional", layout="wide")
# Atualizado o título para refletir o uso do Gemini
st.title("💼 Análise de Extrato Bancário com Google Gemini")

uploaded_file = st.file_uploader("📎 Envie o extrato bancário em PDF", type=["pdf"])

# -------------------------------------------------
# Etapa 1: Extração de texto e estruturação (Mantida)
# -------------------------------------------------
if uploaded_file:
    st.info("Extraindo e estruturando informações do extrato bancário…")

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

    # Detectar banco automaticamente
    banco = detectar_banco(texto)
    st.success(f"🏦 Banco detectado: {banco}")

    # Processar transações com o parser correto
    processador = PROCESSADORES.get(banco, PROCESSADORES["DESCONHECIDO"])
    transacoes = processador(texto)
    df_transacoes = normalizar_transacoes(transacoes)

    if df_transacoes.empty:
        st.warning("Não foi possível identificar movimentações financeiras válidas neste PDF.")
        st.stop()

    st.dataframe(df_transacoes, use_container_width=True)

    # -------------------------------------------------
    # Preparar texto limpo para o modelo Gemini
    # -------------------------------------------------
    # Garantir que a coluna 'Tipo' existe no DataFrame após a normalização
    if 'Tipo' not in df_transacoes.columns:
         st.error("A coluna 'Tipo' (Despesa/Receita) não foi gerada na normalização. O processamento da IA não funcionará corretamente.")
         st.stop()

    texto_formatado = "\n".join(
        f"{row.Data} | {row['Histórico']} | {row['Valor']} | {row['Tipo']}"
        for _, row in df_transacoes.iterrows()
    )

    # -------------------------------------------------
    # Prompt de análise
    # -------------------------------------------------
    # Definindo um esquema (schema) de resposta JSON
    # É uma boa prática para guiar o modelo Gemini na estruturação
    json_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "A data da transação."},
                "historico": {"type": "string", "description": "O histórico ou descrição original da transação."},
                "valor": {"type": "string", "description": "O valor original da transação."},
                "tipo": {"type": "string", "description": "Se é 'Despesa' ou 'Receita'."},
                "categoria": {"type": "string", "description": "Uma classificação detalhada da transação (ex: 'Salário', 'Aluguel', 'Supermercado', 'Investimento')."},
                "natureza": {"type": "string", "description": "Classificação 'Pessoal' ou 'Empresarial' baseada no histórico."}
            },
            "required": ["data", "historico", "valor", "tipo", "categoria", "natureza"]
        }
    }

    prompt = f"""
Você é um analista financeiro da Hedgewise. Sua função é classificar as transações bancárias.

Analise as movimentações bancárias abaixo e retorne um JSON estruturado seguindo o schema fornecido.

Instruções para o preenchimento:
1. 'data', 'historico', 'valor', 'tipo' devem conter os valores EXATOS da movimentação.
2. 'categoria' deve ser uma classificação detalhada.
3. 'natureza' deve ser 'Pessoal' ou 'Empresarial'.

Responda APENAS com o JSON.

Movimentações extraídas:
{texto_formatado}
    """

    with st.expander("🔎 Prompt enviado ao modelo"):
        st.text_area("Prompt:", prompt, height=300)

    # -------------------------------------------------
    # Envio ao modelo Google Gemini
    # -------------------------------------------------
    st.info("Analisando o extrato com IA (Google Gemini 2.5 Flash)…")

    # A chave da API do Gemini é lida automaticamente da variável de ambiente GEMINI_API_KEY
    # No Streamlit Cloud, adicione a variável GEMINI_API_KEY nas Secrets
    API_KEY = os.getenv("GEMINI_API_KEY")

    if not API_KEY:
        st.error("Chave da API do Gemini (GEMINI_API_KEY) não configurada. Adicione nas Secrets do Streamlit.")
        st.stop()

    try:
        # Inicializa o cliente do Gemini
        client = genai.Client(api_key=API_KEY)

        # Configurações para a geração (força a saída JSON)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=json_schema,
            temperature=0.1, # Temperatura baixa para resultados determinísticos/precisos
        )

        # Chama a API
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=config,
        )

        resposta_texto = response.text

    except Exception as e:
        st.error(f"Erro ao conectar com a API do Google Gemini: {e}")
        st.stop()

    # -------------------------------------------------
    # Exibir resultado e converter para JSON (mais fácil agora)
    # -------------------------------------------------
    st.subheader("📊 Resultado da IA (Classificação do Gemini)")

    try:
        # A API do Gemini com response_mime_type="application/json" retorna
        # um JSON puro, eliminando a necessidade de limpeza de string.
        dados = json.loads(resposta_texto)
        st.json(dados)

        # Exibir como DataFrame (opcional)
        st.subheader("Tabela Classificada")
        df_classificado = pd.DataFrame(dados)
        st.dataframe(df_classificado, use_container_width=True)


    except Exception as e:
        st.warning(f"⚠️ Falha ao interpretar o JSON retornado: {e}")
        st.text_area("Resposta completa da IA (não é um JSON válido):", resposta_texto, height=300)
