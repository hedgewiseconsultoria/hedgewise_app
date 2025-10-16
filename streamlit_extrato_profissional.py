import os
import re
import json
import streamlit as st
from io import BytesIO
from openai import OpenAI
import extrato_parser  # seu módulo externo de leitura de extratos

# ---------------------------------------------
# Função: extrair texto de PDF (via extrato_parser)
# ---------------------------------------------
def extrair_texto_pdf(pdf_bytes):
    try:
        pdf_file = BytesIO(pdf_bytes)
        texto = extrato_parser.extrair_texto_pdf(pdf_file)
        return texto
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {e}")
        return ""

# ---------------------------------------------
# Configuração do Streamlit
# ---------------------------------------------
st.set_page_config(page_title="Hedgewise - Extrato Profissional", layout="wide")
st.title("💼 Análise de Extrato Bancário com Llama 3.1 (via Hugging Face Router)")

# Upload do PDF
uploaded_file = st.file_uploader("Envie o extrato bancário em PDF", type=["pdf"])

# ---------------------------------------------
# Processo principal
# ---------------------------------------------
if uploaded_file:
    bytes_pdf = uploaded_file.read()
    st.info("Extraindo texto do PDF…")
    texto = extrair_texto_pdf(bytes_pdf)

    if not texto.strip():
        st.error("Não foi possível extrair texto. Verifique o PDF.")
        st.stop()

    st.success("Texto extraído com sucesso.")
    st.caption(f"📝 Texto extraído: {len(texto)} caracteres")

    with st.expander("📄 Ver texto extraído do PDF"):
        st.text_area("Conteúdo extraído:", texto, height=300)

    # -------------------------------------------------
    # Processar texto com o parser (detectar banco e ler transações)
    # -------------------------------------------------
    banco = extrato_parser.detectar_banco(texto)
    processador = extrato_parser.PROCESSADORES.get(banco, extrato_parser.processar_extrato_generico)
    transacoes = processador(texto)
    df = extrato_parser.normalizar_transacoes(transacoes)

    if df.empty:
        st.warning("Não foi possível identificar transações válidas no extrato.")
        st.stop()

    st.subheader("📑 Transações detectadas automaticamente")
    st.dataframe(df)

    # -------------------------------------------------
    # Construir o prompt para a IA
    # -------------------------------------------------
    prompt = f"""
Você é um analista financeiro da Hedgewise.
Analise as movimentações bancárias abaixo e retorne **somente um JSON válido** com as chaves:
data, descricao, valor, tipo (Receita ou Despesa), categoria e natureza (Pessoal ou Empresarial).

Movimentações extraídas:
{df.to_json(orient='records', force_ascii=False)}
    """

    with st.expander("🔎 Ver prompt enviado ao modelo"):
        st.text_area("Prompt:", prompt, height=300)

    # -------------------------------------------------
    # Envio ao modelo Llama 3.1 via router.huggingface.co
    # -------------------------------------------------
    st.info("Analisando com IA (Llama 3.1 - Hugging Face Router)…")

    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
        )

        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:novita",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )

        result_text = completion.choices[0].message.content
        st.subheader("📊 Resposta da IA")

        # -------------------------------------------------
        # Tentativa de interpretar o JSON retornado
        # -------------------------------------------------
        try:
            json_inicio = result_text.find("[")
            json_fim = result_text.rfind("]") + 1
            json_text = result_text[json_inicio:json_fim]
            dados = json.loads(json_text)
            st.json(dados)
        except Exception:
            st.warning("⚠️ Falha ao interpretar o JSON retornado.")
            st.text_area("Resposta completa da IA:", result_text, height=300)

    except Exception as e:
        st.error(f"Erro ao chamar a API: {e}")
