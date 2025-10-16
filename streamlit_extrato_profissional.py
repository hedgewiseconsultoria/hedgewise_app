import os
import re
import json
import streamlit as st
from io import BytesIO
from huggingface_hub import InferenceClient

# Importa o seu parser de extratos
from extrato_parser import extrair_texto_pdf

# ---------------------------------------------
# Configuração do Streamlit
# ---------------------------------------------
st.set_page_config(page_title="Hedgewise - Extrato Profissional", layout="wide")
st.title("💼 Análise de Extrato Bancário com Llama 3.1")

# ---------------------------------------------
# Configuração do Hugging Face / Llama 3.1
# ---------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("⚠️ O token da Hugging Face (HF_TOKEN) não foi configurado no ambiente.")
    st.stop()

# Inicializa o cliente Hugging Face
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B",
    token=HF_TOKEN,
)

def chamar_llama3_huggingface(prompt_text):
    """
    Envia o prompt para o modelo Llama 3.1 via Hugging Face Inference API.
    Retorna o texto gerado pela IA.
    """
    try:
        result = client.text_generation(
            prompt_text,
            max_new_tokens=2000,
            temperature=0.2,
        )
        return result
    except Exception as e:
        st.error(f"Erro ao conectar com a API da Hugging Face: {e}")
        return ""

# ---------------------------------------------
# Upload do PDF
# ---------------------------------------------
uploaded_file = st.file_uploader("Envie o extrato bancário em PDF", type=["pdf"])
usar_ocr = st.checkbox("Ativar OCR (para PDF escaneado)", value=False)

# ---------------------------------------------
# Processo principal
# ---------------------------------------------
if uploaded_file:
    bytes_pdf = uploaded_file.read()
    st.info("Extraindo texto do PDF…")

    try:
        texto = extrair_texto_pdf(BytesIO(bytes_pdf))
    except Exception as e:
        st.error(f"Erro ao processar o PDF: {e}")
        st.stop()

    if not texto or len(texto.strip()) < 30:
        st.error("Não foi possível extrair texto. Tente ativar o OCR.")
        st.stop()

    st.success("Texto extraído com sucesso.")
    st.caption(f"📝 Texto extraído: {len(texto)} caracteres")

    with st.expander("📄 Ver texto extraído do PDF"):
        st.text_area("Conteúdo extraído:", texto, height=300)

    # -------------------------------------------------
    # Prompt para IA (Llama 3.1 via Hugging Face)
    # -------------------------------------------------
    prompt = f"""
Você é um analista financeiro da Hedgewise.
Extraia do texto abaixo **somente as movimentações financeiras** do extrato bancário.
Ignore cabeçalhos, rodapés e textos institucionais.
Retorne **apenas um JSON válido**, no seguinte formato:

[
  {{
    "data": "AAAA-MM-DD",
    "descricao": "texto descritivo da movimentação",
    "valor": -1000.00,
    "tipo": "Despesa ou Receita",
    "categoria": "Categoria resumida (ex: Tarifa, Transferência, Pagamento, Investimento, etc.)",
    "natureza": "Pessoal ou Empresarial"
  }}
]

Extrato bancário:
{texto}
    """

    with st.expander("🔎 Ver prompt enviado ao modelo"):
        st.text_area("Prompt:", prompt, height=300)

    # -------------------------------------------------
    # Envio ao modelo Llama 3.1 (Hugging Face Inference)
    # -------------------------------------------------
    st.info("Analisando o extrato com IA (Llama 3.1)…")

    result = chamar_llama3_huggingface(prompt)

    # -------------------------------------------------
    # Tentativa de interpretar o JSON retornado
    # -------------------------------------------------
    st.subheader("📊 JSON retornado pela IA")

    try:
        json_inicio = result.find("[")
        json_fim = result.rfind("]") + 1
        json_text = result[json_inicio:json_fim]
        dados = json.loads(json_text)
        st.json(dados)
    except Exception as e:
        st.warning("Falha ao interpretar o JSON da resposta.")
        st.text_area("Resposta completa da IA:", result, height=300)
