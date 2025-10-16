import streamlit as st
import requests
import pandas as pd
import pdfplumber
import json
import os
import base64
from io import BytesIO
from pdf2image import convert_from_path
import pytesseract

# --- Configuração da página ---
st.set_page_config(
    page_title="Hedgewise • Extrato Inteligente",
    page_icon="💼",
    layout="wide"
)

# --- Estilo profissional (baseado na logo) ---
st.markdown("""
    <style>
    body {
        background-color: #f6f8fa;
        color: #111827;
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    h1, h2, h3, h4 {
        color: #0A2342 !important;
        letter-spacing: -0.5px;
    }
    .stButton>button {
        background-color: #004AAD;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #003580;
        transform: translateY(-1px);
    }
    .metric-label {
        color: #374151 !important;
    }
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("logo_hedgewise.png", width=180)
    st.markdown("### Hedgewise")
    st.caption("Risco Controlado • Inteligência Financeira")
    st.markdown("---")
    use_ocr = st.checkbox("Ativar OCR (PDF escaneado)", value=False)
    st.markdown("Versão MVP • Llama 3 via Hugging Face")

# --- Cabeçalho ---
st.title("📊 Leitor de Extratos Inteligente")
st.write("Envie um extrato bancário em PDF e deixe o Llama 3 classificar automaticamente as movimentações.")

# --- Upload do PDF ---
uploaded_file = st.file_uploader("Selecione o arquivo PDF", type=["pdf"])

# --- Token e endpoint do Hugging Face ---
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- Funções auxiliares ---

def extrair_texto_pdf(pdf_bytes, usar_ocr=False):
    texto = ""
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for pagina in pdf.pages:
            t = pagina.extract_text()
            if t:
                texto += t + "\n"
    texto = texto.strip()

    if usar_ocr and not texto:
        imagens = convert_from_path(BytesIO(pdf_bytes))
        for img in imagens:
            texto += pytesseract.image_to_string(img, lang="por") + "\n"

    return texto.strip()

def chamar_llama3_huggingface(prompt_text):
    payload = {
        "inputs": prompt_text,
        "parameters": {"max_new_tokens": 1000, "temperature": 0.0}
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        # a resposta pode vir em formato de lista ou dict
        if isinstance(data, list):
            return data[0].get("generated_text", "")
        return data.get("generated_text", "")
    except Exception as e:
        st.error(f"Erro na API Hugging Face: {e}")
        return ""

def parse_json_resposta(texto_json):
    try:
        arr = json.loads(texto_json)
        return pd.DataFrame(arr)
    except Exception:
        return None

# --- Processamento principal ---
if uploaded_file:
    bytes_pdf = uploaded_file.read()
    st.info("Extraindo texto do PDF…")
    texto = extrair_texto_pdf(bytes_pdf, use_ocr=use_ocr)

    if not texto:
        st.warning("Não foi possível extrair texto. Tente ativar OCR.")
    else:
        st.success("Texto extraído com sucesso.")
        prompt = f"""
Você é um assistente financeiro da Hedgewise.
Analise o extrato abaixo e devolva **somente** um JSON estruturado com as chaves:
data, descricao, valor, tipo (Receita ou Despesa), categoria, natureza (Pessoal ou Empresarial).

Extrato:
{texto}
"""
        with st.spinner("Processando com Llama 3 (Hugging Face)…"):
            resposta = chamar_llama3_huggingface(prompt)

        if not resposta:
            st.error("Não houve resposta do modelo.")
        else:
            st.subheader("JSON retornado pela IA")
            st.code(resposta, language="json")

            df = parse_json_resposta(resposta)
            if df is not None and not df.empty:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Saldo líquido", f"R$ {df['valor'].sum():,.2f}")
                col2.metric("Total Pessoal", f"R$ {df[df['natureza'] == 'Pessoal']['valor'].sum():,.2f}")
                col3.metric("Total Empresarial", f"R$ {df[df['natureza'] == 'Empresarial']['valor'].sum():,.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                csv = df.to_csv(index=False).encode("utf-8")
                b64 = base64.b64encode(csv).decode()
                st.download_button(
                    label="📥 Baixar resultado em CSV",
                    data=csv,
                    file_name="resultado_hedgewise.csv",
                    mime="text/csv"
                )
            else:
                st.error("Falha ao interpretar o JSON da resposta.")
