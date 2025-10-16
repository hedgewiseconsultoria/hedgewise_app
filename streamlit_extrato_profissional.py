import streamlit as st
import requests
import pandas as pd
import pdfplumber
import json
import os
import base64
from io import BytesIO
from pdf2image import convert_from_bytes
import pytesseract

# ==============================
# CONFIGURAÇÃO DA PÁGINA
# ==============================
st.set_page_config(
    page_title="Hedgewise • Extrato Inteligente",
    layout="wide",
    page_icon=None
)

# ==============================
# ESTILO PERSONALIZADO
# ==============================
st.markdown("""
    <style>
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        background-color: #F7F8FA;
        color: #111827;
    }
    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    h1, h2, h3 {
        color: #0A2342 !important;
        letter-spacing: -0.3px;
        font-weight: 600 !important;
    }
    .stButton>button {
        background-color: #004AAD;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.4rem;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #003580;
        transform: translateY(-1px);
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
        margin-top: 1rem;
    }
    [data-testid="stMetricValue"] {
        color: #004AAD !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.image("logo_hedgewise.png", width=180)
    st.markdown("### Hedgewise")
    st.caption("Risco Controlado • Inteligência Financeira")
    st.markdown("---")
    use_ocr = st.checkbox("Ativar OCR (para PDFs escaneados)", value=False)
    st.markdown("---")
    st.markdown("<small>Versão MVP • Llama 3 via Hugging Face</small>", unsafe_allow_html=True)

# ==============================
# CABEÇALHO
# ==============================
st.title("Leitor Inteligente de Extratos Bancários")
st.markdown("Envie seu extrato em PDF para análise automática e categorização financeira baseada em IA.")

# ==============================
# TOKEN E ENDPOINT
# ==============================
HF_TOKEN = os.getenv("HF_TOKEN")
# ✅ endpoint correto
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3-8b-instruct"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# fallback – modelo alternativo leve
FALLBACK_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# ==============================
# FUNÇÕES AUXILIARES
# ==============================
def extrair_texto_pdf(pdf_bytes, usar_ocr=False):
    """Extrai texto de um PDF; usa OCR se necessário."""
    texto = ""
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for pagina in pdf.pages:
                t = pagina.extract_text()
                if t:
                    texto += t + "\n"
    except Exception as e:
        st.warning(f"Erro ao ler PDF com pdfplumber: {e}")

    texto = texto.strip()
    if usar_ocr and not texto:
        imagens = convert_from_bytes(pdf_bytes)
        for img in imagens:
            texto += pytesseract.image_to_string(img, lang="por") + "\n"
    return texto.strip()

def chamar_modelo(prompt_text):
    """Chama o modelo Llama 3 via API, com fallback automático."""
    payload = {"inputs": prompt_text,
               "parameters": {"max_new_tokens": 1000, "temperature": 0.0}}
    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.warning(f"Erro com Llama 3: {e}. Tentando fallback...")
        try:
            resp = requests.post(FALLBACK_URL, headers=HEADERS, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e2:
            st.error(f"Erro também no fallback: {e2}")
            return ""
    if isinstance(data, list):
        return data[0].get("generated_text", "")
    return data.get("generated_text", "")

def parse_json_resposta(texto_json):
    """Converte a resposta JSON do modelo em DataFrame."""
    try:
        arr = json.loads(texto_json)
        return pd.DataFrame(arr)
    except Exception:
        return None

# ==============================
# UPLOAD E PROCESSAMENTO
# ==============================
uploaded_file = st.file_uploader("Selecione o extrato bancário (PDF)", type=["pdf"])

if uploaded_file:
    bytes_pdf = uploaded_file.read()
    st.info("Extraindo texto do PDF...")
    texto = extrair_texto_pdf(bytes_pdf, usar_ocr=use_ocr)

    if not texto:
        st.warning("Não foi possível extrair texto. Tente ativar OCR.")
    else:
        st.success("Texto extraído com sucesso.")
        prompt = f"""
Você é um analista financeiro da Hedgewise.
Analise o extrato abaixo e devolva SOMENTE um JSON válido com os campos:
data, descricao, valor, tipo (Receita ou Despesa), categoria, natureza (Pessoal ou Empresarial).

Extrato:
{texto}
"""
        with st.spinner("Processando dados com IA..."):
            resposta = chamar_modelo(prompt)

        if not resposta:
            st.error("Nenhuma resposta obtida do modelo.")
        else:
            st.subheader("Resposta da IA (JSON)")
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
                st.download_button(
                    label="Baixar resultado em CSV",
                    data=csv,
                    file_name="resultado_hedgewise.csv",
                    mime="text/csv"
                )
            else:
                st.error("Falha ao interpretar o JSON retornado pelo modelo.")
