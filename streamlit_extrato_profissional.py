import streamlit as st
import pandas as pd
import pdfplumber
import json
import os
import base64
from io import BytesIO
import pytesseract
from pdf2image import convert_from_bytes
from huggingface_hub import InferenceClient

# --- Configuração da página ---
st.set_page_config(
    page_title="Hedgewise • Extrato Inteligente",
    page_icon="💼",
    layout="wide"
)

# --- Estilo profissional ---
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
    usar_ocr = st.checkbox("Ativar OCR (PDF escaneado)", value=False)
    st.markdown("Versão MVP • Llama 3.1 via Hugging Face")

# --- Cabeçalho ---
st.title("📊 Leitor de Extratos Inteligente")
st.write("Envie um extrato bancário em PDF e deixe o Llama 3.1 classificar automaticamente as movimentações.")

# --- Upload do PDF ---
uploaded_file = st.file_uploader("Selecione o arquivo PDF", type=["pdf"])

# --- Token e cliente do Hugging Face ---
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.warning("⚠️ O token do Hugging Face (HF_TOKEN) não está configurado. Adicione-o nas Secrets do Streamlit Cloud.")

client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_TOKEN,
)

# --- Funções auxiliares ---
def extrair_texto_pdf(pdf_bytes, usar_ocr=False):
    texto = ""
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for pagina in pdf.pages:
                t = pagina.extract_text()
                if t:
                    texto += t + "\n"
    except Exception as e:
        st.warning(f"⚠️ Não foi possível extrair texto com pdfplumber: {e}")
        texto = ""

    texto = texto.strip()

    if usar_ocr and not texto:
        try:
            st.info("Tentando OCR (pode demorar alguns segundos)...")
            imagens = convert_from_bytes(pdf_bytes)
            ocr_texto = ""
            for img in imagens:
                ocr_texto += pytesseract.image_to_string(img, lang="por") + "\n"
            texto = ocr_texto.strip()
            if not texto:
                st.warning("OCR concluído, mas não foi detectado texto.")
        except Exception as e:
            st.error(f"Erro ao realizar OCR: {e}")
            texto = ""

    if not texto:
        st.warning("❌ Nenhum texto encontrado no PDF. Verifique se o arquivo é legível.")
    return texto

def chamar_llama3_huggingface(prompt_text):
    try:
        result = client.text_generation(
            prompt_text,
            model="meta-llama/Llama-3.1-8B",
            max_new_tokens=1000,
            temperature=0.0,
        )
        return result
    except Exception as e:
        st.error(f"Erro na API Llama 3.1: {e}")
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
    texto = extrair_texto_pdf(bytes_pdf, usar_ocr=usar_ocr)

    if not texto:
        st.warning("Não foi possível extrair texto. Tente ativar o OCR.")
    else:
        st.success("Texto extraído com sucesso.")

        # --- Exibir o texto extraído ---
        with st.expander("📄 Ver texto extraído do PDF"):
            st.text_area("Conteúdo extraído:", texto, height=300)
            st.caption(f"📝 Texto extraído: {len(texto)} caracteres")

        # --- Limpeza básica para remover ruído e menus ---
        texto_limpo = []
        for linha in texto.splitlines():
            if any(palavra in linha.upper() for palavra in ["DATA", "HISTÓRICO", "DOCUMENTO", "VALOR", "SALDO", "TED", "PIX", "DEBITO", "CREDITO", "PAGAMENTO", "APLICACAO", "RESGATE"]):
                texto_limpo.append(linha)
            elif any(ch.isdigit() for ch in linha) and ("R$" in linha or "," in linha):
                texto_limpo.append(linha)
        texto = "\n".join(texto_limpo)

        # --- Checagem se contém dados de movimentação ---
        if "TED" not in texto and "PIX" not in texto and "SALDO" not in texto:
            st.error("O texto extraído não parece conter movimentações bancárias válidas. Verifique o PDF.")
            st.stop()

        # --- Prompt refinado ---
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

        # --- Chamada ao modelo ---
        with st.spinner("Processando com Llama 3.1 (Hugging Face)…"):
            resposta = chamar_llama3_huggingface(prompt)

        # --- Extrair apenas o JSON ---
        if resposta:
            import re
            match = re.search(r"\[.*\]", resposta, re.DOTALL)
            if match:
                resposta = match.group(0)

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
                st.download_button(
                    label="📥 Baixar resultado em CSV",
                    data=csv,
                    file_name="resultado_hedgewise.csv",
                    mime="text/csv"
                )
            else:
                st.error("Falha ao interpretar o JSON da resposta.")
