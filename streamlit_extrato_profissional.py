import os
import re
import streamlit as st
from huggingface_hub import InferenceClient
from io import BytesIO
from pdfminer.high_level import extract_text
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFSyntaxError
from pdf2image import convert_from_bytes
import pytesseract

# ---------------------------------------------
# Função: extrair texto de PDF (com fallback OCR)
# ---------------------------------------------
def extrair_texto_pdf(pdf_bytes, usar_ocr=False):
    try:
        texto = extract_text(BytesIO(pdf_bytes))
        if texto and len(texto.strip()) > 30:
            return texto
        elif not usar_ocr:
            return ""
    except (PDFSyntaxError, PDFTextExtractionNotAllowed):
        if not usar_ocr:
            return ""

    if usar_ocr:
        imagens = convert_from_bytes(pdf_bytes)
        texto = "\n".join(pytesseract.image_to_string(img, lang="por") for img in imagens)
        return texto.strip()
    return ""

# ---------------------------------------------
# Configuração do Streamlit
# ---------------------------------------------
st.set_page_config(page_title="Hedgewise - Extrato Profissional", layout="wide")
st.title("💼 Análise de Extrato Bancário com Llama 3.1")

# Upload do PDF
uploaded_file = st.file_uploader("Envie o extrato bancário em PDF", type=["pdf"])
usar_ocr = st.checkbox("Ativar OCR (para PDF escaneado)", value=False)

# ---------------------------------------------
# Processo principal
# ---------------------------------------------
if uploaded_file:
    bytes_pdf = uploaded_file.read()
    st.info("Extraindo texto do PDF…")
    texto = extrair_texto_pdf(bytes_pdf, usar_ocr=usar_ocr)

    if not texto:
        st.error("Não foi possível extrair texto. Tente ativar o OCR.")
        st.stop()

    st.success("Texto extraído com sucesso.")
    st.caption(f"📝 Texto extraído: {len(texto)} caracteres")

    # Mostrar texto extraído
    with st.expander("📄 Ver texto extraído do PDF"):
        st.text_area("Conteúdo extraído:", texto, height=300)

    # -------------------------------------------------
    # Limpeza do texto para remover ruído e manter lançamentos
    # -------------------------------------------------
    texto_limpo = []
    for linha in texto.splitlines():
        if any(palavra in linha.upper() for palavra in [
            "DATA", "HISTÓRICO", "DOCUMENTO", "VALOR", "SALDO", 
            "TED", "PIX", "BOLETO", "CHEQUE", "DEPÓSITO", "FOLHA", "TARIFA"
        ]):
            texto_limpo.append(linha)
        elif re.search(r"\d{2}/\d{2}/\d{4}.*\d+,\d{2}", linha):
            texto_limpo.append(linha)
    texto = "\n".join(texto_limpo)

    # -------------------------------------------------
    # Checagem mais robusta de movimentações
    # -------------------------------------------------
    linhas_validas = re.findall(r"\d{2}/\d{2}/\d{4}.*\d+,\d{2}", texto)
    if len(linhas_validas) < 5:
        st.error("O texto extraído não parece conter movimentações bancárias válidas. Verifique o PDF.")
        st.stop()

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

    client = InferenceClient(
        provider="featherless-ai",
        api_key=os.environ["HF_TOKEN"],
    )

    result = client.text_generation(
        prompt,
        model="meta-llama/Llama-3.1-8B",
        max_new_tokens=2000,
        temperature=0.2,
    )

    # -------------------------------------------------
    # Tentativa de interpretar o JSON retornado
    # -------------------------------------------------
    import json
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
