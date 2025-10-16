import os
import json
import streamlit as st
from huggingface_hub import InferenceClient
from io import BytesIO

# Importa o seu módulo completo de parsing
from extrato_parser import extrair_texto_pdf, detectar_banco, PROCESSADORES, normalizar_transacoes

# ---------------------------------------------
# Configuração do Streamlit
# ---------------------------------------------
st.set_page_config(page_title="Hedgewise - Extrato Profissional", layout="wide")
st.title("💼 Análise de Extrato Bancário com Llama 3.1")

# Upload do PDF
uploaded_file = st.file_uploader("Envie o extrato bancário em PDF", type=["pdf"])

# ---------------------------------------------
# Processo principal
# ---------------------------------------------
if uploaded_file:
    bytes_pdf = uploaded_file.read()
    st.info("Extraindo texto do PDF…")

    try:
        # Usa seu extrator robusto
        texto = extrair_texto_pdf(BytesIO(bytes_pdf))
    except Exception as e:
        st.error(f"Erro ao processar PDF: {e}")
        st.stop()

    if not texto or len(texto.strip()) < 50:
        st.error("Não foi possível extrair texto legível do PDF.")
        st.stop()

    st.success(f"Texto extraído com sucesso ({len(texto)} caracteres).")
    with st.expander("📄 Ver texto extraído do PDF"):
        st.text_area("Conteúdo extraído:", texto, height=300)

    # ---------------------------------------------
    # Detectar banco e processar transações
    # ---------------------------------------------
    banco = detectar_banco(texto)
    st.subheader(f"🏦 Banco detectado: {banco}")

    processador = PROCESSADORES.get(banco, PROCESSADORES["DESCONHECIDO"])
    transacoes = processador(texto)
    df = normalizar_transacoes(transacoes)

    if df.empty:
        st.error("Não foram encontradas movimentações válidas neste extrato.")
        st.stop()

    st.success(f"{len(df)} transações detectadas com sucesso!")
    st.dataframe(df, use_container_width=True)

    # Mostra JSON que será enviado à IA
    json_transacoes = df.to_json(orient="records", force_ascii=False, indent=2)
    with st.expander("🧾 Ver dados estruturados extraídos do extrato"):
        st.code(json_transacoes, language="json")

    # ---------------------------------------------
    # Prompt para IA (Llama 3.1 via Hugging Face)
    # ---------------------------------------------
    prompt = f"""
Você é um analista financeiro da Hedgewise.
Analise as movimentações financeiras listadas abaixo e retorne **somente** um JSON válido,
seguindo o formato abaixo. Não adicione explicações, apenas o JSON puro.

Formato esperado:
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

Movimentações:
{json_transacoes}
    """

    with st.expander("🔎 Ver prompt enviado ao modelo"):
        st.text_area("Prompt:", prompt, height=300)

    # ---------------------------------------------
    # Envio ao modelo Llama 3.1 (Hugging Face Inference)
    # ---------------------------------------------
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

    # ---------------------------------------------
    # Tentativa de interpretar o JSON retornado
    # ---------------------------------------------
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
