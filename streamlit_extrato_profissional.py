import os
import json
import streamlit as st
from io import BytesIO
from huggingface_hub import InferenceClient
from extrato_parser import extrair_texto_pdf, detectar_banco, PROCESSADORES  # usa seu parser

# ---------------------------------------------
# Configuração do Streamlit
# ---------------------------------------------
st.set_page_config(page_title="Hedgewise - Extrato Profissional", layout="wide")
st.title("💼 Análise de Extrato Bancário com Llama 3.1")

# Upload do PDF
uploaded_file = st.file_uploader("Envie o extrato bancário em PDF", type=["pdf"])
usar_ocr = st.checkbox("Ativar OCR (para PDF escaneado)", value=False)

# ---------------------------------------------
# Execução principal
# ---------------------------------------------
if uploaded_file:
    bytes_pdf = uploaded_file.read()

    st.info("Extraindo texto do PDF…")
    try:
        texto = extrair_texto_pdf(BytesIO(bytes_pdf))
    except Exception as e:
        st.error(f"Erro ao processar PDF: {e}")
        st.stop()

    if not texto.strip():
        st.error("Nenhum texto foi extraído. Tente ativar o OCR.")
        st.stop()

    st.success("Texto extraído com sucesso.")
    st.caption(f"📝 Texto extraído: {len(texto)} caracteres")

    with st.expander("📄 Ver texto extraído do PDF"):
        st.text_area("Conteúdo extraído:", texto, height=300)

    # ---------------------------------------------
    # Detecta banco automaticamente
    # ---------------------------------------------
    banco = detectar_banco(texto)
    st.info(f"🏦 Banco detectado: **{banco}**")

    processador = PROCESSADORES.get(banco, PROCESSADORES["DESCONHECIDO"])
    transacoes = processador(texto)

    if not transacoes or len(transacoes) < 3:
        st.warning("Não foram encontradas movimentações válidas no extrato.")
    else:
        st.success(f"{len(transacoes)} movimentações detectadas.")
        st.dataframe(transacoes)

    # ---------------------------------------------
    # Monta texto consolidado para a IA
    # ---------------------------------------------
    texto_ia = "\n".join(
        f"{t['Data']} | {t['Histórico']} | {t['Valor']} | {t['Tipo']}"
        for t in transacoes
    )

    prompt = f"""
Você é um analista financeiro da Hedgewise.
Analise as movimentações abaixo e retorne **somente** um JSON válido.

As chaves devem ser:
- data
- descricao
- valor
- tipo ("Receita" ou "Despesa")
- categoria
- natureza ("Pessoal" ou "Empresarial")

Exemplo:
[
  {{
    "data": "2024-01-05",
    "descricao": "Pagamento fornecedor",
    "valor": -1200.50,
    "tipo": "Despesa",
    "categoria": "Serviços",
    "natureza": "Empresarial"
  }}
]

Movimentações extraídas:
{texto_ia}
    """

    with st.expander("🔎 Ver prompt enviado ao modelo"):
        st.text_area("Prompt:", prompt, height=300)

    # ---------------------------------------------
    # Envio ao modelo Llama 3.1 (Hugging Face)
    # ---------------------------------------------
    st.info("Enviando dados para o Llama 3.1…")

    client = InferenceClient(
    model="meta-llama/Meta-Llama-3.1-8B",
    token=os.environ["HF_TOKEN"],
    )


    try:
        result = client.text_generation(
            prompt,
            model="meta-llama/Llama-3.1-8B",
            max_new_tokens=2000,
            temperature=0.2,
        )
    except Exception as e:
        st.error(f"Erro ao conectar com a API da Hugging Face: {e}")
        st.stop()

    # ---------------------------------------------
    # Interpreta a resposta JSON
    # ---------------------------------------------
    st.subheader("📊 Resultado da IA")

    try:
        json_inicio = result.find("[")
        json_fim = result.rfind("]") + 1
        json_text = result[json_inicio:json_fim]
        dados = json.loads(json_text)
        st.json(dados)
    except Exception as e:
        st.warning("⚠️ Falha ao interpretar o JSON retornado.")
        st.text_area("Resposta completa da IA:", result, height=300)



