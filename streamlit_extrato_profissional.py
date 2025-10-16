import os
import json
import streamlit as st
from huggingface_hub import InferenceClient
from extrato_parser import extrair_texto_pdf, detectar_banco, PROCESSADORES, normalizar_transacoes
from io import BytesIO

# -------------------------------------------------
# Configuração do Streamlit
# -------------------------------------------------
st.set_page_config(page_title="Hedgewise - Extrato Profissional", layout="wide")
st.title("💼 Análise de Extrato Bancário com Mistral 7B Instruct v0.3")

uploaded_file = st.file_uploader("📎 Envie o extrato bancário em PDF", type=["pdf"])

# -------------------------------------------------
# Etapa 1: Extração de texto e estruturação
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
    # Preparar texto limpo para o modelo Mistral
    # -------------------------------------------------
    texto_formatado = "\n".join(
        f"{row.Data} | {row['Histórico']} | {row['Valor']} | {row['Tipo']}"
        for _, row in df_transacoes.iterrows()
    )

    # -------------------------------------------------
    # Prompt de análise
    # -------------------------------------------------
    prompt = f"""
Você é um analista financeiro da Hedgewise.

Analise as movimentações bancárias abaixo e retorne um JSON estruturado com as colunas:
- data
- histórico
- valor
- tipo (despesa ou receita)
- categoria (identificando do que se trata aquela movimentação)
- natureza (Pessoal ou Empresarial)

Responda apenas com o JSON.

Movimentações extraídas:
{texto_formatado}
    """

    with st.expander("🔎 Prompt enviado ao modelo"):
        st.text_area("Prompt:", prompt, height=300)

    # -------------------------------------------------
    # Envio ao modelo Mistral 7B Instruct v0.3
    # -------------------------------------------------
    st.info("Analisando o extrato com IA (Mistral 7B Instruct v0.3)…")

    HF_TOKEN = os.getenv("HF_TOKEN")

    if not HF_TOKEN:
        st.error("Token do Hugging Face (HF_TOKEN) não configurado. Adicione nas Secrets do Streamlit.")
        st.stop()

    try:
        client = InferenceClient(api_key=HF_TOKEN)

        result = client.chat_completion(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=[
                {"role": "system", "content": "Você é um analista financeiro experiente da Hedgewise."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.2,
        )

        resposta_texto = result.choices[0].message["content"]

    except Exception as e:
        st.error(f"Erro ao conectar com a API da Hugging Face: {e}")
        st.stop()

    # -------------------------------------------------
    # Exibir resultado e tentar converter para JSON
    # -------------------------------------------------
    st.subheader("📊 Resultado da IA")

    try:
        json_inicio = resposta_texto.find("[")
        json_fim = resposta_texto.rfind("]") + 1
        json_text = resposta_texto[json_inicio:json_fim]
        dados = json.loads(json_text)
        st.json(dados)
    except Exception as e:
        st.warning("⚠️ Falha ao interpretar o JSON. Veja a resposta completa abaixo:")
        st.text_area("Resposta completa da IA:", resposta_texto, height=300)
