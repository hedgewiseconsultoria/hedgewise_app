import os
import streamlit as st
from huggingface_hub import InferenceClient

# ---------------------------------------------
# Configuração do Streamlit
# ---------------------------------------------
st.set_page_config(page_title="Teste Llama 3 - Hugging Face", layout="centered")
st.title("🦙 Teste de Conexão com Llama 3.1 via Hugging Face")

# Token do Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("❌ Variável de ambiente HF_TOKEN não encontrada. Configure o token no Secrets do Streamlit Cloud.")
    st.stop()

# Inicializa o cliente
try:
    client = InferenceClient(
    provider="novita",
    api_key=os.environ["HF_TOKEN"],
    )
    completion = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
      ],
    )

    print(completion.choices[0].message)
# Campo de texto para o prompt
prompt = st.text_area("Digite seu prompt para o Llama 3.1:", "Qual é a capital da França?")

# Botão para enviar à IA
if st.button("🚀 Enviar para o Llama 3.1"):
    st.info("Chamando o modelo Llama 3.1… aguarde.")
    try:
        result = client.text_generation(
            prompt,
            model="meta-llama/Llama-3.1-8B",
            max_new_tokens=200,
            temperature=0.3,
        )
        st.subheader("🧩 Resposta da IA:")
        st.write(result)
    except Exception as e:
        st.error(f"Erro ao chamar a API: {e}")

