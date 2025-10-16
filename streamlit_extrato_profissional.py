import os
import streamlit as st
from openai import OpenAI

# ---------------------------------------------
# Configuração do Streamlit
# ---------------------------------------------
st.set_page_config(page_title="Teste Llama 3.1 - Hugging Face (Novita)", layout="centered")
st.title("🦙 Teste de Conexão com Llama 3.1 via Hugging Face / Novita")

# Token da Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("❌ Variável de ambiente HF_TOKEN não encontrada. Configure no Secrets do Streamlit Cloud.")
    st.stop()

# Inicializa o cliente OpenAI apontando para o roteador Hugging Face
try:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )
    st.success("✅ Cliente inicializado com sucesso (via Novita / Hugging Face).")
except Exception as e:
    st.error(f"Erro ao inicializar o cliente: {e}")
    st.stop()

# Campo de texto para o prompt
prompt = st.text_area("Digite seu prompt para o Llama 3.1:", "Qual é a capital da França?")

# Botão para enviar à IA
if st.button("🚀 Enviar para o Llama 3.1"):
    st.info("Chamando o modelo Llama 3.1… aguarde.")
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:novita",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        st.subheader("🧩 Resposta da IA:")
        st.write(completion.choices[0].message["content"])

    except Exception as e:
        st.error(f"Erro ao chamar a API: {e}")
