import os
import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Teste Llama 3.1 - Provider Novita", layout="centered")
st.title("🦙 Teste de Conexão com Llama 3.1 via Hugging Face (Provider Novita)")

# --- Token da Hugging Face ---
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("❌ Variável de ambiente HF_TOKEN não encontrada. Configure-a nos Secrets do Streamlit Cloud.")
    st.stop()

# --- Inicializa o cliente com provider novita ---
try:
    client = InferenceClient(
        provider="novita",
        api_key=HF_TOKEN,
    )
    st.success("✅ Cliente inicializado com sucesso (provider=novita).")
except Exception as e:
    st.error(f"Erro ao inicializar o cliente da Hugging Face: {e}")
    st.stop()

# --- Prompt de entrada ---
prompt = st.text_area("Digite sua pergunta:", "Qual é a capital da França?")

# --- Botão para enviar ---
if st.button("🚀 Enviar para o Llama 3.1"):
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        resposta = completion.choices[0].message["content"]
        st.subheader("🧠 Resposta do modelo:")
        st.write(resposta)

    except Exception as e:
        st.error(f"Erro ao chamar a API: {e}")
