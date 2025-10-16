import streamlit as st
import os
from huggingface_hub import InferenceClient

# -----------------------------
# Configuração da página
# -----------------------------
st.set_page_config(page_title="Teste Llama 3.1", layout="centered")
st.title("🧠 Teste de Conexão com Llama 3.1 (Hugging Face)")

# -----------------------------
# Token da Hugging Face (configure em Secrets)
# -----------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("⚠️ Configure seu token Hugging Face em 'Settings → Secrets' no Streamlit Cloud (chave: HF_TOKEN).")
    st.stop()

# -----------------------------
# Inicializa o cliente Hugging Face
# -----------------------------
try:
    client = InferenceClient(token=HF_TOKEN)
    st.success("✅ Conexão com a API Hugging Face estabelecida!")
except Exception as e:
    st.error(f"Erro ao inicializar o cliente: {e}")
    st.stop()

# -----------------------------
# Entrada de prompt
# -----------------------------
prompt = st.text_area("✍️ Digite o prompt:", "Explique o que é inteligência artificial em poucas palavras.")

model = st.selectbox(
    "Escolha o modelo Llama 3.1:",
    [
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct"
    ],
)

if st.button("🚀 Enviar para Llama 3.1"):
    with st.spinner("Gerando resposta..."):
        try:
            result = client.text_generation(
                prompt,
                model=model,
                max_new_tokens=400,
                temperature=0.7,
            )
            st.subheader("🧾 Resposta do modelo:")
            st.write(result)
        except Exception as e:
            st.error(f"Erro ao chamar a API: {e}")
