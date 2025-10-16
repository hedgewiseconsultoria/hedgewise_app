import streamlit as st
import os
from huggingface_hub import InferenceClient

# -----------------------------
# Configuração da página
# -----------------------------
st.set_page_config(page_title="Teste Llama 3.1", layout="centered")
st.title("🧠 Teste de Conexão com Llama 3.1 (Hugging Face)")

# -----------------------------
# Token da Hugging Face (configure no Secrets do Streamlit Cloud)
# -----------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

if not HF_TOKEN:
    st.error("❌ Token HF_TOKEN não encontrado. Configure-o em 'Settings → Secrets' no Streamlit Cloud.")
    st.stop()

# -----------------------------
# Inicializa o cliente Hugging Face
# -----------------------------
try:
    client = InferenceClient(
        model="meta-llama/Llama-3.1-8B",
        token=HF_TOKEN,
    )
    st.success("✅ Conexão com o modelo Llama 3.1 inicializada com sucesso!")
except Exception as e:
    st.error(f"Erro ao inicializar o cliente da Hugging Face: {e}")
    st.stop()

# -----------------------------
# Entrada de texto do usuário
# -----------------------------
prompt = st.text_area("Digite um prompt para testar:", "Explique o que é inteligência artificial em poucas linhas.")

if st.button("Enviar para Llama 3.1 🚀"):
    with st.spinner("Gerando resposta..."):
        try:
            result = client.text_generation(
                prompt,
                max_new_tokens=400,
                temperature=0.7,
            )
            st.subheader("🧾 Resposta do modelo:")
            st.write(result)
        except Exception as e:
            st.error(f"Erro ao chamar a API da Hugging Face: {e}")


