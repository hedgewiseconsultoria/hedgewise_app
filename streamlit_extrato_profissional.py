import streamlit as st
import os
from huggingface_hub import InferenceClient

# --- Configuração da página ---
st.set_page_config(
    page_title="Teste IA - Hedgewise",
    page_icon="🤖",
    layout="centered"
)

st.title("🧠 Teste de Chamada da IA (Llama 3.1 via Hugging Face)")
st.write("Digite um prompt abaixo e veja a resposta gerada pelo modelo Llama 3.1.")

# --- Token do Hugging Face ---
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.warning("⚠️ O token do Hugging Face (HF_TOKEN) não está configurado. Adicione-o nas Secrets do Streamlit Cloud.")
else:
    client = InferenceClient(api_key=HF_TOKEN)

# --- Input do usuário ---
prompt = st.text_area("Prompt de teste:", "Explique o que é a teoria dos jogos em 3 frases curtas.")

# --- Botão para chamar a IA ---
if st.button("Chamar IA"):
    if not HF_TOKEN:
        st.error("Token Hugging Face ausente. Configure o HF_TOKEN para continuar.")
    else:
        with st.spinner("Chamando Llama 3.1..."):
            try:
                response = client.text_generation(
                    prompt,
                    model="meta-llama/Llama-3.1-8B",
                    max_new_tokens=500,
                    temperature=0.3,
                )
                st.success("✅ Resposta recebida!")
                st.markdown("### Resposta da IA:")
                st.write(response)
            except Exception as e:
                st.error(f"Erro ao chamar a IA: {e}")
