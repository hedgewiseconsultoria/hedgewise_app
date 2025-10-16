import streamlit as st
import io
import pandas as pd
import json

# Importa as funções do seu módulo local
from extrato_parser import (
    extrair_texto_pdf,
    detectar_banco,
    PROCESSADORES,
    normalizar_transacoes
)

# ===========================================================
# CONFIGURAÇÃO DA PÁGINA
# ===========================================================
st.set_page_config(page_title="Leitor de Extratos Bancários Profissional", layout="wide")
st.title("📑 Extrator de Extratos Bancários Inteligente")

st.markdown("""
Este aplicativo extrai, interpreta e organiza dados de extratos bancários PDF de diversos bancos.
Envie o arquivo abaixo para iniciar o processamento.
""")

# ===========================================================
# UPLOAD DO ARQUIVO
# ===========================================================
uploaded_file = st.file_uploader("📎 Envie o extrato bancário (PDF)", type=["pdf"])

if uploaded_file:
    try:
        # ===========================================================
        # ETAPA 1 - LEITURA DO PDF
        # ===========================================================
        pdf_bytes = uploaded_file.read()
        pdf_file = io.BytesIO(pdf_bytes)

        st.info("🔍 Extraindo texto do PDF…")
        texto = extrair_texto_pdf(pdf_file)

        # Mostra o texto extraído (limitado para não travar o Streamlit)
        st.success(f"Texto extraído com sucesso. ({len(texto)} caracteres)")
        with st.expander("📄 Ver texto extraído do PDF"):
            st.text_area("Conteúdo extraído:", texto, height=500)

        # ===========================================================
        # ETAPA 2 - DETECTAR BANCO E PROCESSAR
        # ===========================================================
        banco = detectar_banco(texto)
        st.subheader(f"🏦 Banco detectado: {banco}")

        processador = PROCESSADORES.get(banco, PROCESSADORES["DESCONHECIDO"])
        transacoes = processador(texto)

        # ===========================================================
        # ETAPA 3 - NORMALIZAR E EXIBIR
        # ===========================================================
        df = normalizar_transacoes(transacoes)

        if not df.empty:
            st.success(f"✅ {len(df)} transações detectadas.")
            st.dataframe(df, use_container_width=True)

            # Mostrar JSON antes de enviar à IA
            json_texto = df.to_json(orient="records", force_ascii=False, indent=2)
            with st.expander("🧩 Ver JSON gerado para IA"):
                st.code(json_texto, language="json")

            # ===========================================================
            # ETAPA 4 - EXPORTAR OU ENVIAR À IA
            # ===========================================================
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "💾 Baixar CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "extrato_processado.csv",
                    "text/csv"
                )
            with col2:
                st.download_button(
                    "📘 Baixar JSON",
                    json_texto.encode("utf-8"),
                    "extrato_processado.json",
                    "application/json"
                )

        else:
            st.warning("⚠️ Nenhuma transação válida encontrada no extrato.")

    except Exception as e:
        st.error(f"❌ Erro ao processar o PDF: {e}")

else:
    st.info("⏳ Aguardando upload do arquivo PDF...")

