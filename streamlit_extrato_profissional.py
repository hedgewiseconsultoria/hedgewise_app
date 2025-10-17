import os
import json
import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from io import BytesIO

# IMPORTAÇÕES ESSENCIAIS DO SEU CÓDIGO ORIGINAL
# Você deve garantir que 'extrato_parser.py' está disponível e contém:
# - extrair_texto_pdf
# - detectar_banco
# - PROCESSADORES (dicionário de funções de processamento)
# - normalizar_transacoes
from extrato_parser import extrair_texto_pdf, detectar_banco, PROCESSADORES, normalizar_transacoes 

# -------------------------------------------------
# Configuração do Streamlit
# -------------------------------------------------
st.set_page_config(page_title="Hedgewise - Extrato Profissional", layout="wide")
st.title("💼 Análise de Extrato Bancário com Google Gemini")

uploaded_file = st.file_uploader("📎 Envie o extrato bancário em PDF", type=["pdf"])

# -------------------------------------------------
# 1. Extração de texto e estruturação
# -------------------------------------------------
if uploaded_file:
    st.info("Extraindo texto do PDF...")

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
    
    # -------------------------------------------------
    # 2. CONFIRMAÇÃO DO BANCO (Novo Passo)
    # -------------------------------------------------
    
    # Detectar banco automaticamente
    banco_detectado = detectar_banco(texto)
    
    bancos_disponiveis = list(PROCESSADORES.keys())
    
    # Tenta pré-selecionar o banco detectado
    try:
        index_selecionado = bancos_disponiveis.index(banco_detectado)
    except ValueError:
        index_selecionado = 0 # Seleciona o primeiro da lista se o detectado não existir

    st.success(f"🏦 Banco detectado automaticamente: {banco_detectado}")

    # Permite ao usuário confirmar ou ajustar
    banco_confirmado = st.selectbox(
        "**Confirme ou ajuste o banco para o processamento das transações:**",
        options=bancos_disponiveis,
        index=index_selecionado,
        key="bank_selector"
    )

    if not st.button(f"Processar Transações ({banco_confirmado})"):
        st.stop()
        
    # Processar transações com o parser correto (baseado na confirmação)
    processador = PROCESSADORES.get(banco_confirmado, PROCESSADORES["DESCONHECIDO"])
    transacoes = processador(texto)
    
    try:
        df_transacoes = normalizar_transacoes(transacoes)
    except Exception as e:
        st.error(f"Erro ao normalizar transações: {e}")
        st.stop()

    if df_transacoes.empty:
        st.warning("Não foi possível identificar movimentações financeiras válidas neste PDF.")
        st.stop()
        
    if 'Tipo' not in df_transacoes.columns:
        st.error("A coluna 'Tipo' (Despesa/Receita) não foi gerada na normalização. O processamento da IA não funcionará corretamente.")
        st.stop()

    st.subheader(f"Transações Extraídas ({len(df_transacoes)})")
    st.dataframe(df_transacoes, use_container_width=True)

    # -------------------------------------------------
    # 3. CONFIGURAÇÕES DE LOTE E ESTRUTURA PARA GEMINI
    # -------------------------------------------------
    
    # Ajuste o tamanho do lote para otimizar a velocidade vs. número de chamadas.
    TAMANHO_DO_LOTE = 50 
    
    # Estrutura JSON esperada pelo modelo
    json_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "A data da transação."},
                "historico": {"type": "string", "description": "O histórico ou descrição original da transação."},
                "valor": {"type": "string", "description": "O valor original da transação."},
                "tipo": {"type": "string", "description": "Se é 'Despesa' ou 'Receita'."},
                "categoria": {"type": "string", "description": "Uma classificação detalhada da transação."},
                "natureza": {"type": "string", "description": "Classificação 'Pessoal' ou 'Empresarial'."}
            },
            "required": ["data", "historico", "valor", "tipo", "categoria", "natureza"]
        }
    }
    
    # Variável para armazenar todos os resultados JSON de todos os lotes
    dados_classificados_totais = []

    # Obter a chave API e inicializar o cliente
    API_KEY = os.getenv("GEMINI_API_KEY")

    if not API_KEY:
        st.error("Chave da API do Gemini (GEMINI_API_KEY) não configurada.")
        st.stop()
        
    st.info(f"Analisando o extrato com IA (Gemini 2.5 Flash) em {len(df_transacoes) // TAMANHO_DO_LOTE + 1} lotes...")

    try:
        client = genai.Client(api_key=API_KEY)

        # Configurações para a geração (Forçando JSON e desabilitando Thinking)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=json_schema,
            temperature=0.1, 
            # OTIMIZAÇÃO ADICIONAL: Desativa o raciocínio para classificação direta
            thinking_config=types.ThinkingConfig(thinking_budget=0) 
        )

        # Lógica para processamento em lotes (Chunking)
        n_batches = len(df_transacoes) // TAMANHO_DO_LOTE + (1 if len(df_transacoes) % TAMANHO_DO_LOTE > 0 else 0)
        progress_bar = st.progress(0, text="Iniciando o processamento dos lotes...")
        
        for i in range(n_batches):
            start_index = i * TAMANHO_DO_LOTE
            end_index = start_index + TAMANHO_DO_LOTE
            lote_df = df_transacoes.iloc[start_index:end_index]
            
            # Preparar texto para o lote atual
            texto_formatado_lote = "\n".join(
                f"{row.Data} | {row['Histórico']} | {row['Valor']} | {row['Tipo']}"
                for _, row in lote_df.iterrows()
            )

            # Prompt de análise para o lote
            prompt_lote = f"""
Você é um analista financeiro da Hedgewise. Sua função é classificar as transações bancárias.

Analise AS {len(lote_df)} MOVIMENTAÇÕES BANCÁRIAS abaixo e retorne um JSON estruturado.

Instruções:
1. 'data', 'historico', 'valor', 'tipo' devem conter os valores EXATOS da movimentação.
2. 'categoria' deve ser uma classificação detalhada e específica.
3. 'natureza' deve ser 'Pessoal' ou 'Empresarial'.

Responda APENAS com o JSON.

Movimentações extraídas:
{texto_formatado_lote}
            """
            
            # Atualiza o progresso e envia o lote
            progress_bar.progress((i + 1) / n_batches, text=f"Processando lote {i+1} de {n_batches} ({len(lote_df)} transações)...")
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_lote,
                config=config,
            )

            resposta_texto = response.text

            # Processar e armazenar o JSON do lote
            try:
                dados_lote = json.loads(resposta_texto)
                if isinstance(dados_lote, list):
                    dados_classificados_totais.extend(dados_lote)
                else:
                    st.warning(f"Lote {i+1}: Retorno JSON inesperado (não é uma lista). O resultado deste lote foi ignorado.")
            except json.JSONDecodeError as e:
                st.error(f"Lote {i+1} falhou ao decodificar JSON. O extrato deste lote pode ter sido corrompido.")
                st.text_area(f"Resposta bruta do Lote {i+1}:", resposta_texto, height=150)
        
        progress_bar.empty() # Remove a barra de progresso após o término
        st.success("✅ Classificação de todas as movimentações concluída com sucesso!")

    except Exception as e:
        st.error(f"Erro ao conectar com a API do Google Gemini ou durante o processamento: {e}")
        st.stop()

    # -------------------------------------------------
    # Exibir Resultado Final
    # -------------------------------------------------
    st.subheader("📊 Resultado Final da IA (Extrato Classificado)")

    if dados_classificados_totais:
        # Exibir JSON
        st.json(dados_classificados_totais)

        # Exibir como DataFrame
        st.subheader("Tabela Classificada Completa")
        df_classificado = pd.DataFrame(dados_classificados_totais)
        st.dataframe(df_classificado, use_container_width=True)
    else:
        st.warning("Nenhum dado classificado foi retornado. Verifique os erros acima.")
