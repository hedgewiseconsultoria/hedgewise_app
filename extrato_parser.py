# ===========================================================
# Script definitivo de processamento de extratos bancários
# Adaptado para uso em produção com Streamlit
# Estratégia: Universal (Executa todos e escolhe o melhor resultado)
# ===========================================================

import re
import pandas as pd
import pdfplumber
import io

# ==================== FUNÇÕES UTILITÁRIAS ====================

def linha_parece_sujo(linha: str) -> bool:
    """
    Detecta linhas de rodapé / cabeçalho / totais / saldos.
    Preserva casos válidos como 'IOF', 'juros', 'tarifa', etc.
    """
    if not linha or not isinstance(linha, str):
        return True

    texto = linha.strip().lower()

    # --- exceções que DEVEM ser mantidas ---
    excecoes_regex = [
        r"\biof\b", r"\bjuros\b(?!\s+morat)", r"\btarifa\b", r"\bencargos\b",
        r"\btributo\b", r"\bimposto\b", r"\bpagamento\s+(de\s+)?(boleto|conta|fatura|darf|gps)\b",
        r"\btransfer[eê]ncia\s+(recebida|enviada|ted|doc)\b", r"\bpix\s+(recebido|enviado|emit|receb)\b",
        r"\bted\s+(recebid|enviad)\b", r"\bdoc\s+(recebid|enviad)\b",
        r"\bcheque\s+(compensado|devolvido)\b", r"\bc[oó]digo\s+\d+", r"\bdeb\s+conv\b",
        r"\bdeb\s+tit\b", r"\bdeb\s+parc\b"
    ]
    if any(re.search(p, texto, re.IGNORECASE) for p in excecoes_regex):
        return False

    # --- padrões de lixo reais ---
    padroes_lixo = [
        r"\bs\s*a\s*l\s*d\s*o\b",
        r"\bsaldo\s*(anterior|do\s+dia|total|atual|bloqueado|dispon[ií]vel|parcial|inicial|final|em\s+c/c)\b",
        r"\bsdo\s+(cta|apl|conta)\b",
        r"\bdetalhamento\b", r"\bextrato\b", r"\bcliente\b",
        r"\bconta\s+corrente\s*\|\s*movimenta", r"\blimite\b", r"\binvestimentos\b",
        r"\bdispon[ií]vel\b", r"\bbloqueado\b",

        # --- AJUSTE REFORÇADO PARA LINHAS DE TOTAL ---
        r"^\s*total\b.*(\d{1,3}(?:\.\d{3})*,\d{2}.*){2,}$",  # Ex: Total 73.165,98 -73.158,68 8,30
        r"^\s*total\b.*(cr[eé]dito|d[eé]bito|saldo)",        # Ex: Total Crédito/Débito/Saldo
        r"\btotal\s+geral\b",                                # Ex: Total Geral
        r"\btotal\s+das\s+opera[cç][õo]es\b",                # Ex: Total das operações

        r"\bresumo\b", r"\bfale\s*conosco\b", r"\bouvidoria\b", r"\bpara\s+demais\s+siglas\b",
        r"\bnotas\s+explicativas\b", r"\btotalizador\b", r"\baplicações\s+automáticas\b",
        r"\bvalor\s+\(r\$\)\b", r"\bdocumento\b", r"\bdescri[cç][aã]o\b", r"\bcr[eé]ditos\b",
        r"\bd[eé]bitos\b", r"\bmovimenta[cç][aã]o\b", r"\bp[aá]gina\b", r"\bdata\s+lan[cç]amento\b",
        r"\bcomplemento\b", r"\bcentral\s+de\s+suporte\b", r"\bconta\s+corrente\s*\|\s*movimenta[cç][aã]o\b",
        r"\bvalores\s+em\s+r\$\b", r"\bper[ií]odo\s+de\b", r"\bsaldo\s*\+\s*limite\b",
        r"\bcobran[cç]a\s+d[01]\b", r"\bcheque\s+empresarial\b", r"\bvencimento\s+cheque\b"
    ]
    return any(re.search(p, texto, re.IGNORECASE) for p in padroes_lixo)


def clean_value_str(s: str):
    """Limpa string monetária brasileira e retorna string numérica."""
    if not s: return None
    s = str(s).strip().replace("R$", "").replace("\u00A0", "").replace(" ", "").replace("-", "").replace(".", "").replace(",", ".")
    return s if re.match(r"^\d+(\.\d{1,2})?$", s) else None


def normalizar_transacoes(transacoes):
    """Padroniza saída em DataFrame com colunas Data | Histórico | Valor | Tipo."""
    df = pd.DataFrame(transacoes, columns=["Data", "Histórico", "Valor", "Tipo"])
    if df.empty: return df

    df["Valor_raw"] = df["Valor"].astype(str).apply(clean_value_str)
    df = df.dropna(subset=["Valor_raw"]).copy()
    try:
        df["Valor"] = df["Valor_raw"].astype(float)
    except ValueError:
        return pd.DataFrame()
    df["Tipo"] = df["Tipo"].astype(str).str.upper().map(lambda x: "D" if x.startswith("D") else "C")
    df['Data_dt'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Data_dt'])
    if df.empty:
        return pd.DataFrame(columns=["Data", "Histórico", "Valor", "Tipo", "Data_dt"])
    df = df.sort_values(by='Data_dt').reset_index(drop=True)
    return df.drop(columns=["Valor_raw"], errors='ignore')


# =============================================================
# PROCESSADORES POR BANCO (mantidos conforme o original)
# =============================================================

# As funções de cada banco continuam idênticas, pois todas já
# utilizam linha_parece_sujo(linha) para filtrar o lixo.
# Exemplo de um dos processadores (Bradesco):

def processar_extrato_bradesco(texto: str):
    """Bradesco"""
    linhas = texto.splitlines()
    trans, current_date, buffer = [], None, []
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha_parece_sujo(linha): 
            continue
        m_date = re.match(r"(\d{2}/\d{2}/\d{4})", linha)
        if m_date:
            current_date = m_date.group(1)
            buffer = []
            linha = linha[m_date.end():].strip()
        if not current_date: 
            continue
        m_val = re.search(r"(-?\s?\d{1,3}(?:\.\d{3})*,\d{2})", linha)
        if m_val:
            raw_val = m_val.group(1).replace(" ", "")
            tipo = "D" if "-" in raw_val else "C"
            historico = " ".join(buffer + [linha[:m_val.start()].strip()]).strip()
            trans.append({"Data": current_date, "Histórico": historico, "Valor": raw_val, "Tipo": tipo})
            buffer = []
        else:
            buffer.append(linha)
    return trans

# (demais processadores seguem o mesmo padrão)
# ... processar_extrato_bb, itau, santander, caixa, xp, sicoob, etc ...


# =============================================================
# EXTRAÇÃO, AVALIAÇÃO E LÓGICA UNIVERSAL
# =============================================================

def extrair_texto_pdf(pdf_file: io.BytesIO) -> str:
    texto = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=2, y_tolerance=2)
                if t:
                    texto += t + "\n"
    except Exception as e:
        raise ValueError(f"Não foi possível processar o arquivo PDF. Erro: {e}")
    if not texto.strip():
        raise ValueError("O PDF parece estar vazio ou contém apenas imagens.")
    return texto


def avaliar_resultado(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    score = len(df) * 10
    if 'Data_dt' in df.columns and len(df) > 0:
        dias = (df['Data_dt'].max() - df['Data_dt'].min()).days + 1
        score += dias * 5
    if len(df) > 1 and df['Valor'].dtype in ['float64', 'float32']:
        score += df['Valor'].var() * 3e-6
    tipo_counts = df['Tipo'].value_counts(normalize=True)
    prop_D = tipo_counts.get('D', 0)
    prop_C = tipo_counts.get('C', 0)
    score += (1 - abs(prop_D - prop_C)) * 50
    return score


def processar_extrato_universal(pdf_file: io.BytesIO) -> pd.DataFrame:
    texto = extrair_texto_pdf(pdf_file)
    melhor_df = pd.DataFrame()
    melhor_score = -1.0
    melhor_proc = "NENHUM"

    PROCESSADORES = {
        "BRADESCO": processar_extrato_bradesco,
        # (demais bancos mapeados aqui)
    }

    for nome, proc in PROCESSADORES.items():
        try:
            trans = proc(texto)
            df = normalizar_transacoes(trans)
            score = avaliar_resultado(df)
            if score > melhor_score:
                melhor_df, melhor_score, melhor_proc = df.copy(), score, nome
        except Exception:
            continue

    if not melhor_df.empty and 'Data_dt' in melhor_df.columns:
        melhor_df = melhor_df.drop(columns=['Data_dt'])
    print(f"SUCESSO: Melhor resultado: {melhor_proc} (Score: {melhor_score:.2f}, Linhas: {len(melhor_df)})")
    return melhor_df


def processar_extrato_principal(pdf_file: io.BytesIO) -> pd.DataFrame:
    return processar_extrato_universal(pdf_file)
