# ===========================================================
# Script definitivo de processamento de extratos bancários
# Adaptado para uso em produção com Streamlit
# Bancos: BB, Itaú, Bradesco, Santander, Caixa, XP, Sicoob,
# BNB, Nubank, Inter, Safra + fallback genérico
# ===========================================================

import re
import pandas as pd
import pdfplumber
import io  # Importante: para lidar com arquivos em memória

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
        r"\btransfer[eê]ncia\s+(recebida|enviada|ted|doc)\b", r"\bpix\s+(recebido|enviado)\b",
        r"\bted\s+(recebid|enviad)\b", r"\bdoc\s+(recebid|enviad)\b",
        r"\bcheque\s+(compensado|devolvido)\b", r"\bc[oó]digo\s+\d+"
    ]
    if any(re.search(p, texto, re.IGNORECASE) for p in excecoes_regex):
        return False

    # --- padrões de lixo reais ---
    padroes_lixo = [
        r"\bs\s*a\s*l\s*d\s*o\b", r"\bsaldo\s*(anterior|do\s+dia|total|atual|bloqueado|dispon[ií]vel|parcial|inicial|final|em\s+c/c)\b",
        r"\bsdo\s+(cta|apl|conta)\b", r"\bdetalhamento\b", r"\bextrato\b", r"\bcliente\b",
        r"\bconta\s+corrente\s*\|\s*movimenta", r"\blimite\b", r"\binvestimentos\b",
        r"\bdispon[ií]vel\b", r"\bbloqueado\b", r"\btotal\s+(de\s+)?(entradas|sa[ií]das|cr[ée]ditos|d[ée]bitos)\b",
        r"\bresumo\b", r"\bfale\s*conosco\b", r"\bouvidoria\b", r"\bpara\s+demais\s+siglas\b",
        r"\bnotas\s+explicativas\b", r"\btotalizador\b", r"\baplicações\s+automáticas\b",
        r"\bvalor\s+\(r\$\)\b", r"\bdocumento\b", r"\bdescri[cç][aã]o\b", r"\bcr[eé]ditos\b",
        r"\bd[eé]bitos\b", r"\bmovimenta[cç][aã]o\b", r"\bp[aá]gina\b", r"\bdata\s+lan[cç]amento\b",
        r"\bcomplemento\b", r"\bcentral\s+de\s+suporte\b", r"\bconta\s+corrente\s*\|\s*movimenta[cç][aã]o\b",
        r"\bvalores\s+em\s+r\$\b", r"\bper[ií]odo\s+de\b", r"\bsaldo\s*\+\s*limite\b",
        r"\bcobran[cç]a\s+d[01]\b", r"\bcheque\s+empresarial\b"
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
    df["Valor"] = df["Valor_raw"].astype(float)
    df["Tipo"] = df["Tipo"].astype(str).str.upper().map(lambda x: "D" if x.startswith("D") else "C")
    
    # Adiciona coluna de data em formato datetime para ordenação e remove inválidas
    df['Data_dt'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Data_dt'])
    if df.empty: return pd.DataFrame(columns=["Data", "Histórico", "Valor", "Tipo"])
    
    # Ordena por data e remove a coluna auxiliar
    df = df.sort_values(by='Data_dt').drop(columns=['Data_dt']).reset_index(drop=True)
    
    return df.drop(columns=["Valor_raw"], errors='ignore')


# ==================== PROCESSADORES POR BANCO ====================

def processar_extrato_bb(texto: str):
    """Banco do Brasil"""
    linhas = texto.splitlines()
    trans, current_date, buffer = [], None, []
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha_parece_sujo(linha): continue
        m_date = re.match(r"(\d{2}/\d{2}/\d{4})", linha)
        if m_date:
            current_date = m_date.group(1)
            buffer = []
            linha = linha[m_date.end():].strip()
        if not current_date: continue
        m_val = re.search(r"(-?\s?\d{1,3}(?:\.\d{3})*,\d{2})", linha)
        m_cd = re.search(r"\b([CD])\b", linha)
        if m_val:
            valor_raw = m_val.group(1).replace(" ", "")
            tipo = "D" if (m_cd and m_cd.group(1) == "D") or "-" in valor_raw else "C"
            historico = " ".join(buffer + [linha[:m_val.start()].strip()]).strip()
            trans.append({"Data": current_date, "Histórico": historico, "Valor": valor_raw, "Tipo": tipo})
            buffer = []
        else:
            buffer.append(linha)
    return trans

def processar_extrato_itau_formato1(texto: str):
    """Itaú - Formato antigo: '1 / mai valor -1.234,56'"""
    linhas = texto.splitlines()
    trans = []
    meses = {'jan': '01', 'fev': '02', 'mar': '03', 'abr': '04', 'mai': '05', 'jun': '06', 'jul': '07', 'ago': '08', 'set': '09', 'out': '10', 'nov': '11', 'dez': '12'}
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha_parece_sujo(linha) or re.search(r"\bsdo\s+(cta|apl|conta)", linha, re.IGNORECASE): continue
        m_data = re.search(r"(\d{1,2})\s*/\s*([a-z]{3})", linha, re.IGNORECASE)
        m_val = re.search(r"(-?\d{1,3}(?:\.\d{3})*,\d{2})", linha)
        if m_data and m_val:
            dia, mes_abrev = m_data.groups()
            mes_num = meses.get(mes_abrev.lower(), "00")
            data = f"{dia.zfill(2)}/{mes_num}/2024" # Assumindo ano, pode ser melhorado
            valor_raw = m_val.group(1).replace(" ", "")
            tipo = "D" if "-" in valor_raw else "C"
            historico = linha[m_data.end():m_val.start()].strip()
            trans.append({"Data": data, "Histórico": historico, "Valor": valor_raw, "Tipo": tipo})
    return trans

def processar_extrato_itau_formato2(texto: str):
    """Itaú - Formato novo: 'dd/mm DESCRIÇÃO valor-'"""
    linhas = texto.splitlines()
    trans, current_date, buffer = [], None, []
    for linha in linhas:
        linha = linha.strip()
        if not linha or re.search(r"\bsaldo\s+final\b", linha, re.IGNORECASE) or linha_parece_sujo(linha): continue
        m_date = re.match(r"(\d{2}/\d{2})\b", linha)
        if m_date:
            current_date = m_date.group(1) + "/2024" # Assumindo ano
            buffer = []
            linha = linha[m_date.end():].strip()
        if not current_date: continue
        m_val = re.search(r"(\d{1,3}(?:\.\d{3})*,\d{2})(-?)", linha)
        if m_val:
            raw_val, negativo = m_val.groups()
            tipo = "D" if negativo == "-" else "C"
            historico = " ".join(buffer + [linha[:m_val.start()].strip()]).strip()
            historico = re.sub(r"\s+", " ", historico).strip()
            if historico and len(historico) > 1:
                trans.append({"Data": current_date, "Histórico": historico, "Valor": raw_val, "Tipo": tipo})
            buffer = []
        elif linha:
            buffer.append(linha)
    return trans

def processar_extrato_itau(texto: str):
    """Itaú - Processador universal que tenta ambos os formatos"""
    trans_f2 = processar_extrato_itau_formato2(texto)
    return trans_f2 if trans_f2 else processar_extrato_itau_formato1(texto)

def processar_extrato_bradesco(texto: str):
    """Bradesco"""
    linhas = texto.splitlines()
    trans, current_date, buffer = [], None, []
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha_parece_sujo(linha): continue
        m_date = re.match(r"(\d{2}/\d{2}/\d{4})", linha)
        if m_date:
            current_date = m_date.group(1)
            buffer = []
            linha = linha[m_date.end():].strip()
        if not current_date: continue
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

def processar_extrato_santander_formato1(texto: str):
    """Santander - Formato 1 (antigo)"""
    linhas = texto.splitlines()
    trans, current_date = [], None
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha_parece_sujo(linha): continue
        m_date = re.match(r"(\d{2}/\d{2})\b", linha)
        if m_date:
            current_date = m_date.group(1) + "/2025" # Assumindo ano
            linha = linha[m_date.end():].strip()
        if not current_date: continue
        valores = re.findall(r"(\d{1,3}(?:\.\d{3})*,\d{2})\s*-?", linha)
        if not valores: continue
        historico = linha
        for val in valores: historico = historico.replace(val, "")
        historico = re.sub(r"\b\d{5,}\b", "", historico.replace("-", "")).strip()
        historico = re.sub(r"\s+", " ", historico).strip()
        tipo = "D" if linha.rstrip().endswith("-") else "C"
        if historico and len(historico) > 2:
            trans.append({"Data": current_date, "Histórico": historico, "Valor": valores[0], "Tipo": tipo})
    return trans

def processar_extrato_santander_formato2(texto: str):
    """Santander - Formato 2 (novo)"""
    linhas = texto.splitlines()
    trans = []
    padrao = re.compile(r"(\d{2}/\d{2}/\d{4})\s+(.+?)\s+\d+\s+(-?\d{1,3}(?:\.\d{3})*,\d{2})")
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha_parece_sujo(linha): continue
        m = padrao.search(linha)
        if m:
            data, hist, raw_val = m.groups()
            tipo = "D" if "-" in raw_val else "C"
            trans.append({"Data": data, "Histórico": hist.strip(), "Valor": raw_val.replace("-", ""), "Tipo": tipo})
    return trans

def processar_extrato_santander(texto: str):
    """Santander - Processador universal"""
    trans_f2 = processar_extrato_santander_formato2(texto)
    return trans_f2 if trans_f2 else processar_extrato_santander_formato1(texto)

def processar_extrato_caixa(texto: str):
    """Caixa Econômica Federal"""
    linhas = texto.splitlines()
    trans = []
    padrao = re.compile(r"(\d{2}/\d{2}/\d{4})\s+\d+\s+(.+?)\s+(-?\d{1,3}(?:\.\d{3})*,\d{2})\s+([CD])")
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha_parece_sujo(linha): continue
        m = padrao.search(linha)
        if m:
            data, hist, raw_val, tipo_flag = m.groups()
            tipo = "D" if tipo_flag.upper() == "D" or "-" in raw_val else "C"
            trans.append({"Data": data, "Histórico": hist.strip(), "Valor": raw_val.replace("-", ""), "Tipo": tipo})
    return trans

def processar_extrato_xp(texto: str):
    """XP Investimentos"""
    linhas = texto.splitlines()
    trans = []
    padrao = re.compile(r"(\d{2}/\d{2}/\d{4})\s+(.+?)\s+(-?R?\$?\s*\d{1,3}(?:\.\d{3})*,\d{2})")
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha_parece_sujo(linha): continue
        m = padrao.search(linha)
        if m:
            data, hist, raw_val = m.groups()
            raw_val = raw_val.replace("R$", "").strip()
            tipo = "D" if "-" in raw_val else "C"
            trans.append({"Data": data, "Histórico": hist.strip(), "Valor": raw_val.replace("-", ""), "Tipo": tipo})
    return trans

def processar_extrato_sicoob(texto: str):
    """Sicoob"""
    linhas = texto.splitlines()
    trans = []
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha_parece_sujo(linha): continue
        m_date = re.match(r"(\d{2}/\d{2})", linha)
        m_val = re.search(r"(-?\d{1,3}(?:\.\d{3})*,\d{2})\s*([CD])", linha)
        if m_date and m_val:
            data = m_date.group(1) + "/2024" # Assumindo ano
            raw_val, tipo_flag = m_val.groups()
            tipo = "D" if tipo_flag.upper() == "D" or "-" in raw_val else "C"
            hist = linha[m_date.end():m_val.start()].strip()
            trans.append({"Data": data, "Histórico": hist, "Valor": raw_val.replace("-", ""), "Tipo": tipo})
    return trans

def processar_extrato_bnb(texto: str):
    """Banco do Nordeste"""
    linhas = texto.splitlines()
    trans, current_date, buffer = [], None, []
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha_parece_sujo(linha) or re.search(r"detalhamento\s+do\s+saldo", linha, re.IGNORECASE): continue
        m_date = re.match(r"(\d{2}/\d{2}/\d{4})", linha)
        if m_date:
            current_date = m_date.group(1)
            buffer = []
            linha = linha[m_date.end():].strip()
        if not current_date: continue
        m_val_tipo = re.search(r"(\d{1,3}(?:\.\d{3})*,\d{2})\s+([DC])\b", linha)
        if m_val_tipo:
            raw_val, tipo_flag = m_val_tipo.groups()
            historico = " ".join(buffer + [linha[:m_val_tipo.start()].strip()]).strip()
            trans.append({"Data": current_date, "Histórico": historico, "Valor": raw_val, "Tipo": tipo_flag.upper()})
            buffer = []
        else:
            m_val = re.search(r"(-?\s?\d{1,3}(?:\.\d{3})*,\d{2})", linha)
            if m_val:
                raw_val = m_val.group(1).replace(" ", "")
                tipo = "D" if "-" in raw_val else "C"
                historico = " ".join(buffer + [linha[:m_val.start()].strip()]).strip()
                trans.append({"Data": current_date, "Histórico": historico, "Valor": raw_val.replace("-", ""), "Tipo": tipo})
                buffer = []
            else:
                buffer.append(linha)
    return trans

def processar_extrato_nubank(texto: str):
    """Nubank"""
    linhas = texto.splitlines()
    trans, current_date, buffer = [], None, []
    meses = {'JAN': '01', 'FEV': '02', 'MAR': '03', 'ABR': '04', 'MAI': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08', 'SET': '09', 'OUT': '10', 'NOV': '11', 'DEZ': '12', 'MÇO': '03'}
    for linha in linhas:
        linha = linha.strip()
        m_date = re.match(r"(\d{2})\s+([A-ZÇ]{3})\s+(\d{4})", linha)
        if m_date:
            dia, mes_abrev, ano = m_date.groups()
            mes_num = meses.get(mes_abrev, '00')
            current_date = f"{dia}/{mes_num}/{ano}"
            buffer = []
            continue
        if not current_date or linha_parece_sujo(linha): continue
        m_val = re.search(r"(-?\s?R?\$?\s?\d{1,3}(?:\.\d{3})*,\d{2})", linha)
        if m_val:
            raw_val = m_val.group(1).replace("R$", "").replace(" ", "").strip()
            historico = linha[:m_val.start()].strip() or " ".join(buffer).strip()
            historico = re.sub(r"\s+", " ", historico).strip()
            tipo = 'C' if 'recebid' in historico.lower() else ('D' if 'enviado' in historico.lower() or '-' in raw_val else 'C')
            if historico:
                trans.append({"Data": current_date, "Histórico": historico, "Valor": raw_val.replace("-", ""), "Tipo": tipo})
            buffer = []
        elif linha and not linha.lower().startswith("total"):
            buffer.append(linha)
    return trans

def processar_extrato_safra(texto: str):
    """Banco Safra"""
    linhas = texto.splitlines()
    trans, current_date, buffer = [], None, []
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha_parece_sujo(linha) or linha.startswith('/'): continue
        if len(re.findall(r"R\$\s*\d{1,3}(?:\.\d{3})*,\d{2}", linha)) >= 3: continue
        m_novo = re.match(r"(\d{2}/\d{2})\s+(.+?)\s+(-?\d{1,3}(?:\.\d{3})*,\d{2})\s*$", linha)
        if m_novo:
            data_curta, historico, raw_val = m_novo.groups()
            data = data_curta + "/2024" # Assumindo ano
            tipo = "D" if "-" in raw_val else "C"
            historico = re.sub(r"\s+\d{6,}\s*$", "", historico).strip()
            if len(historico) > 3:
                trans.append({"Data": data, "Histórico": historico, "Valor": raw_val.replace("-", ""), "Tipo": tipo})
            continue
        m_date = re.match(r"(\d{2}/\d{2})", linha)
        if m_date:
            current_date = m_date.group(1) + "/2024" # Assumindo ano
            buffer = []
            linha = linha[m_date.end():].strip()
        if not current_date: continue
        m_val = re.search(r"(\d{1,3}(?:\.\d{3})*,\d{2})\s*-?\s*$", linha)
        if m_val:
            raw_val = m_val.group(1)
            tipo = "D" if "-" in linha[m_val.start():] else "C"
            historico = " ".join(buffer + [linha[:m_val.start()].strip()]).strip()
            historico = re.sub(r"-\d{6}", "", historico).strip()
            if len(historico) > 2:
                trans.append({"Data": current_date, "Histórico": historico, "Valor": raw_val, "Tipo": tipo})
            buffer = []
        elif linha:
            buffer.append(linha)
    return trans

def processar_extrato_inter(texto: str):
    """Banco Inter"""
    linhas = texto.splitlines()
    trans, data_atual = [], None
    meses = {'janeiro': '01', 'fevereiro': '02', 'março': '03', 'marco': '03', 'abril': '04', 'maio': '05', 'junho': '06', 'julho': '07', 'agosto': '08', 'setembro': '09', 'outubro': '10', 'novembro': '11', 'dezembro': '12'}
    for linha in linhas:
        linha = linha.strip()
        if not linha: continue
        m_data = re.search(r"(\d{1,2})\s+de\s+([a-zç]+)\s+de\s+(\d{4})\s+Saldo\s+do\s+dia:", linha, re.IGNORECASE)
        if m_data:
            dia, mes_ext, ano = m_data.groups()
            mes_num = meses.get(mes_ext.lower(), "00")
            data_atual = f"{dia.zfill(2)}/{mes_num}/{ano}"
            continue
        if not data_atual or linha_parece_sujo(linha): continue
        valores = re.findall(r"(-?R?\$?\s?\d{1,3}(?:\.\d{3})*,\d{2})", linha)
        if len(valores) < 2: continue
        valor_transacao = valores[0]
        raw_val = valor_transacao.replace("R$", "").replace(" ", "").strip()
        tipo = "D" if "-" in raw_val else "C"
        raw_val = raw_val.replace("-", "")
        historico = linha[:linha.find(valor_transacao)].strip().replace('"', '').removesuffix(':').strip()
        trans.append({"Data": data_atual, "Histórico": historico, "Valor": raw_val, "Tipo": tipo})
    return trans

def processar_extrato_generico(texto: str):
    """Processador genérico (fallback)"""
    linhas = texto.splitlines()
    trans = []
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha_parece_sujo(linha): continue
        m = re.search(r"(\d{2}/\d{2}/\d{4}).*?(-?\s?\d{1,3}(?:\.\d{3})*,\d{2})", linha)
        if m:
            data, raw_val = m.groups()
            tipo = "D" if "-" in raw_val else "C"
            hist = re.sub(r"\s+", " ", linha[linha.find(data) + len(data):linha.find(raw_val)].strip())
            trans.append({"Data": data, "Histórico": hist, "Valor": raw_val.replace("-", ""), "Tipo": tipo})
    return trans


# ==================== DETECTOR DE BANCO ====================

def detectar_banco(texto: str):
    """Detecta automaticamente o banco pelo conteúdo do PDF"""
    tu = texto.upper()
    if "NUBANK" in tu or "NU PAGAMENTOS" in tu: return "NUBANK"
    if "BANCO DO BRASIL" in tu: return "BANCO DO BRASIL"
    if "ITAÚ" in tu or "ITAU" in tu: return "ITAU"
    if "BRADESCO" in tu: return "BRADESCO"
    if "SANTANDER" in tu: return "SANTANDER"
    if "CAIXA" in tu or "CAIXA ECONÔMICA" in tu: return "CAIXA"
    if "XP" in tu and "INVESTIMENTOS" in tu: return "XP INVESTIMENTOS"
    if "SICOOB" in tu: return "SICOOB"
    if "NORDESTE" in tu or "BNB" in tu: return "BANCO DO NORDESTE"
    if "INTER" in tu and "BANCO" in tu: return "INTER"
    if "SAFRA" in tu or "BANCO SAFRA" in tu: return "SAFRA"
    return "DESCONHECIDO"


# ==================== MAPEAMENTO DE PROCESSADORES ====================

PROCESSADORES = {
    "BANCO DO BRASIL": processar_extrato_bb,
    "ITAU": processar_extrato_itau,
    "BRADESCO": processar_extrato_bradesco,
    "SANTANDER": processar_extrato_santander,
    "CAIXA": processar_extrato_caixa,
    "XP INVESTIMENTOS": processar_extrato_xp,
    "SICOOB": processar_extrato_sicoob,
    "BANCO DO NORDESTE": processar_extrato_bnb,
    "NUBANK": processar_extrato_nubank,
    "INTER": processar_extrato_inter,
    "SAFRA": processar_extrato_safra,
    "DESCONHECIDO": processar_extrato_generico
}


# ==================== EXTRAÇÃO DE TEXTO (MODIFICADA) ====================

def extrair_texto_pdf(pdf_file: io.BytesIO) -> str:
    """
    Extrai texto de um objeto de arquivo PDF em memória (vindo do Streamlit).
    """
    texto = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=2, y_tolerance=2)
                if t:
                    texto += t + "\n"
    except Exception as e:
        # Lança uma exceção que pode ser capturada pela interface do Streamlit
        # para mostrar uma mensagem de erro amigável ao usuário.
        raise ValueError(f"Não foi possível processar o arquivo PDF. Pode estar corrompido, protegido por senha ou em um formato não suportado. Erro técnico: {e}")
        
    if not texto.strip():
        raise ValueError("O PDF parece estar vazio ou contém apenas imagens. Nenhum texto foi extraído.")
        
    return texto

# NENHUM CÓDIGO DEVE VIR DEPOIS DESTA LINHA.