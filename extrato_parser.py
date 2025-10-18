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
        r"\bdeb\s+tit\b", r"\bdeb\s+parc\b" # Adicionados padrões comuns de débito
    ]
    if any(re.search(p, texto, re.IGNORECASE) for p in excecoes_regex):
        return False

    # --- padrões de lixo reais ---
    padroes_lixo = [
        r"\bs\s*a\s*l\s*d\s*o\b", r"\bsaldo\s*(anterior|do\s+dia|total|atual|bloqueado|dispon[ií]vel|parcial|inicial|final|em\s+c/c)\b",
        r"\bsdo\s+(cta|apl|conta)\b", r"\bdetalhamento\b", r"\bextrato\b", r"\bcliente\b",
        r"\bconta\s+corrente\s*\|\s*movimenta", r"\blimite\b", r"\binvestimentos\b",
        r"\bdispon[ií]vel\b", r"\bbloqueado\b", 
        
        # PADRÃO CORRIGIDO/ADICIONADO: Captura linhas simples de TOTAL com valor
        r"^\s*total\s*r?\s*\$?\s*[\d\.\,]+\s*$", 
        
        r"\btotal\s+(de\s+)?(entradas|sa[ií]das|cr[ée]ditos|d[ée]bitos)\b",
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
    # Verifica se o resultado é um número válido (inteiro ou float com até 2 casas decimais)
    return s if re.match(r"^\d+(\.\d{1,2})?$", s) else None


def normalizar_transacoes(transacoes):
    """Padroniza saída em DataFrame com colunas Data | Histórico | Valor | Tipo."""
    df = pd.DataFrame(transacoes, columns=["Data", "Histórico", "Valor", "Tipo"])
    if df.empty: return df

    df["Valor_raw"] = df["Valor"].astype(str).apply(clean_value_str)
    df = df.dropna(subset=["Valor_raw"]).copy()
    
    # Tentativa de conversão para float
    try:
        df["Valor"] = df["Valor_raw"].astype(float)
    except ValueError:
        return pd.DataFrame() # Retorna vazio se a conversão falhar

    df["Tipo"] = df["Tipo"].astype(str).str.upper().map(lambda x: "D" if x.startswith("D") else "C")
    
    # Adiciona coluna de data em formato datetime para ordenação e remove inválidas
    df['Data_dt'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Data_dt'])
    
    if df.empty: return pd.DataFrame(columns=["Data", "Histórico", "Valor", "Tipo", "Data_dt"])
    
    # Ordena por data e remove a coluna auxiliar
    df = df.sort_values(by='Data_dt').reset_index(drop=True)
    
    # Mantém 'Data_dt' para a função de avaliação antes de remover
    return df.drop(columns=["Valor_raw"], errors='ignore')


# ==================== PROCESSADORES POR BANCO ====================
# (Mantidos inalterados, pois representam as 'tentativas' de extração)

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
    """
    Sicoob - AJUSTADO para o padrão de data curta + histórico multilinha + valor/tipo.
    """
    linhas = texto.splitlines()
    trans, current_date, buffer = [], None, []
    # Usando o ano 2021 do extrato exemplo
    ano_fixo = "2021" 

    for linha in linhas:
        linha = linha.strip()
        if not linha: continue

        # 1. Tenta identificar a data (dd/mm) no início da linha
        m_date = re.match(r"(\d{2}/\d{2})", linha)
        
        # 2. Tenta encontrar o valor e o tipo (C ou D) no final de qualquer linha
        # Padrão: 1.234,56C ou 1.234,56D (com ou sem espaço antes do C/D)
        m_val_tipo = re.search(r"(\d{1,3}(?:\.\d{3})*,\d{2})\s*([CD])\s*$", linha)
        
        # 3. Tenta encontrar apenas o valor. Tipo será inferido do histórico se 'm_val_tipo' falhar
        m_val_only = re.search(r"(\d{1,3}(?:\.\d{3})*,\d{2})\s*$", linha)

        if m_date:
            # Nova data encontrada, reseta o buffer
            current_date = m_date.group(1) + "/" + ano_fixo
            linha = linha[m_date.end():].strip()
            buffer = []
            
        if not current_date: 
            # Ainda não encontrou a primeira data, ignora lixo
            if not linha_parece_sujo(linha):
                buffer.append(linha) # Acumula lixo potencial
            continue
        
        # Processar a transação
        if m_val_tipo:
            # Caso mais fácil: valor e tipo (C/D) explícitos
            raw_val, tipo_flag = m_val_tipo.groups()
            historico = " ".join(buffer + [linha[:m_val_tipo.start()].strip()]).strip()
            
            # Limpar números de documento/código no histórico (se não for a descrição principal)
            historico = re.sub(r"(DOC\.:\s*\d+|CODIGO\s+TED:\s*T\d+)", "", historico, flags=re.IGNORECASE).strip()
            historico = re.sub(r"\s+", " ", historico).strip()

            if historico and len(historico) > 2:
                trans.append({"Data": current_date, "Histórico": historico, "Valor": raw_val, "Tipo": tipo_flag.upper()})
            
            buffer = []
            
        elif m_val_only:
             # Caso 2: Apenas valor, inferir o tipo pelo histórico ou pela ausência de C/D explícito
            raw_val = m_val_only.group(1)
            historico_tentativa = " ".join(buffer + [linha[:m_val_only.start()].strip()]).strip().lower()
            
            # Tenta inferir o tipo do histórico
            if "emit.outra if" in historico_tentativa or "deb" in historico_tentativa:
                tipo = "D"
            elif "receb.outra if" in historico_tentativa or "cred" in historico_tentativa or "transf.contas" in historico_tentativa:
                tipo = "C"
            else:
                # Se não puder inferir, usa um chute padrão (ex: D) se o histórico for limpo
                tipo = "D" 

            historico = " ".join(buffer + [linha[:m_val_only.start()].strip()]).strip()
            historico = re.sub(r"(DOC\.:\s*\d+|CODIGO\s+TED:\s*T\d+)", "", historico, flags=re.IGNORECASE).strip()
            historico = re.sub(r"\s+", " ", historico).strip()
            
            # Evita capturar a linha de SALDO DO DIA/FINAL que termina apenas com valor
            # OBS: Esta verificação é redundante se `linha_parece_sujo` funcionar, mas a mantemos como segurança.
            if not linha_parece_sujo(historico) and len(historico) > 2:
                trans.append({"Data": current_date, "Histórico": historico, "Valor": raw_val, "Tipo": tipo})

            buffer = []
        
        elif linha and not linha_parece_sujo(linha):
            # Linha não tem data, nem valor/tipo. É continuação do histórico.
            buffer.append(linha)
            
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


# ==================== MAPEAMENTO DE PROCESSADORES ====================

# Dicionário usado para ITERAR sobre todas as opções de processamento.
PROCESSADORES = {
    "BB": processar_extrato_bb,
    "ITAU": processar_extrato_itau,
    "BRADESCO": processar_extrato_bradesco,
    "SANTANDER": processar_extrato_santander,
    "CAIXA": processar_extrato_caixa,
    "XP": processar_extrato_xp,
    "SICOOB": processar_extrato_sicoob, 
    "BNB": processar_extrato_bnb,
    "NUBANK": processar_extrato_nubank,
    "INTER": processar_extrato_inter,
    "SAFRA": processar_extrato_safra,
    "GENERICO": processar_extrato_generico
}


# ==================== EXTRAÇÃO E AVALIAÇÃO (LÓGICA MESTRA) ====================

def extrair_texto_pdf(pdf_file: io.BytesIO) -> str:
    """Extrai texto de um objeto de arquivo PDF em memória (vindo do Streamlit)."""
    texto = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                # Usar x_tolerance e y_tolerance ajuda a manter linhas juntas
                t = page.extract_text(x_tolerance=2, y_tolerance=2)
                if t:
                    texto += t + "\n"
    except Exception as e:
        raise ValueError(f"Não foi possível processar o arquivo PDF. Erro: {e}")
        
    if not texto.strip():
        raise ValueError("O PDF parece estar vazio ou contém apenas imagens. Nenhum texto foi extraído.")
        
    return texto

def avaliar_resultado(df: pd.DataFrame) -> float:
    """
    Atribui uma 'nota' (score) a um DataFrame de transações para
    selecionar o melhor resultado.
    """
    if df.empty:
        return 0.0

    score = 0.0
    num_linhas = len(df)
    
    # Peso 1: Quantidade de linhas (o mais importante)
    score += num_linhas * 10 
    
    # Peso 2: Diversidade de datas (evita extrair a mesma transação repetidamente)
    if 'Data_dt' in df.columns and num_linhas > 0:
        dias_cobertos = (df['Data_dt'].max() - df['Data_dt'].min()).days + 1
        score += dias_cobertos * 5
        
    # Peso 3: Variância dos valores (em R$) - Extratos reais têm valores variados
    if num_linhas > 1 and df['Valor'].dtype in ['float64', 'float32']:
        # O fator 3e-6 é um ajuste para que a variância não domine o score
        score += df['Valor'].var() * 3e-6 
    
    # Peso 4: Proporção de Débitos/Créditos (balanceamento)
    tipo_counts = df['Tipo'].value_counts(normalize=True)
    prop_D = tipo_counts.get('D', 0)
    prop_C = tipo_counts.get('C', 0)
    # Penaliza resultados muito desequilibrados (ex: 100% Débito)
    balance_score = 1 - abs(prop_D - prop_C)
    score += balance_score * 50

    return score


def processar_extrato_universal(pdf_file: io.BytesIO) -> pd.DataFrame:
    """
    Tenta TODOS os processadores no texto do PDF e retorna o melhor resultado.
    """
    # 1. Extrair texto de forma robusta
    texto = extrair_texto_pdf(pdf_file)
    
    melhor_df = pd.DataFrame()
    melhor_score = -1.0
    melhor_processador = "NENHUM"
    
    # 2. Executar e avaliar cada processador
    for nome_banco, processador in PROCESSADORES.items():
        try:
            # 2.1. Extrair transações usando o modelo (pode levantar exceções)
            transacoes_raw = processador(texto)
            
            # 2.2. Normalizar e limpar o resultado
            df_atual = normalizar_transacoes(transacoes_raw)
            
            # 2.3. Avaliar a qualidade do resultado
            score_atual = avaliar_resultado(df_atual)
            
            # 2.4. Comparar e selecionar o melhor
            if score_atual > melhor_score:
                melhor_score = score_atual
                # Copiar para evitar que a próxima iteração altere o dataframe
                melhor_df = df_atual.copy() 
                melhor_processador = nome_banco
                
            # print(f"DEBUG: {nome_banco} - Transações: {len(df_atual)} - Score: {score_atual:.2f}") # Log de debug
            
        except Exception as e:
            # print(f"AVISO: Processador {nome_banco} falhou: {e}") # Log de debug
            continue

    if not melhor_df.empty and 'Data_dt' in melhor_df.columns:
        # Remover a coluna auxiliar de data do resultado final
        melhor_df = melhor_df.drop(columns=['Data_dt'])

    print(f"SUCESSO: Melhor resultado encontrado por: {melhor_processador} (Score: {melhor_score:.2f}, Transações: {len(melhor_df)})")
    
    return melhor_df


# ==================== FUNÇÃO PRINCIPAL DE EXECUÇÃO ====================

def processar_extrato_principal(pdf_file: io.BytesIO) -> pd.DataFrame:
    """
    Função principal que a aplicação (ex: Streamlit) deve chamar.
    """
    return processar_extrato_universal(pdf_file)
