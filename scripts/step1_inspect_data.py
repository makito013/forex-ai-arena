"""
Step 1 — Data Inspector
Pipeline de preparação de dados para treino de agentes RL de Forex.

Etapa 1 de 4: Inspeção e limpeza dos dados brutos.
"""

import os
import re
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path("/Users/bruno.andrade/projetos/pessoal/estatistica/forex-ai-arena")
CALENDAR_RAW = BASE / "data/news/Calender_data.csv"
HISTORICAL_DIR = BASE / "data/historical"
PROCESSED_NEWS = BASE / "data/news/processed"
PROCESSED_DIR = BASE / "data/processed"

# ── 4. Criar estrutura de diretórios ──────────────────────────────────────────
PROCESSED_NEWS.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

MULTIPLIERS = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}


def parse_numeric(value: str) -> float:
    """
    Converte string de valor econômico para float.

    Exemplos tratados:
        '5.046M'  → 5_046_000.0
        '-67.4K'  → -67_400.0
        '1.2B'    → 1_200_000_000.0
        '56.8%'   → 56.8
        '3,269K'  → 3_269_000.0
        '3.2|4.1' → 3.2   (pega o primeiro valor)
        '-0.3%'   → -0.3
        '496'     → 496.0
        ''        → NaN
        'Tentative' → NaN
    """
    if not isinstance(value, str):
        return np.nan

    value = value.strip()
    if not value:
        return np.nan

    # pipe-separated: pega o primeiro
    if "|" in value:
        value = value.split("|")[0].strip()

    # remove espaços e vírgulas de milhar
    value = value.replace(",", "").replace(" ", "")

    # remove % (o valor já é percentual — preservamos a magnitude)
    value = value.replace("%", "")

    # multiplicador sufixo (K/M/B/T) — case-insensitive
    multiplier = 1.0
    upper = value.upper()
    if upper and upper[-1] in MULTIPLIERS:
        multiplier = MULTIPLIERS[upper[-1]]
        value = value[:-1]

    try:
        return float(value) * multiplier
    except (ValueError, TypeError):
        return np.nan


def parse_datetime_calendar(date_str: str, time_str: str):
    """Combina date (DD/MM/YYYY) + time (HH:MM) em Timestamp UTC."""
    if not isinstance(time_str, str):
        return np.nan
    time_str = time_str.strip()
    if time_str in ("All Day", "Tentative", ""):
        return np.nan
    try:
        dt = pd.to_datetime(f"{date_str} {time_str}", format="%d/%m/%Y %H:%M", utc=True)
        return dt
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 1. Limpa o Calendário Econômico
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("ETAPA 1 — Limpeza do Calendário Econômico")
print("=" * 70)

print(f"  Lendo {CALENDAR_RAW} ...")
cal_raw = pd.read_csv(CALENDAR_RAW, low_memory=False)
total_original = len(cal_raw)
print(f"  Total de linhas brutas : {total_original:,}")

# Moedas relevantes
RELEVANT_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}
RELEVANT_IMPORTANCE = {"high", "medium"}

cal = cal_raw.copy()

# Converter datas
print("  Convertendo datas ...")
cal["datetime_utc"] = cal.apply(
    lambda r: parse_datetime_calendar(r["date"], r["time"]), axis=1
)

# Filtrar moeda e importância
cal = cal[
    cal["currency"].isin(RELEVANT_CURRENCIES) &
    cal["importance"].isin(RELEVANT_IMPORTANCE)
].copy()

# Remover linhas sem datetime válido (All Day, Tentative)
cal = cal[cal["datetime_utc"].notna()].copy()

total_filtered = len(cal)

# Limpar colunas numéricas
print("  Limpando colunas actual/forecast/previous ...")
for col in ("actual", "forecast", "previous"):
    cal[col + "_num"] = cal[col].apply(parse_numeric)

# Ordenar
cal.sort_values("datetime_utc", inplace=True)
cal.reset_index(drop=True, inplace=True)

# Salvar
OUT_CSV = PROCESSED_NEWS / "calendar_clean.csv"
cal.to_csv(OUT_CSV, index=False)
print(f"  Salvo em {OUT_CSV}")

cal_date_min = cal["datetime_utc"].min()
cal_date_max = cal["datetime_utc"].max()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Inspeciona os CSVs de Preço
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("ETAPA 2 — Inspeção dos CSVs de Preço")
print("=" * 70)

price_files = sorted(HISTORICAL_DIR.glob("*.csv"))
price_info = {}

for fp in price_files:
    name = fp.stem  # ex.: EURUSDH1
    # Extrair par e timeframe
    # Padrão: 6 chars base + timeframe (H1, M1, M5, M15, M30)
    match = re.match(r"^([A-Z]{6})([A-Z]+\d+)$", name)
    if match:
        pair = match.group(1)
        tf = match.group(2)
    else:
        pair = name
        tf = "UNKNOWN"

    try:
        df_price = pd.read_csv(
            fp,
            encoding="utf-16",
            sep=",",
            header=None,
            names=["datetime", "open", "high", "low", "close", "volume", "flag"],
        )
        df_price["datetime"] = pd.to_datetime(
            df_price["datetime"], format="%Y.%m.%d %H:%M", utc=True
        )
        n = len(df_price)
        dt_min = df_price["datetime"].min()
        dt_max = df_price["datetime"].max()
        price_info[name] = {
            "pair": pair,
            "timeframe": tf,
            "rows": n,
            "date_min": dt_min,
            "date_max": dt_max,
        }
        print(f"  {name:<20} | {tf:<5} | {n:>8,} linhas | {dt_min.date()} → {dt_max.date()}")
    except Exception as e:
        print(f"  ERRO em {name}: {e}")
        price_info[name] = {"pair": pair, "timeframe": tf, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Relatório
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("RELATÓRIO FINAL")
print("=" * 70)

print(f"\n  CALENDÁRIO ECONÔMICO")
print(f"    Eventos originais (todas linhas) : {total_original:>10,}")
print(f"    Após filtro moeda + importância  : {total_filtered:>10,}")
print(f"    Range do calendário limpo        : {cal_date_min.date()} → {cal_date_max.date()}")
print(f"    Arquivo salvo em                 : {OUT_CSV}")

print(f"\n  DADOS DE PREÇO — {len(price_info)} arquivos")
print(f"  {'Arquivo':<20} | {'Par':<8} | {'TF':<5} | {'Linhas':>8} | {'Inicio':<12} | {'Fim':<12}")
print("  " + "-" * 75)
for name, info in price_info.items():
    if "error" in info:
        print(f"  {name:<20} | ERRO: {info['error']}")
    else:
        print(
            f"  {name:<20} | {info['pair']:<8} | {info['timeframe']:<5} | "
            f"{info['rows']:>8,} | {info['date_min'].date()} | {info['date_max'].date()}"
        )

# Sobreposição de datas entre calendário e preço
valid_prices = {k: v for k, v in price_info.items() if "error" not in v}
if valid_prices:
    price_global_min = min(v["date_min"] for v in valid_prices.values())
    price_global_max = max(v["date_max"] for v in valid_prices.values())
    overlap_start = max(cal_date_min, price_global_min)
    overlap_end = min(cal_date_max, price_global_max)

    print(f"\n  SOBREPOSIÇÃO DE DATAS")
    print(f"    Calendário   : {cal_date_min.date()} → {cal_date_max.date()}")
    print(f"    Preço (geral): {price_global_min.date()} → {price_global_max.date()}")
    if overlap_start <= overlap_end:
        cal_in_overlap = cal[
            (cal["datetime_utc"] >= overlap_start) &
            (cal["datetime_utc"] <= overlap_end)
        ]
        print(f"    Sobreposição : {overlap_start.date()} → {overlap_end.date()}")
        print(f"    Eventos de calendário na sobreposição: {len(cal_in_overlap):,}")
    else:
        print("    SEM sobreposição de datas!")


# ─────────────────────────────────────────────────────────────────────────────
# 4. JSON Resumo
# ─────────────────────────────────────────────────────────────────────────────
pairs_available = list({v["pair"] for v in valid_prices.values()})
pairs_available.sort()

date_range_prices = {}
for name, info in valid_prices.items():
    pair = info["pair"]
    tf = info["timeframe"]
    key = f"{pair}_{tf}"
    date_range_prices[key] = {
        "date_min": info["date_min"].isoformat(),
        "date_max": info["date_max"].isoformat(),
        "rows": info["rows"],
    }

summary = {
    "date_range_calendar": {
        "date_min": cal_date_min.isoformat(),
        "date_max": cal_date_max.isoformat(),
    },
    "total_events_original": total_original,
    "total_events_filtered": total_filtered,
    "date_range_prices": date_range_prices,
    "pairs_available": pairs_available,
    "overlap": {
        "date_min": overlap_start.isoformat() if overlap_start <= overlap_end else None,
        "date_max": overlap_end.isoformat() if overlap_start <= overlap_end else None,
    },
    "calendar_clean_path": str(OUT_CSV),
}

# Salva o JSON no diretório processed também
SUMMARY_JSON = PROCESSED_DIR / "step1_summary.json"
with open(SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2, default=str)

print()
print("=" * 70)
print("JSON RESUMO (próxima etapa):")
print("=" * 70)
print(json.dumps(summary, indent=2, default=str))
print()
print(f"  JSON salvo em: {SUMMARY_JSON}")
print()
print("Etapa 1 concluída com sucesso.")
