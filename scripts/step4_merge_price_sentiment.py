"""
Step 4 — Merge Price + Sentiment
Pipeline: news_data_guide_v1
Agente: Merger (Etapa 4 de 4)
"""

import os
import glob
import json
import pandas as pd
from datetime import datetime

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_DIR       = os.path.join(BASE_DIR, "data", "historical")
SENTIMENT_DIR  = os.path.join(BASE_DIR, "data", "news", "processed")
OUT_DIR        = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────
PAIRS = ["EURUSD", "GBPUSD", "USDCHF", "USDCAD", "NZDUSD", "XAUUSD"]
# Todos os timeframes disponíveis — script descobre via glob quais existem
TIMEFRAMES = ["H1", "M30", "M15", "M5", "M1"]

WINDOW_START = pd.Timestamp("2023-08-13")
WINDOW_END   = pd.Timestamp("2025-10-01")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.85  # até aqui val, depois test


# ──────────────────────────────────────────────
# Funções
# ──────────────────────────────────────────────

def load_price_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        header=None,
        encoding="utf-16",
        names=["datetime", "open", "high", "low", "close", "volume", "flag"],
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y.%m.%d %H:%M")
    df = df.set_index("datetime").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def load_sentiment_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    return df


def add_split_column(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    df = df.copy()
    df["split"] = "train"
    val_start  = int(total * TRAIN_RATIO)
    test_start = int(total * VAL_RATIO)
    df.iloc[val_start:test_start, df.columns.get_loc("split")] = "val"
    df.iloc[test_start:,          df.columns.get_loc("split")] = "test"
    return df


def sentiment_coverage_pct(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    return round((df["sentiment_score"] != 0.0).sum() / len(df) * 100, 2)


# ──────────────────────────────────────────────
# Descobre arquivos de preço disponíveis
# ──────────────────────────────────────────────
available_price = {}
for f in glob.glob(os.path.join(HIST_DIR, "*.csv")):
    fname = os.path.basename(f).replace(".csv", "")
    available_price[fname] = f

print(f"\nArquivos de preço encontrados: {len(available_price)}")

# ──────────────────────────────────────────────
# Loop principal
# ──────────────────────────────────────────────
results   = []
manifest_files = []

# Cabeçalho do relatório
header = (
    f"{'Arquivo':<45} {'Linhas':>7} {'Treino':>7} {'Val':>6} {'Teste':>6} "
    f"{'Cobertura%':>11} {'Início':>12} {'Fim':>12}"
)
print("\n" + "=" * len(header))
print(header)
print("=" * len(header))

for pair in PAIRS:
    for tf in TIMEFRAMES:
        price_key  = f"{pair}{tf}"
        price_path = available_price.get(price_key)

        if price_path is None:
            continue  # pula silenciosamente

        sentiment_path = os.path.join(SENTIMENT_DIR, f"{pair}_sentiment_{tf}.csv")
        if not os.path.exists(sentiment_path):
            # Sem sentiment para este TF — ainda gera com zeros
            sentiment_df = None
        else:
            sentiment_df = load_sentiment_csv(sentiment_path)

        # Carrega preço
        price_df = load_price_csv(price_path)

        # Merge
        if sentiment_df is not None:
            merged = price_df.join(sentiment_df, how="left")
        else:
            merged = price_df.copy()
            merged["sentiment_score"] = float("nan")
            merged["event_count"]     = float("nan")
            merged["max_importance"]  = float("nan")

        # Preenche NaN com neutro
        merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
        merged["event_count"]     = merged["event_count"].fillna(0).astype(int)
        merged["max_importance"]  = merged["max_importance"].fillna(0.0)

        # Filtra janela de sobreposição
        merged = merged.loc[WINDOW_START:WINDOW_END]

        if len(merged) == 0:
            print(f"  [AVISO] {price_key}: sem dados no intervalo {WINDOW_START.date()} — {WINDOW_END.date()}")
            continue

        # Adiciona coluna split
        merged = add_split_column(merged)

        # Salva
        out_fname = f"{pair}_{tf}_with_sentiment.csv"
        out_path  = os.path.join(OUT_DIR, out_fname)
        merged.to_csv(out_path)

        # Estatísticas
        total      = len(merged)
        train_rows = (merged["split"] == "train").sum()
        val_rows   = (merged["split"] == "val").sum()
        test_rows  = (merged["split"] == "test").sum()
        coverage   = sentiment_coverage_pct(merged)
        date_start = merged.index.min().date()
        date_end   = merged.index.max().date()

        row_report = (
            f"{out_fname:<45} {total:>7} {train_rows:>7} {val_rows:>6} {test_rows:>6} "
            f"{coverage:>10.1f}% {str(date_start):>12} {str(date_end):>12}"
        )
        print(row_report)

        manifest_files.append({
            "path":                  f"data/processed/{out_fname}",
            "pair":                  pair,
            "timeframe":             tf,
            "rows":                  int(total),
            "train_rows":            int(train_rows),
            "val_rows":              int(val_rows),
            "test_rows":             int(test_rows),
            "date_start":            str(date_start),
            "date_end":              str(date_end),
            "sentiment_coverage_pct": float(coverage),
        })

print("=" * len(header))

# ──────────────────────────────────────────────
# Manifesto JSON
# ──────────────────────────────────────────────
manifest = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "pipeline":     "news_data_guide_v1",
    "files":        manifest_files,
}

manifest_path = os.path.join(OUT_DIR, "dataset_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"\nManifesto salvo em: {manifest_path}")
print(f"\nPipeline concluído. {len(manifest_files)} datasets prontos para treino.")
