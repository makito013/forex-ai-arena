#!/usr/bin/env python3
"""
Step 3: Aggregate Sentiment Scores por Par Forex

Pipeline de preparação de dados para treino de agentes RL de Forex.
Converte eventos econômicos pontuados (events_scored.csv) em séries
temporais de sentimento H1/M15/M30 por par de moedas.

Etapas anteriores:
  - Etapa 1: data/news/processed/calendar_clean.csv   (28.418 eventos)
  - Etapa 2: data/news/processed/events_scored.csv    (10.216 eventos, 2023-08 a 2025-10)

Saída:
  - data/news/processed/{PAIR}_sentiment_H1.csv
  - data/news/processed/{PAIR}_sentiment_M30.csv
  - data/news/processed/{PAIR}_sentiment_M15.csv
  - data/processed/step3_summary.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Configuração ─────────────────────────────────────────────────────────────

BASE_DIR   = Path("/Users/bruno.andrade/projetos/pessoal/estatistica/forex-ai-arena")
INPUT_CSV  = BASE_DIR / "data/news/processed/events_scored.csv"
OUTPUT_DIR = BASE_DIR / "data/news/processed"
SUMMARY_PATH = BASE_DIR / "data/processed/step3_summary.json"

# Mapeamento par → [(moeda, sinal)]
#   sinal +1.0 : moeda é base  → score positivo = bullish para o par
#   sinal -1.0 : moeda é quote → score positivo = bearish para o par
PAIR_CURRENCY_MAP = {
    "EURUSD": [("EUR", +1.0), ("USD", -1.0)],
    "GBPUSD": [("GBP", +1.0), ("USD", -1.0)],
    "USDCHF": [("USD", +1.0), ("CHF", -1.0)],
    "USDCAD": [("USD", +1.0), ("CAD", -1.0)],
    "NZDUSD": [("NZD", +1.0), ("USD", -1.0)],
    "XAUUSD": [("USD", -1.0)],           # Ouro: USD forte → bearish XAU/USD
}

IMPORTANCE_WEIGHTS = {"high": 2.0, "medium": 1.0}

# Janela de sobreposição (cobertura do calendário econômico vs. dados de preço)
DATE_START = pd.Timestamp("2023-08-13 00:00:00")
DATE_END   = pd.Timestamp("2025-10-01 23:00:00")

# Janela de influência de cada evento (em horas)
INFLUENCE_WINDOW_HOURS = 24


# ─── Carga de dados ───────────────────────────────────────────────────────────

def load_events() -> pd.DataFrame:
    """Carrega e normaliza o CSV de eventos pontuados."""
    df = pd.read_csv(INPUT_CSV)

    # Parse e normalização de timezone → UTC naive
    df["event_datetime"] = (
        pd.to_datetime(df["event_datetime"], utc=True)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

    # Peso de importância numérico
    df["importance_weight"] = (
        df["importance"].str.lower()
        .map(IMPORTANCE_WEIGHTS)
        .fillna(1.0)
    )

    print(f"Eventos carregados: {len(df):,}")
    print(f"Período: {df['event_datetime'].min()} → {df['event_datetime'].max()}")
    print(f"Moedas presentes: {sorted(df['currency'].unique().tolist())}")
    return df


# ─── Cálculo de sentimento ────────────────────────────────────────────────────

def compute_pair_h1(
    pair_events: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
    pair: str,
) -> pd.DataFrame:
    """
    Para cada hora T no índice, agrega os eventos das últimas INFLUENCE_WINDOW_HOURS
    com decaimento linear: decay = max(0, 1 - age_hours / INFLUENCE_WINDOW_HOURS).

    Formula:
        sentiment_score = Σ(adjusted_score × imp_weight × decay)
                        / Σ(imp_weight × decay)

    Usa arrays numpy para eficiência (O(events × window_hours)).
    """
    n = len(hourly_index)

    # Buffers de acumulação
    weighted_scores = np.zeros(n, dtype=np.float64)
    total_weights   = np.zeros(n, dtype=np.float64)
    event_counts    = np.zeros(n, dtype=np.int32)
    max_imp         = np.zeros(n, dtype=np.float64)

    # Converte índice horário para int64 nanosegundos para searchsorted eficiente
    idx_ns = hourly_index.values.astype(np.int64)
    window_ns = int(INFLUENCE_WINDOW_HOURS * 3600 * 1e9)

    # Pre-extrai colunas como numpy arrays (evita overhead de iterrows/itertuples)
    event_times_ns  = pair_events["event_datetime"].values.astype(np.int64)
    adjusted_scores = pair_events["adjusted_score"].values.astype(np.float64)
    imp_weights     = pair_events["importance_weight"].values.astype(np.float64)

    print(f"  Processando {len(pair_events):,} eventos para {pair}...")

    for j in range(len(event_times_ns)):
        event_ns       = event_times_ns[j]
        adj_score      = adjusted_scores[j]
        imp_weight     = imp_weights[j]

        # Primeira hora >= event_ns  (início da janela de influência)
        first_idx = int(np.searchsorted(idx_ns, event_ns, side="left"))
        # Primeira hora > event_ns + 24h  (fim da janela, exclusive)
        last_idx  = int(np.searchsorted(idx_ns, event_ns + window_ns, side="right"))

        if first_idx >= n:
            continue  # Evento após o fim do índice

        for i in range(first_idx, min(last_idx, n)):
            age_ns   = idx_ns[i] - event_ns
            age_h    = age_ns / 3.6e12          # nanosegundos → horas
            decay    = max(0.0, 1.0 - age_h / INFLUENCE_WINDOW_HOURS)
            w        = imp_weight * decay

            if w <= 0.0:
                continue

            weighted_scores[i] += adj_score * w
            total_weights[i]   += w
            event_counts[i]    += 1
            if imp_weight > max_imp[i]:
                max_imp[i] = imp_weight

    # Score final normalizado; horas sem eventos = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        final_scores = np.where(
            total_weights > 0,
            weighted_scores / total_weights,
            0.0,
        )

    # Clamp no intervalo [-1, 1]
    final_scores = np.clip(final_scores, -1.0, 1.0)

    return pd.DataFrame({
        "datetime":        hourly_index,
        "sentiment_score": final_scores,
        "event_count":     event_counts.astype(np.int32),
        "max_importance":  max_imp,
    })


# ─── Reamostragem para M30 e M15 ─────────────────────────────────────────────

def resample_from_h1(
    h1_df: pd.DataFrame,
    freq: str,
    label: str,
) -> pd.DataFrame:
    """
    Reamostrar H1 para timeframe inferior via forward-fill.

    - sentiment_score  : ffill (sentimento da última hora válida)
    - event_count      : 0 em timestamps intermediários
    - max_importance   : 0 em timestamps intermediários
    """
    tf_index = pd.date_range(start=DATE_START, end=DATE_END, freq=freq)

    base = h1_df.set_index("datetime")

    # Reindexar e aplicar ffill no sentiment_score
    tf_df = base.reindex(tf_index)
    tf_df["sentiment_score"] = tf_df["sentiment_score"].ffill().fillna(0.0)
    tf_df["event_count"]     = tf_df["event_count"].fillna(0).astype(np.int32)
    tf_df["max_importance"]  = tf_df["max_importance"].fillna(0.0)
    tf_df.index.name = "datetime"

    return tf_df.reset_index()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("ETAPA 3 — Aggregate Sentiment por Par Forex")
    print("=" * 65)

    events = load_events()

    # Filtra para a janela de sobreposição
    events = events[
        (events["event_datetime"] >= DATE_START) &
        (events["event_datetime"] <= DATE_END)
    ].copy()
    print(
        f"\nEventos no range {DATE_START.date()} → {DATE_END.date()}: "
        f"{len(events):,}"
    )

    # Índice horário base
    hourly_index = pd.date_range(start=DATE_START, end=DATE_END, freq="h")
    print(f"Índice horário: {len(hourly_index):,} horas\n")

    summary = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for pair, currency_signs in PAIR_CURRENCY_MAP.items():
        print(f"\n{'=' * 65}")
        print(f"Par: {pair}  |  Moedas: {[c for c, _ in currency_signs]}")
        print("-" * 65)

        # Filtra eventos das moedas relevantes e aplica sinal direcional
        pair_dfs = []
        for currency, sign in currency_signs:
            cur_ev = events[events["currency"] == currency].copy()
            cur_ev["adjusted_score"] = cur_ev["sentiment_score"] * sign
            pair_dfs.append(cur_ev)
            cnt = len(cur_ev)
            print(f"  {currency}  (sinal={sign:+.1f}) : {cnt:,} eventos")

        if not pair_dfs:
            print(f"  [AVISO] Nenhum evento para {pair}, pulando.")
            continue

        pair_events = (
            pd.concat(pair_dfs, ignore_index=True)
            .sort_values("event_datetime")
            .reset_index(drop=True)
        )

        # ── Calcula H1 ────────────────────────────────────────────────────
        h1_df = compute_pair_h1(pair_events, hourly_index, pair)

        h1_path  = OUTPUT_DIR / f"{pair}_sentiment_H1.csv"
        m30_path = OUTPUT_DIR / f"{pair}_sentiment_M30.csv"
        m15_path = OUTPUT_DIR / f"{pair}_sentiment_M15.csv"

        h1_df.to_csv(h1_path, index=False)
        print(f"  Salvo: {h1_path.name}")

        # ── Reamostragem M30 / M15 ────────────────────────────────────────
        m30_df = resample_from_h1(h1_df, freq="30min", label="M30")
        m30_df.to_csv(m30_path, index=False)
        print(f"  Salvo: {m30_path.name}  ({len(m30_df):,} linhas)")

        m15_df = resample_from_h1(h1_df, freq="15min", label="M15")
        m15_df.to_csv(m15_path, index=False)
        print(f"  Salvo: {m15_path.name}  ({len(m15_df):,} linhas)")

        # ── Relatório de cobertura ────────────────────────────────────────
        covered  = int((h1_df["event_count"] > 0).sum())
        total    = len(h1_df)
        pct      = covered / total * 100
        mean_s   = float(h1_df["sentiment_score"].mean())
        std_s    = float(h1_df["sentiment_score"].std())
        bull_pct = float((h1_df["sentiment_score"] >  0.1).mean() * 100)
        bear_pct = float((h1_df["sentiment_score"] < -0.1).mean() * 100)

        print(f"\n  Relatório {pair}:")
        print(f"    Total de horas no índice  : {total:,}")
        print(f"    Horas com ≥1 evento        : {covered:,}  ({pct:.1f} %)")
        print(f"    Score médio                : {mean_s:+.5f}")
        print(f"    Desvio padrão              : {std_s:.5f}")
        print(f"    Horas bullish (> 0.1)      : {bull_pct:.1f} %")
        print(f"    Horas bearish (< -0.1)     : {bear_pct:.1f} %")

        # Top eventos high-impact (por adjusted_score absoluto)
        high_ev = pair_events[pair_events["importance"] == "high"].copy()
        high_ev["abs_adj"] = high_ev["adjusted_score"].abs()
        top5 = high_ev.nlargest(5, "abs_adj")

        if not top5.empty:
            print(f"    Top eventos high-impact (maior |adjusted_score|):")
            for _, e in top5.iterrows():
                dt_str  = e["event_datetime"].strftime("%Y-%m-%d %H:%M")
                name_s  = str(e["event_name"])[:48]
                print(
                    f"      [{dt_str}] {e['currency']}  "
                    f"{name_s:<48}  adj={e['adjusted_score']:+.3f}"
                )

        # ── Accumula summary ──────────────────────────────────────────────
        summary[pair] = {
            "total_hours":        total,
            "hours_with_events":  covered,
            "coverage_pct":       round(pct, 2),
            "mean_score":         round(mean_s, 6),
            "std_score":          round(std_s, 6),
            "bullish_pct":        round(bull_pct, 2),
            "bearish_pct":        round(bear_pct, 2),
            "total_events":       int(len(pair_events)),
            "currencies": {
                c: int((pair_events["currency"] == c).sum())
                for c, _ in currency_signs
            },
            "files": {
                "H1":  str(h1_path),
                "M30": str(m30_path),
                "M15": str(m15_path),
            },
        }

    # ── Salva summary ──────────────────────────────────────────────────────
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 65}")
    print(f"Summary salvo em: {SUMMARY_PATH}")
    print("Etapa 3 concluída com sucesso!")
    print("=" * 65)

    # ── Tabela-resumo final ────────────────────────────────────────────────
    print("\nResumo consolidado:")
    header = f"{'Par':<8} {'Horas':>8} {'Cobertura':>10} {'Score μ':>10} {'Score σ':>10}"
    print(header)
    print("-" * len(header))
    for pair, s in summary.items():
        print(
            f"{pair:<8} {s['total_hours']:>8,} "
            f"{s['coverage_pct']:>9.1f}%  "
            f"{s['mean_score']:>+10.5f}  "
            f"{s['std_score']:>10.5f}"
        )


if __name__ == "__main__":
    main()
