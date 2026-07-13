#!/usr/bin/env python3
"""
Step 2 — Score Events
Pipeline de preparação de dados para treino RL Forex.

Calcula sentiment_score para cada evento econômico:
- surprise_formula: quando actual, forecast e previous disponíveis
- actual_vs_previous: quando actual e previous disponíveis (sem forecast)
- no_baseline: quando só actual disponível
- claude_haiku: quando nenhum dado numérico disponível (chama API)
"""

import os
import json
import sqlite3
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/bruno.andrade/projetos/pessoal/estatistica/forex-ai-arena")
CALENDAR_CSV  = BASE_DIR / "data/news/processed/calendar_clean.csv"
DB_PATH       = BASE_DIR / "data/news/sentiment_cache.db"
OUTPUT_CSV    = BASE_DIR / "data/news/processed/events_scored.csv"
SUMMARY_JSON  = BASE_DIR / "data/processed/step2_summary.json"

# Janela de treino (mesma da Etapa 1)
WINDOW_START = "2023-08-13"
WINDOW_END   = "2025-10-01"

# Limite de chamadas Claude (por nome único de evento)
MAX_CLAUDE_CALLS = 100
BATCH_SIZE       = 50


# ─── Banco de dados ────────────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS event_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_datetime TEXT NOT NULL,
            currency TEXT NOT NULL,
            event_name TEXT NOT NULL,
            importance TEXT,
            actual REAL,
            forecast REAL,
            previous REAL,
            sentiment_score REAL NOT NULL,
            method TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def save_batch(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany("""
        INSERT INTO event_sentiment
            (event_datetime, currency, event_name, importance,
             actual, forecast, previous, sentiment_score, method)
        VALUES
            (:event_datetime, :currency, :event_name, :importance,
             :actual, :forecast, :previous, :sentiment_score, :method)
    """, rows)
    conn.commit()


# ─── Scoring por fórmula ──────────────────────────────────────────────────────
def clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def score_numeric(actual, forecast, previous) -> tuple[float, str]:
    """Retorna (sentiment_score, method)."""
    has_actual   = actual   is not None and not pd.isna(actual)
    has_forecast = forecast is not None and not pd.isna(forecast)
    has_previous = previous is not None and not pd.isna(previous)

    if has_actual and has_forecast and has_previous:
        surprise = (actual - forecast) / max(abs(previous), 0.001)
        score    = clamp(surprise * 2.0)
        return score, "surprise_formula"

    if has_actual and has_previous and not has_forecast:
        raw   = (actual - previous) / max(abs(previous), 0.001)
        score = clamp(raw * 2.0)
        return score, "surprise_formula"   # mesma família de fórmula

    if has_actual:
        return 0.0, "no_baseline"

    return None, "claude_haiku"


# ─── Scoring via Claude Haiku ─────────────────────────────────────────────────
def normalize_event_name(name: str) -> str:
    """Normaliza nome do evento para usar como chave de cache."""
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    # Remove sufixos de período entre parênteses: (Jan), (Q1), (MoM), etc.
    name = re.sub(r"\s*\([^)]+\)\s*$", "", name)
    return name


def call_claude(client, event: str, currency: str) -> float:
    """Chama Claude Haiku e retorna float [-1, 1]."""
    prompt = (
        f"You are a Forex analyst. Rate this economic event's typical market impact. "
        f"Event: {event} ({currency}). "
        f"Return ONLY a float from -1.0 (strongly bearish) to 1.0 (strongly bullish) "
        f"based on what this event typically signals. No text, just the number."
    )
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        text = message.content[0].text.strip()
        # Extrai primeiro número float do texto (defesa contra texto extra)
        match = re.search(r"-?\d+\.?\d*", text)
        if match:
            return clamp(float(match.group()))
        return 0.0
    except Exception as e:
        print(f"  [WARN] Claude error for '{event}': {e}")
        return 0.0


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Etapa 2 — Score Events")
    print("=" * 65)

    # 1. Carrega calendário
    print(f"\n[1] Carregando calendário: {CALENDAR_CSV}")
    df = pd.read_csv(CALENDAR_CSV, low_memory=False)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)

    # Filtra janela de treino
    mask = (df["datetime_utc"] >= WINDOW_START) & (df["datetime_utc"] <= WINDOW_END)
    df   = df[mask].reset_index(drop=True)
    print(f"    Eventos na janela [{WINDOW_START} → {WINDOW_END}]: {len(df):,}")

    # 2. Inicia banco
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # 3. Identifica linhas que precisam do Claude
    print("\n[2] Classificando eventos por método de scoring…")
    needs_claude_idx = []
    for i, row in df.iterrows():
        _, method = score_numeric(row["actual_num"], row["forecast_num"], row["previous_num"])
        if method == "claude_haiku":
            needs_claude_idx.append(i)

    print(f"    Eventos com fórmula numérica: {len(df) - len(needs_claude_idx):,}")
    print(f"    Eventos que precisam Claude:  {len(needs_claude_idx):,}")

    # 4. Monta cache de nomes únicos para Claude
    unique_names: dict[str, float] = {}  # norm_name → score
    claude_scored: set[str] = set()      # nomes realmente pontuados pelo Claude
    claude_map:    dict[int, float] = {} # idx → score
    claude_tokens_in  = 0
    claude_tokens_out = 0
    claude_calls      = 0

    names_to_score: list[tuple[str, str, str]] = []  # (norm_name, raw_event, currency)
    seen_norm: set[str] = set()
    for i in needs_claude_idx:
        row       = df.iloc[i]
        norm      = normalize_event_name(str(row["event"]))
        currency  = str(row["currency"])
        if norm not in seen_norm:
            seen_norm.add(norm)
            names_to_score.append((norm, str(row["event"]), currency))

    print(f"    Nomes únicos para Claude:     {len(names_to_score):,}")
    print(f"    Limite de chamadas:           {MAX_CLAUDE_CALLS}")

    # 5. Chama Claude (com limite)
    if names_to_score:
        import anthropic
        try:
            # O SDK detecta a chave automaticamente (env var, ~/.anthropic, etc.)
            client = anthropic.Anthropic()
            # Teste rápido para validar a chave
            client.models.list(limit=1)
            claude_available = True
        except Exception as e:
            print(f"\n  [WARN] Anthropic API indisponível: {e}")
            print("  Eventos sem dados numéricos serão marcados como no_baseline (score=0.0)")
            claude_available = False

        if not claude_available:
            for norm, _, _ in names_to_score:
                unique_names[norm] = 0.0
        else:
            print(f"\n[3] Chamando Claude Haiku para {min(len(names_to_score), MAX_CLAUDE_CALLS)} eventos únicos…")
            for norm, raw_event, currency in names_to_score:
                if claude_calls >= MAX_CLAUDE_CALLS:
                    print(f"  [INFO] Limite de {MAX_CLAUDE_CALLS} chamadas atingido — restantes → 0.0")
                    unique_names[norm] = 0.0
                    continue

                score = call_claude(client, raw_event, currency)
                unique_names[norm] = score
                claude_scored.add(norm)
                claude_calls += 1

                # Estimativa de tokens: ~80 in, ~5 out por chamada
                claude_tokens_in  += 80
                claude_tokens_out += 5

                if claude_calls % 10 == 0:
                    print(f"    … {claude_calls}/{min(len(names_to_score), MAX_CLAUDE_CALLS)} chamadas concluídas")

                # Pequeno delay para não sobrecarregar rate limit
                if claude_calls < MAX_CLAUDE_CALLS:
                    time.sleep(0.1)
    else:
        print("\n[3] Nenhum evento precisa do Claude.")

    # Mapeia índices → (score, foi_pontuado_pelo_claude)
    for i in needs_claude_idx:
        row  = df.iloc[i]
        norm = normalize_event_name(str(row["event"]))
        claude_map[i] = (unique_names.get(norm, 0.0), norm in claude_scored)

    # 6. Processa todos os eventos em batches
    print(f"\n[4] Processando {len(df):,} eventos em batches de {BATCH_SIZE}…")
    results = []
    batch   = []

    counters = {"surprise_formula": 0, "claude_haiku": 0, "no_baseline": 0}

    for i, row in df.iterrows():
        score, method = score_numeric(row["actual_num"], row["forecast_num"], row["previous_num"])

        if method == "claude_haiku":
            entry = claude_map.get(i, (0.0, False))
            score, was_scored = entry
            if not was_scored:
                method = "no_baseline"
                score  = 0.0

        counters[method] += 1

        record = {
            "event_datetime": row["datetime_utc"].isoformat(),
            "currency":       str(row["currency"]),
            "event_name":     str(row["event"]),
            "importance":     str(row["importance"]),
            "actual":         row["actual_num"] if pd.notna(row["actual_num"]) else None,
            "forecast":       row["forecast_num"] if pd.notna(row["forecast_num"]) else None,
            "previous":       row["previous_num"] if pd.notna(row["previous_num"]) else None,
            "sentiment_score": round(score, 6),
            "method":          method,
        }
        batch.append(record)
        results.append(record)

        if len(batch) >= BATCH_SIZE:
            save_batch(conn, batch)
            batch.clear()

    # Flush último batch
    if batch:
        save_batch(conn, batch)

    conn.close()
    print(f"    Banco salvo: {DB_PATH}")

    # 7. Salva CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"    CSV salvo:   {OUTPUT_CSV}")

    # 8. Análise de distribuição
    scores = [r["sentiment_score"] for r in results]
    bearish = sum(1 for s in scores if s < -0.2)
    neutral = sum(1 for s in scores if -0.2 <= s <= 0.2)
    bullish = sum(1 for s in scores if s > 0.2)
    total   = len(scores)

    # 9. Top 10 extremos
    df_out_sorted = df_out.sort_values("sentiment_score")
    top_negative  = df_out_sorted.head(5)[["event_datetime", "currency", "event_name", "sentiment_score", "method"]].to_dict("records")
    top_positive  = df_out_sorted.tail(5).sort_values("sentiment_score", ascending=False)[["event_datetime", "currency", "event_name", "sentiment_score", "method"]].to_dict("records")
    top10_extreme = top_negative + top_positive

    # Custo estimado Claude Haiku (preços aproximados)
    # Input: $0.80/MTok, Output: $4.00/MTok
    estimated_cost_usd = (claude_tokens_in / 1_000_000 * 0.80) + (claude_tokens_out / 1_000_000 * 4.00)

    # 10. Relatório
    print("\n" + "=" * 65)
    print("  RELATÓRIO FINAL — Etapa 2")
    print("=" * 65)
    print(f"\n  Total de eventos processados: {total:,}")
    print(f"\n  Breakdown por método:")
    print(f"    surprise_formula : {counters['surprise_formula']:>6,}  ({counters['surprise_formula']/total*100:.1f}%)")
    print(f"    claude_haiku     : {counters['claude_haiku']:>6,}  ({counters['claude_haiku']/total*100:.1f}%)")
    print(f"    no_baseline      : {counters['no_baseline']:>6,}  ({counters['no_baseline']/total*100:.1f}%)")
    print(f"\n  Distribuição de scores:")
    print(f"    Bearish  (< -0.2) : {bearish:>6,}  ({bearish/total*100:.1f}%)")
    print(f"    Neutral  (-0.2/0.2): {neutral:>6,}  ({neutral/total*100:.1f}%)")
    print(f"    Bullish  (>  0.2) : {bullish:>6,}  ({bullish/total*100:.1f}%)")
    print(f"\n  Top 10 eventos mais extremos:")
    for r in top10_extreme:
        print(f"    [{r['sentiment_score']:+.3f}] {r['currency']} | {r['event_name'][:45]:<45} | {str(r['event_datetime'])[:19]}")
    print(f"\n  Claude Haiku:")
    print(f"    Chamadas realizadas : {claude_calls}")
    print(f"    Tokens estimados    : {claude_tokens_in:,} in / {claude_tokens_out:,} out")
    print(f"    Custo estimado      : ${estimated_cost_usd:.4f} USD")

    # 11. Salva JSON de métricas
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "step": 2,
        "generated_at": datetime.utcnow().isoformat(),
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "total_events": total,
        "methods": counters,
        "score_distribution": {
            "bearish_pct":  round(bearish / total * 100, 2),
            "neutral_pct":  round(neutral / total * 100, 2),
            "bullish_pct":  round(bullish / total * 100, 2),
        },
        "claude": {
            "model": "claude-haiku-4-5-20251001",
            "calls": claude_calls,
            "unique_event_names": len(names_to_score) if "names_to_score" in dir() else 0,
            "tokens_in_estimated": claude_tokens_in,
            "tokens_out_estimated": claude_tokens_out,
            "estimated_cost_usd": round(estimated_cost_usd, 6),
        },
        "outputs": {
            "db":  str(DB_PATH),
            "csv": str(OUTPUT_CSV),
        },
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  JSON salvo: {SUMMARY_JSON}")
    print("\n  Etapa 2 concluída com sucesso!")
    print("=" * 65)


if __name__ == "__main__":
    main()
