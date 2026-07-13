import re
from pathlib import Path

import pandas as pd

TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}

# Sorted longest-first so "M15" is matched before "M1"
_TF_SORTED = sorted(TIMEFRAME_MINUTES, key=len, reverse=True)

_MT4_DATE_FMT = "%Y.%m.%d %H:%M"


def parse_filename(path: "Path | str") -> tuple[str, str]:
    """Return (symbol, timeframe) from a filename like EURUSDM15.csv."""
    stem = Path(path).stem.upper()
    for tf in _TF_SORTED:
        if stem.endswith(tf):
            return stem[: -len(tf)], tf
    raise ValueError(f"Cannot extract timeframe from filename '{Path(path).name}'")


def load_csv(path: "Path | str") -> pd.DataFrame:
    """Load a single MT4/MT5 OHLCV CSV.

    Handles UTF-16 BOM (common in MetaTrader exports) and the headerless
    format: <YYYY.MM.DD HH:MM>,Open,High,Low,Close,Volume[,Spread]

    Returns a DataFrame with a UTC DatetimeIndex and columns
    Open/High/Low/Close/Volume (float32) plus a zeroed 'sentiment' column
    ready for later enrichment.
    """
    path = Path(path)
    df: pd.DataFrame | None = None

    for encoding in ("utf-16", "utf-8-sig", "utf-8", "latin-1"):
        try:
            raw = pd.read_csv(
                path,
                encoding=encoding,
                header=None,
                dtype=str,
            )
            if not raw.empty:
                df = raw
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as exc:
            raise IOError(f"Unexpected error reading {path}: {exc}") from exc

    if df is None:
        raise IOError(f"Could not decode {path} with any supported encoding")

    # Keep only the first 6 columns (date, O, H, L, C, Volume); ignore Spread
    df = df.iloc[:, :6].copy()
    df.columns = ["datetime", "Open", "High", "Low", "Close", "Volume"]

    df["datetime"] = pd.to_datetime(
        df["datetime"].str.strip(), format=_MT4_DATE_FMT, errors="coerce"
    )
    df = df.dropna(subset=["datetime", "Open", "High", "Low", "Close"])
    df = df.set_index("datetime")

    # Localise to UTC (MT4 exports are broker-time but we treat as UTC for consistency)
    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["sentiment"] = 0.0

    return df
