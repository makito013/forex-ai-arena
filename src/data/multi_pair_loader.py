"""
Scans data/historical/, loads all MT4/MT5 CSVs, and returns train/val/test
splits keyed by (symbol, timeframe).

Usage:
    from src.data.multi_pair_loader import load_all, DataSplit

    data = load_all(symbols=["EURUSD", "GBPUSD"], timeframes=["M15", "H1"])
    split: DataSplit = data[("EURUSD", "M15")]
    print(split.train.head())
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.csv_loader import TIMEFRAME_MINUTES, load_csv, parse_filename


@dataclass
class DataSplit:
    symbol: str
    timeframe: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def full(self) -> pd.DataFrame:
        """All rows in chronological order (no shuffling)."""
        return pd.concat([self.train, self.val, self.test])

    def summary(self) -> str:
        def _fmt(df: pd.DataFrame) -> str:
            if df.empty:
                return "empty"
            return f"{len(df):>7,} rows  {df.index[0].date()} → {df.index[-1].date()}"

        return (
            f"{self.symbol}/{self.timeframe}\n"
            f"  train : {_fmt(self.train)}\n"
            f"  val   : {_fmt(self.val)}\n"
            f"  test  : {_fmt(self.test)}"
        )


def load_all(
    data_dir: "str | Path" = "data/historical",
    symbols: "list[str] | None" = None,
    timeframes: "list[str] | None" = None,
    split: tuple[float, float, float] = (0.70, 0.15, 0.15),
    min_rows: int = 500,
) -> dict[tuple[str, str], DataSplit]:
    """Load every CSV in *data_dir* and return chronological train/val/test splits.

    Args:
        data_dir:   directory that contains the MT4/MT5 CSV files.
        symbols:    whitelist of symbols to load (e.g. ['EURUSD']). None = all.
        timeframes: whitelist of timeframes (e.g. ['M15', 'H1']). None = all.
        split:      (train_pct, val_pct, test_pct) — must sum to 1.0.
        min_rows:   CSVs with fewer rows than this are silently skipped.

    Returns:
        dict keyed by (symbol, timeframe) → DataSplit.
    """
    if abs(sum(split) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {sum(split):.4f}")

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir.resolve()}")

    _sym_filter = {s.upper() for s in symbols} if symbols else None
    _tf_filter = {t.upper() for t in timeframes} if timeframes else None

    result: dict[tuple[str, str], DataSplit] = {}
    skipped: list[str] = []

    for csv_path in sorted(data_dir.glob("*.csv")):
        try:
            symbol, timeframe = parse_filename(csv_path)
        except ValueError:
            skipped.append(f"{csv_path.name} (unknown timeframe)")
            continue

        if _sym_filter and symbol not in _sym_filter:
            continue
        if _tf_filter and timeframe not in _tf_filter:
            continue

        try:
            df = load_csv(csv_path)
        except Exception as exc:
            skipped.append(f"{csv_path.name} ({exc})")
            continue

        if len(df) < min_rows:
            skipped.append(f"{csv_path.name} (only {len(df)} rows)")
            continue

        n = len(df)
        i_val = int(n * split[0])
        i_test = int(n * (split[0] + split[1]))

        result[(symbol, timeframe)] = DataSplit(
            symbol=symbol,
            timeframe=timeframe,
            train=df.iloc[:i_val].copy(),
            val=df.iloc[i_val:i_test].copy(),
            test=df.iloc[i_test:].copy(),
        )

    if skipped:
        print(f"[multi_pair_loader] skipped {len(skipped)} file(s):")
        for s in skipped:
            print(f"  • {s}")

    return result


def available_symbols(data_dir: "str | Path" = "data/historical") -> list[str]:
    """Sorted list of unique symbols found in *data_dir*."""
    symbols: set[str] = set()
    for p in Path(data_dir).glob("*.csv"):
        try:
            sym, _ = parse_filename(p)
            symbols.add(sym)
        except ValueError:
            pass
    return sorted(symbols)


def available_timeframes(data_dir: "str | Path" = "data/historical") -> list[str]:
    """Timeframes present in *data_dir*, sorted from shortest to longest."""
    tfs: set[str] = set()
    for p in Path(data_dir).glob("*.csv"):
        try:
            _, tf = parse_filename(p)
            tfs.add(tf)
        except ValueError:
            pass
    return sorted(tfs, key=lambda t: TIMEFRAME_MINUTES.get(t, 9999))


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    print("Scanning data/historical/ ...\n")
    data = load_all()

    print(f"\nLoaded {len(data)} dataset(s):\n")
    for key, split in sorted(data.items()):
        print(split.summary())

    print(f"\nSymbols  : {available_symbols()}")
    print(f"Timeframes: {available_timeframes()}")
