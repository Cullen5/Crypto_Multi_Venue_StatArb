"""
Binance Top-100 Data Loader
============================
Loads all downloaded parquet files into a single wide DataFrame
with a complete 15-min datetime index (outer join). Missing bars are NaN.

Usage:
    from loader import load_binance_data
    df = load_binance_data("./data")
"""

import pandas as pd
from pathlib import Path


def load_binance_data(
    data_dir: str = "./data",
    price_col: str = "close",
    volume: bool = True,
) -> pd.DataFrame:
    """
    Load all parquet files into a single wide DataFrame.

    Parameters
    ----------
    data_dir : str
        Path to the data directory (containing klines/ folder).
    price_col : str
        Which price column to use: 'open', 'high', 'low', 'close'.
    volume : bool
        If True, also includes quote_volume columns as {symbol}_volume.

    Returns
    -------
    pd.DataFrame
        Index: open_time (UTC, 15-min freq)
        Columns: one per symbol (price), optionally {symbol}_volume
    """
    kline_dir = Path(data_dir) / "klines"
    if not kline_dir.exists():
        raise FileNotFoundError(f"No klines directory found at {kline_dir}")

    # Collect all parquet files grouped by symbol
    symbol_files: dict[str, list[Path]] = {}
    for f in sorted(kline_dir.rglob("*.parquet")):
        symbol = f.stem  # e.g. BTCUSDT
        symbol_files.setdefault(symbol, []).append(f)

    print(f"Found {len(symbol_files)} unique symbols across {len(list(kline_dir.iterdir()))} months\n")

    # Load each symbol, concat across months, deduplicate
    series_price = {}
    series_volume = {}
    stats = []

    for symbol in sorted(symbol_files):
        dfs = [pd.read_parquet(f) for f in symbol_files[symbol]]
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset="open_time").sort_values("open_time").set_index("open_time")

        series_price[symbol] = df[price_col]
        if volume:
            series_volume[f"{symbol}_volume"] = df["quote_volume"]

        stats.append({
            "symbol": symbol,
            "start": df.index.min(),
            "end": df.index.max(),
            "bars": len(df),
        })

    # Combine into wide df with outer join (union of all timestamps)
    print("Joining all symbols on datetime index...")
    combined = pd.DataFrame(series_price)
    if volume:
        vol_df = pd.DataFrame(series_volume)
        combined = pd.concat([combined, vol_df], axis=1)

    # Ensure complete 15-min grid from global min to global max
    full_index = pd.date_range(
        start=combined.index.min(),
        end=combined.index.max(),
        freq="15min",
        tz="UTC",
        name="open_time",
    )
    combined = combined.reindex(full_index)

    # Print diagnostics
    print(f"\n{'='*70}")
    print(f"Combined DataFrame: {combined.shape[0]:,} rows × {combined.shape[1]:,} columns")
    print(f"Date range: {combined.index.min()} → {combined.index.max()}")
    print(f"{'='*70}\n")

    # Per-symbol stats
    price_cols = [c for c in combined.columns if not c.endswith("_volume")]
    stat_rows = []
    for symbol in price_cols:
        col = combined[symbol]
        first_valid = col.first_valid_index()
        last_valid = col.last_valid_index()
        if first_valid is None:
            stat_rows.append((symbol, "no data", "no data", 0, 0, "N/A"))
            continue

        in_range = col.loc[first_valid:last_valid]
        missing = int(in_range.isna().sum())
        total = len(in_range)
        pct = missing / total * 100 if total > 0 else 0

        stat_rows.append((
            symbol,
            str(first_valid.date()),
            str(last_valid.date()),
            total,
            missing,
            f"{pct:.1f}%",
        ))

    stat_df = pd.DataFrame(stat_rows, columns=["symbol", "start", "end", "bars", "missing", "missing%"])
    stat_df = stat_df.sort_values("start")

    print(stat_df.to_string(index=False))
    print(f"\nTotal symbols: {len(price_cols)}")
    print(f"Symbols with >1% missing: {len(stat_df[stat_df['missing%'].str.rstrip('%').astype(float) > 1])}")

    return combined


def load_binance_for_backtester(
    data_dir: str = "./data",
    venue_name: str = "binance",
) -> dict:
    """
    Load parquet files into the nested dict format the Backtester expects.

    Returns
    -------
    dict
        {venue_name: {symbol: DataFrame}} where each DataFrame has columns
        'close_ohlcv', 'open', 'high', 'low', 'volume', 'quote_volume'
        and a 'timestamp' datetime index (named 'open_time').
    """
    kline_dir = Path(data_dir) / "klines"
    if not kline_dir.exists():
        raise FileNotFoundError(f"No klines directory found at {kline_dir}")

    symbol_files: dict[str, list[Path]] = {}
    for f in sorted(kline_dir.rglob("*.parquet")):
        symbol = f.stem
        symbol_files.setdefault(symbol, []).append(f)

    print(f"Found {len(symbol_files)} symbols")

    venue_data = {}
    for symbol in sorted(symbol_files):
        dfs = [pd.read_parquet(f) for f in symbol_files[symbol]]
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset="open_time").sort_values("open_time")

        # Rename 'close' -> 'close_ohlcv' so backtester can find it
        if "close" in df.columns and "close_ohlcv" not in df.columns:
            df = df.rename(columns={"close": "close_ohlcv"})

        # Backtester expects 'timestamp' column or index
        df = df.rename(columns={"open_time": "timestamp"})

        # Strip quote currency suffix so keys match SECTOR_MAPPING (e.g. "BTCUSDT" -> "BTC")
        short = symbol.removesuffix("USDT")
        venue_data[short] = df

    print(f"Loaded {len(venue_data)} symbols for venue '{venue_name}'")
    return {venue_name: venue_data}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Binance top-100 data")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--price", default="close", choices=["open", "high", "low", "close"])
    parser.add_argument("--no-volume", action="store_true", help="Exclude volume columns")
    args = parser.parse_args()

    df = load_binance_data(args.data_dir, price_col=args.price, volume=not args.no_volume)

    print(f"\nSample (last 5 rows, first 6 cols):")
    print(df.iloc[-5:, :6])
