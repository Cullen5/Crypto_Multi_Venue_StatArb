"""
Binance Top-100 Liquidity 15-Minute OHLCV Data Collecter
Pulls 15-minute kline data for the top 100 most liquid USDT pairs on Binance,
re-evaluated at the start of each month. Once a coin enters the top 100, it
stays in the collection as long as it:
  - Is not delisted from Binance
  - Maintains > MIN_DAILY_VOLUME_USD average daily quote volume

This means the active universe grows over time: each month = top 100 fresh +
all carry-over coins still meeting the threshold.

Approach:
1. For each month, fetch daily klines for ALL USDT pairs over a ranking window
2. Rank by quote volume (USDT volume), take top 100 → these are the "new entrants"
3. Also check volume for all coins carried from the previous month
4. Active universe = new top 100 added to carry-over coins with vol > threshold
5. Fetch full 15-min kline data for the active universe for the entire month
6. Save per-month parquet files + a universe mapping CSV

Output:
  data/
    universe.csv              — month → symbol mapping with status (new/carry-over/dropped)
    klines/
      2023-01/BTCUSDT.parquet — 15-min OHLCV per symbol per month
      2023-01/ETHUSDT.parquet
      ...

Usage:
    python binance_top100_15m.py --start 2023-01 --end 2025-05 --output ./data
    python binance_top100_15m.py --start 2023-01 --end 2025-05 --output ./data --top-n 50
    python binance_top100_15m.py --start 2023-01 --end 2025-05 --output ./data --min-volume 5000000
"""

import argparse
import os
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = "https://api.binance.com"
KLINE_LIMIT = 1000  # max candles per request
REQUEST_DELAY = 0.12  # ~8 req/s, well within 1200/min limit
DAILY_RANKING_WINDOW = 7  # days of daily candles to average for ranking
MIN_DAILY_VOLUME_USD = 2_000_000  # $2M minimum to keep a carry-over coin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Binance API helpers
# ---------------------------------------------------------------------------
session = requests.Session()
session.headers.update({"User-Agent": "BinanceTop100Puller/1.0"})


def _get(endpoint: str, params: dict, retries: int = 5) -> Optional[dict | list]:
    """GET with retry + backoff for rate limits."""
    for attempt in range(retries):
        try:
            r = session.get(f"{BASE_URL}{endpoint}", params=params, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 30))
                log.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code == 418:  # IP ban
                log.error("IP banned by Binance. Wait and retry later.")
                time.sleep(120)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            log.warning(f"Request error (attempt {attempt+1}/{retries}): {e}. Retrying in {wait}s")
            time.sleep(wait)
    log.error(f"Failed after {retries} retries: {endpoint} {params}")
    return None


def get_all_usdt_symbols() -> list[dict]:
    """Get all active USDT spot trading pairs from exchange info."""
    info = _get("/api/v3/exchangeInfo", {})
    if not info:
        raise RuntimeError("Failed to fetch exchange info")

    symbols = []
    for s in info["symbols"]:
        if (
            s["quoteAsset"] == "USDT"
            and s["status"] == "TRADING"
            and s["isSpotTradingAllowed"]
        ):
            symbols.append({
                "symbol": s["symbol"],
                "baseAsset": s["baseAsset"],
            })
    log.info(f"Found {len(symbols)} active USDT spot pairs")
    return symbols


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    """Fetch klines between start and end, handling pagination."""
    all_klines = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": KLINE_LIMIT,
        }
        data = _get("/api/v3/klines", params)
        time.sleep(REQUEST_DELAY)

        if not data:
            break
        all_klines.extend(data)

        if len(data) < KLINE_LIMIT:
            break

        # Next batch starts after last candle
        current_start = data[-1][0] + 1

    return all_klines


def klines_to_df(klines: list[list], symbol: str) -> pd.DataFrame:
    """Convert raw kline data to a clean DataFrame."""
    if not klines:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume",
                    "taker_buy_volume", "taker_buy_quote_volume"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["trades"] = df["trades"].astype(int)
    df["symbol"] = symbol
    df.drop(columns=["ignore"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Volume check for a single symbol
# ---------------------------------------------------------------------------
def get_symbol_avg_volume(
    symbol: str,
    month_start: datetime,
    window_days: int = DAILY_RANKING_WINDOW,
) -> Optional[dict]:
    """
    Fetch average daily volume for a single symbol over the ranking window.
    Returns dict with volume info, or None if no data (delisted).
    """
    start_ms = int(month_start.timestamp() * 1000)
    end_ms = int((month_start + timedelta(days=window_days)).timestamp() * 1000)

    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": window_days,
    }
    data = _get("/api/v3/klines", params)
    time.sleep(REQUEST_DELAY)

    if not data or len(data) == 0:
        return None

    avg_qvol = sum(float(k[7]) for k in data) / len(data)
    avg_trades = sum(int(k[8]) for k in data) / len(data)

    return {
        "symbol": symbol,
        "avg_daily_quote_volume": avg_qvol,
        "avg_daily_trades": avg_trades,
        "days_with_data": len(data),
    }


# ---------------------------------------------------------------------------
# Ranking: determine top-N by liquidity for a given month
# ---------------------------------------------------------------------------
def rank_all_pairs_for_month(
    month_start: datetime,
) -> pd.DataFrame:
    """
    Rank ALL USDT pairs by average daily quote volume over the first
    DAILY_RANKING_WINDOW days of the month. Returns full DataFrame with
    symbol, avg_daily_quote_volume, rank for every pair with data.
    """
    start_ms = int(month_start.timestamp() * 1000)
    end_ms = int((month_start + timedelta(days=DAILY_RANKING_WINDOW)).timestamp() * 1000)

    all_symbols = get_all_usdt_symbols()
    symbol_names = [s["symbol"] for s in all_symbols]

    log.info(
        f"Ranking {len(symbol_names)} USDT pairs for {month_start.strftime('%Y-%m')} "
        f"using {DAILY_RANKING_WINDOW}-day volume window..."
    )

    volumes = []
    for i, sym in enumerate(symbol_names):
        if (i + 1) % 100 == 0:
            log.info(f"  Fetching volume {i+1}/{len(symbol_names)}...")

        params = {
            "symbol": sym,
            "interval": "1d",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": DAILY_RANKING_WINDOW,
        }
        data = _get("/api/v3/klines", params)
        time.sleep(REQUEST_DELAY)

        if data and len(data) > 0:
            avg_qvol = sum(float(k[7]) for k in data) / len(data)
            avg_trades = sum(int(k[8]) for k in data) / len(data)
            volumes.append({
                "symbol": sym,
                "avg_daily_quote_volume": avg_qvol,
                "avg_daily_trades": avg_trades,
                "days_with_data": len(data),
            })

    vol_df = pd.DataFrame(volumes)
    vol_df.sort_values("avg_daily_quote_volume", ascending=False, inplace=True)
    vol_df["rank"] = range(1, len(vol_df) + 1)

    return vol_df


def build_active_universe(
    month_start: datetime,
    full_ranking: pd.DataFrame,
    prev_month_symbols: set[str],
    top_n: int = 100,
    min_daily_volume: float = MIN_DAILY_VOLUME_USD,
) -> pd.DataFrame:
    """
    Build the active universe for a month:
      1. Top N by volume → always included (status = "top_n")
      2. Previous month's symbols not in top N → kept if volume > threshold
         (status = "carry_over")
      3. Previous month's symbols with insufficient volume → dropped
         (status = "dropped", not included in active set but logged)

    Returns DataFrame of active symbols with columns:
      symbol, rank, avg_daily_quote_volume, avg_daily_trades, status
    """
    month_str = month_start.strftime("%Y-%m")

    # Top N symbols
    top_n_df = full_ranking.head(top_n).copy()
    top_n_symbols = set(top_n_df["symbol"].tolist())
    top_n_df["status"] = "top_n"

    # Carry-over candidates: were in previous month but not in current top N
    carry_candidates = prev_month_symbols - top_n_symbols

    carry_rows = []
    dropped_rows = []

    if carry_candidates:
        log.info(
            f"  Checking {len(carry_candidates)} carry-over candidates "
            f"(min vol: ${min_daily_volume:,.0f}/day)..."
        )

        for sym in sorted(carry_candidates):
            # Check if this symbol appears in the full ranking (i.e., it has data)
            sym_row = full_ranking[full_ranking["symbol"] == sym]

            if sym_row.empty:
                # Symbol not found in ranking data — likely delisted
                dropped_rows.append({
                    "symbol": sym,
                    "rank": None,
                    "avg_daily_quote_volume": 0,
                    "avg_daily_trades": 0,
                    "status": "dropped_delisted",
                    "days_with_data": 0,
                })
                log.info(f"    {sym}: DROPPED (delisted / no data)")
                continue

            row = sym_row.iloc[0]
            vol = row["avg_daily_quote_volume"]

            if vol >= min_daily_volume:
                carry_rows.append({
                    "symbol": sym,
                    "rank": int(row["rank"]),
                    "avg_daily_quote_volume": vol,
                    "avg_daily_trades": row["avg_daily_trades"],
                    "status": "carry_over",
                    "days_with_data": row["days_with_data"],
                })
                log.info(f"    {sym}: KEPT (rank #{int(row['rank'])}, ${vol:,.0f}/day)")
            else:
                dropped_rows.append({
                    "symbol": sym,
                    "rank": int(row["rank"]),
                    "avg_daily_quote_volume": vol,
                    "avg_daily_trades": row["avg_daily_trades"],
                    "status": "dropped_low_volume",
                    "days_with_data": row["days_with_data"],
                })
                log.info(f"    {sym}: DROPPED (rank #{int(row['rank'])}, ${vol:,.0f}/day < threshold)")

    # Combine top N + carry-overs
    active_records = []
    for _, row in top_n_df.iterrows():
        active_records.append({
            "symbol": row["symbol"],
            "rank": int(row["rank"]),
            "avg_daily_quote_volume": row["avg_daily_quote_volume"],
            "avg_daily_trades": row["avg_daily_trades"],
            "status": "top_n",
            "days_with_data": row["days_with_data"],
        })
    active_records.extend(carry_rows)

    active_df = pd.DataFrame(active_records)

    # Log summary
    n_new = len(top_n_symbols - prev_month_symbols) if prev_month_symbols else len(top_n_symbols)
    n_carry = len(carry_rows)
    n_dropped = len(dropped_rows)

    log.info(
        f"  {month_str} universe: {len(active_df)} symbols total "
        f"({top_n} top-N, {n_carry} carry-over, {n_dropped} dropped, "
        f"{n_new} genuinely new)"
    )

    return active_df, dropped_rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def month_range(start: str, end: str) -> list[datetime]:
    """Generate first-of-month datetimes from 'YYYY-MM' strings."""
    s = datetime.strptime(start, "%Y-%m").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m").replace(tzinfo=timezone.utc)

    months = []
    current = s
    while current <= e:
        months.append(current)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def month_end(dt: datetime) -> datetime:
    """Get the last millisecond of a month."""
    if dt.month == 12:
        next_month = dt.replace(year=dt.year + 1, month=1)
    else:
        next_month = dt.replace(month=dt.month + 1)
    return next_month - timedelta(milliseconds=1)


def pull_month(
    month_start: datetime,
    top_symbols: list[str],
    output_dir: Path,
    skip_existing: bool = True,
) -> dict:
    """Pull 15-min klines for all top symbols for one month."""
    month_str = month_start.strftime("%Y-%m")
    kline_dir = output_dir / "klines" / month_str
    kline_dir.mkdir(parents=True, exist_ok=True)

    start_ms = int(month_start.timestamp() * 1000)
    end_ms = int(month_end(month_start).timestamp() * 1000)

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "empty": 0}

    for i, symbol in enumerate(top_symbols):
        out_file = kline_dir / f"{symbol}.parquet"

        if skip_existing and out_file.exists():
            stats["skipped"] += 1
            continue

        log.info(f"  [{month_str}] {i+1}/{len(top_symbols)} Fetching {symbol}...")

        klines = fetch_klines(symbol, "15m", start_ms, end_ms)

        if not klines:
            log.warning(f"  No data for {symbol} in {month_str} (may be delisted)")
            stats["empty"] += 1
            continue

        df = klines_to_df(klines, symbol)
        if df.empty:
            stats["empty"] += 1
            continue

        df.to_parquet(out_file, index=False, engine="pyarrow")
        stats["downloaded"] += 1

        log.debug(
            f"    {symbol}: {len(df)} candles, "
            f"{df['open_time'].min()} → {df['open_time'].max()}"
        )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Pull Binance top-100 liquidity 15-min OHLCV data (sticky universe)"
    )
    parser.add_argument("--start", required=True, help="Start month (YYYY-MM)")
    parser.add_argument("--end", required=True, help="End month inclusive (YYYY-MM)")
    parser.add_argument("--output", default="./data", help="Output directory")
    parser.add_argument("--top-n", type=int, default=100, help="Number of top pairs per month")
    parser.add_argument("--min-volume", type=float, default=MIN_DAILY_VOLUME_USD,
                        help="Min avg daily USD volume to keep a carry-over coin (default: 2000000)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip already downloaded files")
    parser.add_argument("--ranking-only", action="store_true",
                        help="Only compute rankings, don't download klines")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    months = month_range(args.start, args.end)
    log.info(f"Processing {len(months)} months: {args.start} → {args.end}")
    log.info(f"Top {args.top_n} pairs, min carry-over volume: ${args.min_volume:,.0f}/day")
    log.info(f"Output → {output_dir.resolve()}")

    # -----------------------------------------------------------------------
    # Phase 1: Build sticky universe for each month
    # -----------------------------------------------------------------------
    universe_records = []  # rows for universe.csv (active symbols)
    dropped_records = []   # rows for dropped.csv (symbols that fell off)
    month_active_symbols: dict[str, list[str]] = {}  # month_str → [symbols]

    universe_file = output_dir / "universe.csv"
    dropped_file = output_dir / "dropped.csv"

    # Load existing universe if resuming
    if universe_file.exists():
        log.info("Loading existing universe file...")
        existing_universe = pd.read_csv(universe_file)
        for _, row in existing_universe.iterrows():
            universe_records.append(row.to_dict())
            m = row["month"]
            if m not in month_active_symbols:
                month_active_symbols[m] = []
            month_active_symbols[m].append(row["symbol"])

    if dropped_file.exists():
        existing_dropped = pd.read_csv(dropped_file)
        dropped_records = existing_dropped.to_dict("records")

    # Determine which months still need ranking
    months_to_rank = [
        m for m in months
        if m.strftime("%Y-%m") not in month_active_symbols
    ]

    if months_to_rank:
        log.info(f"Need to rank {len(months_to_rank)} months...")

        # We need the previous month's active symbols to seed carry-overs.
        # If we're resuming, get the last known active set.
        prev_active: set[str] = set()

        if month_active_symbols:
            # Find the month just before the first month we need to rank
            all_known_months = sorted(month_active_symbols.keys())
            first_to_rank_str = months_to_rank[0].strftime("%Y-%m")
            preceding = [m for m in all_known_months if m < first_to_rank_str]
            if preceding:
                prev_active = set(month_active_symbols[preceding[-1]])
                log.info(
                    f"Seeding carry-over from {preceding[-1]}: "
                    f"{len(prev_active)} symbols"
                )

        for m in months_to_rank:
            month_str = m.strftime("%Y-%m")
            log.info(f"\n{'='*50}")
            log.info(f"Ranking month: {month_str}")
            log.info(f"Previous active universe: {len(prev_active)} symbols")
            log.info(f"{'='*50}")

            # Get full ranking of ALL pairs
            full_ranking = rank_all_pairs_for_month(m)

            # Build active universe with carry-over logic
            active_df, month_dropped = build_active_universe(
                month_start=m,
                full_ranking=full_ranking,
                prev_month_symbols=prev_active,
                top_n=args.top_n,
                min_daily_volume=args.min_volume,
            )

            # Store results
            symbols = active_df["symbol"].tolist()
            month_active_symbols[month_str] = symbols

            for _, row in active_df.iterrows():
                universe_records.append({
                    "month": month_str,
                    "symbol": row["symbol"],
                    "rank": int(row["rank"]),
                    "avg_daily_quote_volume_usd": round(row["avg_daily_quote_volume"], 2),
                    "avg_daily_trades": round(row["avg_daily_trades"], 1),
                    "status": row["status"],
                })

            for d in month_dropped:
                dropped_records.append({
                    "month": month_str,
                    "symbol": d["symbol"],
                    "rank": d["rank"],
                    "avg_daily_quote_volume_usd": round(d["avg_daily_quote_volume"], 2) if d["avg_daily_quote_volume"] else 0,
                    "status": d["status"],
                })

            # Update prev_active for next month
            prev_active = set(symbols)

        # Save universe and dropped files
        universe_df = pd.DataFrame(universe_records)
        universe_df.to_csv(universe_file, index=False)
        log.info(f"Universe saved to {universe_file}")

        if dropped_records:
            dropped_df = pd.DataFrame(dropped_records)
            dropped_df.to_csv(dropped_file, index=False)
            log.info(f"Dropped symbols saved to {dropped_file}")

        # Summary stats
        all_unique = set()
        for syms in month_active_symbols.values():
            all_unique.update(syms)

        sorted_months = sorted(month_active_symbols.keys())
        log.info(f"\n{'='*50}")
        log.info("Universe size per month:")
        for m_str in sorted_months:
            n = len(month_active_symbols[m_str])
            log.info(f"  {m_str}: {n} symbols")
        log.info(f"Total unique symbols across all months: {len(all_unique)}")
        log.info(f"{'='*50}")

    if args.ranking_only:
        log.info("Ranking-only mode. Done.")
        return

    # -----------------------------------------------------------------------
    # Phase 2: Download 15-min klines
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Phase 2: Downloading 15-min klines")
    log.info("=" * 60)

    total_stats = {"downloaded": 0, "skipped": 0, "failed": 0, "empty": 0}

    for m in months:
        month_str = m.strftime("%Y-%m")
        symbols = month_active_symbols.get(month_str, [])

        if not symbols:
            log.warning(f"No symbols for {month_str}, skipping")
            continue

        log.info(f"\n{'='*40}")
        log.info(f"Month: {month_str} — {len(symbols)} symbols")
        log.info(f"{'='*40}")

        stats = pull_month(m, symbols, output_dir, skip_existing=args.skip_existing)

        for k in total_stats:
            total_stats[k] += stats[k]

        log.info(
            f"  {month_str} done: {stats['downloaded']} downloaded, "
            f"{stats['skipped']} skipped, {stats['empty']} empty/delisted"
        )

    log.info(f"\n{'='*60}")
    log.info("COMPLETE")
    log.info(f"  Total downloaded: {total_stats['downloaded']}")
    log.info(f"  Total skipped:    {total_stats['skipped']}")
    log.info(f"  Total empty:      {total_stats['empty']}")
    log.info(f"  Output:           {output_dir.resolve()}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
