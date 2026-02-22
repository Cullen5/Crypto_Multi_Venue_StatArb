"""
Binance Data Collection - BTC Futures Curve Trading
=====================================================
Three collection methods:

1. Binance Data Vision (bulk CSV): BTC spot klines (1h)
2. Binance Data Vision (bulk CSV): BTC quarterly futures klines (1h)
3. Binance Futures API: BTC perpetual funding rates (paginated)

Usage:
  python collect_binance_data.py                     # Collect all
  python collect_binance_data.py --source spot        # Spot only
  python collect_binance_data.py --source futures     # Quarterly futures only
  python collect_binance_data.py --source funding     # Funding rates only
  python collect_binance_data.py --source perp        # Perpetual OHLCV only

Output: Parquet files in ./data/raw/binance/
"""

import os
import io
import time
import gzip
import zipfile
import logging
import argparse
import requests
import pandas as pd
from datetime import datetime as dt, date, timedelta
from pathlib import Path
from typing import Optional

# ─────────────────────────── CONFIG ───────────────────────────

DATA_DIR = Path("./data")
RAW_DIR = DATA_DIR / "raw" / "binance"
START_DATE = date(2022, 1, 1)
END_DATE = date(2025, 12, 31)

# Binance Data Vision base URL
VISION_BASE = "https://data.binance.vision/data"

# Binance Futures API
FAPI_BASE = "https://fapi.binance.com/fapi/v1"

# Known quarterly futures contract symbols
# COIN-margined (cm): BTCUSD_YYMMDD — this is what's on data.binance.vision
# USDT-margined (um): BTCUSDT_YYMMDD — also available but fewer contracts

# COIN-margined quarterly contracts (primary — confirmed on data.binance.vision)
COINM_QUARTERLY_SYMBOLS = [
    "BTCUSD_231229",
    "BTCUSD_240329", "BTCUSD_240628", "BTCUSD_240927", "BTCUSD_241227",
    "BTCUSD_250328", "BTCUSD_250627", "BTCUSD_250926", "BTCUSD_251226",
    "BTCUSD_260327", "BTCUSD_260626",
]

# USDT-margined quarterly contracts (try these too — may or may not exist)
USDTM_QUARTERLY_SYMBOLS = [
    # 2022
    "BTCUSDT_220325", "BTCUSDT_220624", "BTCUSDT_220930", "BTCUSDT_221230",
    # 2023
    "BTCUSDT_230331", "BTCUSDT_230630", "BTCUSDT_230929", "BTCUSDT_231229",
    # 2024
    "BTCUSDT_240329", "BTCUSDT_240628", "BTCUSDT_240927", "BTCUSDT_241227",
    # 2025
    "BTCUSDT_250328", "BTCUSDT_250627",
]

# Kline CSV columns
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ─────────────────────────── HELPERS ───────────────────────────

def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

def save_parquet(df: pd.DataFrame, path: Path, label: str):
    if df.empty:
        logger.warning(f"  {label}: empty DataFrame, skipping save")
        return
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"  {label}: saved {len(df):,} rows → {path}")

def generate_month_range(start: date, end: date) -> list[tuple[int, int]]:
    """Generate (year, month) tuples covering the date range."""
    months = []
    current = date(start.year, start.month, 1)
    while current <= end:
        months.append((current.year, current.month))
        # Advance to next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return months

def parse_kline_csv(content: bytes, symbol: str) -> pd.DataFrame:
    """Parse a Binance kline CSV (no header) into a clean DataFrame."""
    try:
        df = pd.read_csv(io.BytesIO(content), header=None, names=KLINE_COLUMNS)
    except Exception as e:
        logger.error(f"    Failed to parse CSV: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    # Convert types
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                 "taker_buy_base_volume", "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce", downcast="integer")

    # Add datetime and symbol
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["timestamp"] = df["open_time"]
    df["symbol"] = symbol

    # Drop the unused column
    df = df.drop(columns=["ignore", "close_time"], errors="ignore")

    return df


# ═══════════════════════════════════════════════════════════════
#  BINANCE DATA VISION - BULK CSV DOWNLOADS
# ═══════════════════════════════════════════════════════════════

class BinanceVisionDownloader:
    """
    Downloads bulk CSV klines from data.binance.vision.
    
    URL patterns:
      Spot:    /data/spot/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY}-{MM}.zip
      Futures: /data/futures/um/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY}-{MM}.zip
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "btc-curve-research/1.0"})

    def _download_zip(self, url: str) -> Optional[bytes]:
        """Download a zip file and extract the CSV content."""
        try:
            resp = self.session.get(url, timeout=60)
            if resp.status_code == 404:
                return None
            if resp.status_code == 451:
                # Region-blocked — try .gz alternative
                return None
            resp.raise_for_status()

            # Binance Vision serves .zip files containing a single CSV
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    return None
                return zf.read(csv_names[0])

        except zipfile.BadZipFile:
            # Sometimes the file is actually gzipped, not zipped
            try:
                return gzip.decompress(resp.content)
            except:
                logger.warning(f"    Bad archive: {url}")
                return None
        except Exception as e:
            logger.warning(f"    Download failed: {url} — {e}")
            return None

    # ── Spot Klines ──

    def download_spot_klines(self, symbol: str = "BTCUSDT", interval: str = "1h") -> pd.DataFrame:
        """Download all monthly spot kline CSVs for the date range."""
        logger.info(f"  Downloading spot klines: {symbol} {interval}")
        months = generate_month_range(START_DATE, END_DATE)
        all_dfs = []

        for year, month in months:
            url = (f"{VISION_BASE}/spot/monthly/klines/{symbol}/{interval}/"
                   f"{symbol}-{interval}-{year}-{month:02d}.zip")

            logger.info(f"    {year}-{month:02d} ... ", )
            content = self._download_zip(url)

            if content is None:
                logger.info(f"    {year}-{month:02d}: not available (may not exist yet)")
                continue

            df = parse_kline_csv(content, symbol)
            if not df.empty:
                all_dfs.append(df)
                logger.info(f"    {year}-{month:02d}: {len(df):,} candles")

            time.sleep(0.2)  # Be polite

        if all_dfs:
            result = pd.concat(all_dfs, ignore_index=True)
            result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            return result
        return pd.DataFrame()

    # ── Futures Klines (Perpetual) ──

    def download_perpetual_klines(self, symbol: str = "BTCUSDT", interval: str = "1h") -> pd.DataFrame:
        """Download USDT-M perpetual futures klines."""
        logger.info(f"  Downloading perpetual klines: {symbol} {interval}")
        months = generate_month_range(START_DATE, END_DATE)
        all_dfs = []

        for year, month in months:
            url = (f"{VISION_BASE}/futures/um/monthly/klines/{symbol}/{interval}/"
                   f"{symbol}-{interval}-{year}-{month:02d}.zip")

            content = self._download_zip(url)

            if content is None:
                logger.info(f"    {year}-{month:02d}: not available")
                continue

            df = parse_kline_csv(content, symbol)
            if not df.empty:
                df["contract_type"] = "perpetual"
                all_dfs.append(df)
                logger.info(f"    {year}-{month:02d}: {len(df):,} candles")

            time.sleep(0.2)

        if all_dfs:
            result = pd.concat(all_dfs, ignore_index=True)
            result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            return result
        return pd.DataFrame()

    # ── Futures Klines (Quarterly Contracts) ──

    def download_quarterly_klines(self, interval: str = "1h") -> pd.DataFrame:
        """
        Download all quarterly futures contract klines.
        
        Tries both COIN-margined (cm: BTCUSD_*) and USDT-margined (um: BTCUSDT_*).
        Each contract only has data for ~6 months (listing to expiry), 
        so many month files will 404 — that's normal.
        """
        logger.info(f"  Downloading quarterly futures klines ({interval})")

        # Define both sets: (symbol_list, url_path_segment, label)
        contract_sets = [
            (COINM_QUARTERLY_SYMBOLS, "futures/cm/monthly/klines", "COIN-M"),
            (USDTM_QUARTERLY_SYMBOLS, "futures/um/monthly/klines", "USDT-M"),
        ]

        all_dfs = []
        contracts_found = []

        for symbols, url_path, margin_label in contract_sets:
            logger.info(f"\n  --- {margin_label} contracts ({len(symbols)} symbols) ---")

            for symbol in symbols:
                # Parse expiry from symbol to determine sensible month range
                expiry_str = symbol.split("_")[1]  # e.g. "240329"
                try:
                    expiry_date = dt.strptime(expiry_str, "%y%m%d").date()
                except ValueError:
                    logger.warning(f"    Can't parse expiry from {symbol}, skipping")
                    continue

                # Contracts typically list ~6 months before expiry
                listing_date = expiry_date - timedelta(days=200)
                # Clamp to our collection range
                eff_start = max(listing_date, START_DATE)
                eff_end = min(expiry_date, END_DATE)

                if eff_start > eff_end:
                    continue  # Contract entirely outside our range

                months = generate_month_range(eff_start, eff_end)
                contract_dfs = []

                for year, month in months:
                    url = (f"{VISION_BASE}/{url_path}/{symbol}/{interval}/"
                           f"{symbol}-{interval}-{year}-{month:02d}.zip")

                    content = self._download_zip(url)
                    if content is None:
                        continue

                    df = parse_kline_csv(content, symbol)
                    if not df.empty:
                        df["contract_type"] = "quarterly"
                        df["margin_type"] = margin_label
                        df["expiry_date"] = expiry_date.isoformat()
                        contract_dfs.append(df)

                    time.sleep(0.2)

                if contract_dfs:
                    contract_df = pd.concat(contract_dfs, ignore_index=True)
                    all_dfs.append(contract_df)
                    contracts_found.append(f"{symbol} ({margin_label})")
                    logger.info(f"    ✓ {symbol}: {len(contract_df):,} candles "
                               f"({contract_df['datetime'].min().strftime('%Y-%m-%d')} → "
                               f"{contract_df['datetime'].max().strftime('%Y-%m-%d')})")
                else:
                    logger.info(f"    ✗ {symbol}: no data found")

        logger.info(f"\n  Found data for {len(contracts_found)} contracts total")
        for c in contracts_found:
            logger.info(f"    ✓ {c}")

        if all_dfs:
            result = pd.concat(all_dfs, ignore_index=True)
            result = result.drop_duplicates(
                subset=["timestamp", "symbol"]
            ).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
            return result
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
#  BINANCE FUTURES API - FUNDING RATES
# ═══════════════════════════════════════════════════════════════

class BinanceFundingCollector:
    """
    Collects historical funding rates via Binance Futures API.
    
    Endpoint: GET /fapi/v1/fundingRate
    - Paginates using startTime / endTime
    - Max 1000 per request
    - Funding is every 8 hours (3x daily)
    - Rate limit: 1200 weight/min (this endpoint is weight=1)
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "btc-curve-research/1.0"})

    def get_funding_rates(self, symbol: str = "BTCUSDT",
                          start: Optional[date] = None,
                          end: Optional[date] = None) -> pd.DataFrame:
        """Fetch all historical funding rates with pagination."""
        if start is None:
            start = START_DATE
        if end is None:
            end = END_DATE

        url = f"{FAPI_BASE}/fundingRate"
        all_rates = []
        current_start = int(dt.combine(start, dt.min.time()).timestamp() * 1000)
        end_ms = int(dt.combine(end, dt.max.time()).timestamp() * 1000)

        logger.info(f"  Fetching Binance funding rates: {symbol} ({start} → {end})")

        while current_start < end_ms:
            params = {
                "symbol": symbol,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000,
            }

            try:
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    logger.warning(f"    Rate limited, waiting {wait}s")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if not data:
                    break

                all_rates.extend(data)

                # Advance past last entry
                last_ts = data[-1]["fundingTime"]
                current_start = last_ts + 1

                last_dt = dt.utcfromtimestamp(last_ts / 1000)
                logger.info(f"    ... {len(all_rates):,} entries (up to {last_dt.strftime('%Y-%m-%d %H:%M')})")

                if len(data) < 1000:
                    break

                # Respect rate limits — light touch since weight=1
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                logger.error(f"    Error: {e}")
                # Skip forward by a week
                current_start += 86400000 * 7
                time.sleep(5)
                continue

        df = pd.DataFrame(all_rates)
        if not df.empty:
            df["timestamp"] = pd.to_numeric(df["fundingTime"])
            df["datetime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
            df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
            df["mark_price"] = pd.to_numeric(df.get("markPrice", pd.Series(dtype=float)), errors="coerce")
            df["symbol"] = symbol
            df["venue"] = "binance"
            df["venue_type"] = "CEX"

            # Clean up column names
            df = df.rename(columns={"fundingTime": "funding_time_ms"})
            df = df.drop(columns=["fundingRate", "markPrice"], errors="ignore")
            df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        logger.info(f"    Total: {len(df):,} Binance funding rate entries")
        return df


# ═══════════════════════════════════════════════════════════════
#  VALIDATION & SUMMARY
# ═══════════════════════════════════════════════════════════════

def validate_and_summarize():
    """Print summary of all collected Binance data."""
    logger.info("\n" + "=" * 60)
    logger.info("BINANCE DATA VALIDATION SUMMARY")
    logger.info("=" * 60)

    parquet_files = list(RAW_DIR.rglob("*.parquet"))
    if not parquet_files:
        logger.warning("No parquet files found in Binance directory!")
        return

    for f in sorted(parquet_files):
        try:
            df = pd.read_parquet(f)
            rel_path = f.relative_to(RAW_DIR)

            logger.info(f"\n  {rel_path}")
            logger.info(f"    Rows: {len(df):,}")

            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                logger.info(f"    Date range: {df['datetime'].min()} → {df['datetime'].max()}")

            if "symbol" in df.columns:
                symbols = df["symbol"].unique()
                logger.info(f"    Symbols: {list(symbols)}")

                if len(symbols) > 1:
                    for s in symbols:
                        sub = df[df["symbol"] == s]
                        logger.info(f"      {s}: {len(sub):,} rows "
                                   f"({sub['datetime'].min().strftime('%Y-%m-%d')} → "
                                   f"{sub['datetime'].max().strftime('%Y-%m-%d')})")

            if "close" in df.columns:
                logger.info(f"    Price range: ${df['close'].min():,.2f} → ${df['close'].max():,.2f}")

            if "funding_rate" in df.columns:
                fr = df["funding_rate"]
                logger.info(f"    Funding rate: mean={fr.mean():.6f}, min={fr.min():.6f}, max={fr.max():.6f}")

            if "timestamp" in df.columns and len(df) > 1:
                diffs = df.groupby("symbol" if "symbol" in df.columns else pd.Series(0, index=df.index))["timestamp"].apply(
                    lambda x: x.sort_values().diff().dropna()
                )
                if len(diffs) > 0:
                    max_gap_h = diffs.max() / 3_600_000
                    median_gap_h = diffs.median() / 3_600_000
                    logger.info(f"    Gaps: median={median_gap_h:.1f}h, max={max_gap_h:.1f}h")
                    if max_gap_h > 48:
                        logger.warning(f"    ⚠ Large gap: {max_gap_h:.0f}h")

        except Exception as e:
            logger.error(f"  Error reading {f}: {e}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Binance BTC Data Collector")
    parser.add_argument("--source", choices=["spot", "perp", "futures", "funding", "all"],
                        default="all", help="Which data to collect")
    parser.add_argument("--start", type=str, default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    global START_DATE, END_DATE
    if args.start:
        START_DATE = date.fromisoformat(args.start)
    if args.end:
        END_DATE = date.fromisoformat(args.end)

    logger.info(f"Collection range: {START_DATE} → {END_DATE}")
    ensure_dirs()

    vision = BinanceVisionDownloader()
    funding = BinanceFundingCollector()

    # ── Spot OHLCV ──
    if args.source in ("spot", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("[1] BTC SPOT KLINES (Binance Data Vision)")
        logger.info("=" * 60)
        df_spot = vision.download_spot_klines("BTCUSDT", "1h")
        df_spot["contract_type"] = "spot"
        df_spot["venue"] = "binance"
        save_parquet(df_spot, RAW_DIR / "btc_spot_ohlcv_1h.parquet", "BTC Spot")

    # ── Perpetual Futures OHLCV ──
    if args.source in ("perp", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("[2] BTC PERPETUAL KLINES (Binance Data Vision)")
        logger.info("=" * 60)
        df_perp = vision.download_perpetual_klines("BTCUSDT", "1h")
        df_perp["venue"] = "binance"
        save_parquet(df_perp, RAW_DIR / "btc_perpetual_ohlcv_1h.parquet", "BTC Perpetual")

    # ── Quarterly Futures OHLCV ──
    if args.source in ("futures", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("[3] BTC QUARTERLY FUTURES KLINES (Binance Data Vision)")
        logger.info("=" * 60)
        df_futures = vision.download_quarterly_klines("1h")
        df_futures["venue"] = "binance"
        save_parquet(df_futures, RAW_DIR / "btc_quarterly_futures_ohlcv_1h.parquet", "BTC Quarterly Futures")

        # Save contract summary
        if not df_futures.empty:
            contract_summary = df_futures.groupby("symbol").agg(
                first_date=("datetime", "min"),
                last_date=("datetime", "max"),
                n_candles=("datetime", "count"),
                expiry_date=("expiry_date", "first"),
            ).reset_index()
            save_parquet(contract_summary, RAW_DIR / "btc_quarterly_contracts_summary.parquet",
                        "Contract Summary")

    # ── Funding Rates ──
    if args.source in ("funding", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("[4] BTC PERPETUAL FUNDING RATES (Binance API)")
        logger.info("=" * 60)
        df_funding = funding.get_funding_rates("BTCUSDT")
        save_parquet(df_funding, RAW_DIR / "btc_perpetual_funding_rates.parquet", "BTC Funding Rates")

    # ── Validate ──
    if not args.skip_validation:
        validate_and_summarize()

    logger.info(f"\nDONE. Data saved to: {RAW_DIR.resolve()}")


if __name__ == "__main__":
    main()
