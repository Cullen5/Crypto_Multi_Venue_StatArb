"""
BTC Futures Curve Trading - Data Collection Script
====================================================
Collects data for Strategy A (Calendar Spreads) and Strategy C (Synthetic Futures from Perp Funding):

Sources:
  - Deribit: BTC spot/index OHLCV, quarterly futures OHLCV, perpetual OHLCV, funding rates
  - Hyperliquid: BTC perpetual funding rates + perpetual OHLCV (1h candles)
  - dYdX v4: BTC perpetual funding rates + perpetual OHLCV (1h candles)

Output: Parquet files in ./data/ directory

Usage:
  python collect_btc_data.py              # Collect all data
  python collect_btc_data.py --source deribit   # Collect Deribit only
  python collect_btc_data.py --source hyperliquid
  python collect_btc_data.py --source dydx
"""

import os
import time
import json
import logging
import argparse
import requests
import pandas as pd
from datetime import datetime as dt, date, timedelta
from pathlib import Path
from typing import Optional

# ─────────────────────────── CONFIG ───────────────────────────

DATA_DIR = Path("./data")
RAW_DIR = DATA_DIR / "raw"
START_DATE = date(2022, 1, 1)
END_DATE = date(2025, 12, 31)  # Adjust to today or your cutoff

# Rate limit defaults (requests per second)
DERIBIT_RPS = 10       # Conservative; Deribit allows ~20/sec public
HYPERLIQUID_RPS = 5    # No strict limits, but be polite
DYDX_RPS = 5           # Moderate limits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─────────────────────────── HELPERS ───────────────────────────

def ms(datetime_obj: dt) -> int:
    """Datetime → Unix timestamp in milliseconds."""
    return int(datetime_obj.timestamp() * 1000)

def from_ms(timestamp: int) -> dt:
    """Unix timestamp in milliseconds → datetime."""
    return dt.utcfromtimestamp(timestamp / 1000)

def ensure_dirs():
    """Create output directories."""
    for sub in ["deribit", "hyperliquid", "dydx"]:
        (RAW_DIR / sub).mkdir(parents=True, exist_ok=True)

def save_parquet(df: pd.DataFrame, path: Path, label: str):
    """Save DataFrame to parquet with logging."""
    if df.empty:
        logger.warning(f"  {label}: empty DataFrame, skipping save")
        return
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"  {label}: saved {len(df):,} rows → {path}")


class RateLimitedSession:
    """Requests session with automatic rate limiting and retries."""

    def __init__(self, rps: float = 10, max_retries: int = 5, backoff_factor: float = 1.0):
        self.min_interval = 1.0 / rps
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "btc-curve-research/1.0"})

    def _wait(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def get(self, url: str, params: dict = None, **kwargs) -> requests.Response:
        for attempt in range(self.max_retries):
            self._wait()
            try:
                resp = self.session.get(url, params=params, timeout=30, **kwargs)
                self.last_request = time.time()

                if resp.status_code == 429:
                    wait = self.backoff_factor * (2 ** attempt)
                    logger.warning(f"  Rate limited, waiting {wait:.1f}s (attempt {attempt+1})")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp

            except requests.exceptions.RequestException as e:
                wait = self.backoff_factor * (2 ** attempt)
                logger.warning(f"  Request error: {e}, retrying in {wait:.1f}s (attempt {attempt+1})")
                time.sleep(wait)

        raise Exception(f"Failed after {self.max_retries} retries: {url}")

    def post(self, url: str, json_data: dict = None, **kwargs) -> requests.Response:
        for attempt in range(self.max_retries):
            self._wait()
            try:
                resp = self.session.post(url, json=json_data, timeout=30, **kwargs)
                self.last_request = time.time()

                if resp.status_code == 429:
                    wait = self.backoff_factor * (2 ** attempt)
                    logger.warning(f"  Rate limited, waiting {wait:.1f}s (attempt {attempt+1})")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp

            except requests.exceptions.RequestException as e:
                wait = self.backoff_factor * (2 ** attempt)
                logger.warning(f"  Request error: {e}, retrying in {wait:.1f}s (attempt {attempt+1})")
                time.sleep(wait)

        raise Exception(f"Failed after {self.max_retries} retries: {url}")


# ═══════════════════════════════════════════════════════════════
#  DERIBIT — KNOWN QUARTERLY CONTRACT NAMES
# ═══════════════════════════════════════════════════════════════
#
# Deribit BTC quarterly futures expire on the last Friday of
# March, June, September, December at 08:00 UTC.
#
# The API endpoint get_instruments(expired=True) does NOT reliably
# return contracts that expired long ago — Deribit prunes them.
#
# Fix: hardcode the known contract names (predictable naming)
# and use history.deribit.com for OHLCV of expired contracts.
#

# Format: BTC-{DDMMMYY} where DD is the last Friday of the quarter month
KNOWN_QUARTERLY_CONTRACTS = [
    # 2022
    {"name": "BTC-25MAR22", "expiry": "2022-03-25", "listed_approx": "2021-09-25"},
    {"name": "BTC-24JUN22", "expiry": "2022-06-24", "listed_approx": "2021-12-25"},
    {"name": "BTC-30SEP22", "expiry": "2022-09-30", "listed_approx": "2022-03-26"},
    {"name": "BTC-30DEC22", "expiry": "2022-12-30", "listed_approx": "2022-06-25"},
    # 2023
    {"name": "BTC-31MAR23", "expiry": "2023-03-31", "listed_approx": "2022-09-30"},
    {"name": "BTC-30JUN23", "expiry": "2023-06-30", "listed_approx": "2022-12-31"},
    {"name": "BTC-29SEP23", "expiry": "2023-09-29", "listed_approx": "2023-03-31"},
    {"name": "BTC-29DEC23", "expiry": "2023-12-29", "listed_approx": "2023-06-30"},
    # 2024
    {"name": "BTC-29MAR24", "expiry": "2024-03-29", "listed_approx": "2023-09-30"},
    {"name": "BTC-28JUN24", "expiry": "2024-06-28", "listed_approx": "2023-12-30"},
    {"name": "BTC-27SEP24", "expiry": "2024-09-27", "listed_approx": "2024-03-30"},
    {"name": "BTC-27DEC24", "expiry": "2024-12-27", "listed_approx": "2024-06-29"},
    # 2025
    {"name": "BTC-28MAR25", "expiry": "2025-03-28", "listed_approx": "2024-09-28"},
    {"name": "BTC-27JUN25", "expiry": "2025-06-27", "listed_approx": "2024-12-28"},
]


# ═══════════════════════════════════════════════════════════════
#  DERIBIT COLLECTOR
# ═══════════════════════════════════════════════════════════════

class DeribitCollector:
    """
    Collects from Deribit public API (no auth needed):
      - BTC-PERPETUAL OHLCV + funding rates
      - BTC quarterly futures OHLCV (all historical contracts via history endpoint)
    """

    BASE_URL = "https://www.deribit.com/api/v2/public"
    HISTORY_URL = "https://history.deribit.com/api/v2/public"

    def __init__(self):
        self.session = RateLimitedSession(rps=DERIBIT_RPS, backoff_factor=1.5)
        self.out_dir = RAW_DIR / "deribit"

    def get_active_futures(self) -> list[dict]:
        """Fetch currently active BTC futures from the live API."""
        url = f"{self.BASE_URL}/get_instruments"
        params = {"currency": "BTC", "kind": "future", "expired": "false"}
        try:
            resp = self.session.get(url, params=params)
            instruments = resp.json().get("result", [])
            active = []
            for inst in instruments:
                name = inst["instrument_name"]
                if "PERPETUAL" in name:
                    continue
                active.append({
                    "name": name,
                    "expiry": from_ms(inst["expiration_timestamp"]).strftime("%Y-%m-%d"),
                    "listed_approx": from_ms(inst.get("creation_timestamp", 0)).strftime("%Y-%m-%d"),
                })
            logger.info(f"  Found {len(active)} active futures from API: "
                        f"{[c['name'] for c in active]}")
            return active
        except Exception as e:
            logger.warning(f"  Could not fetch active futures: {e}")
            return []

    def get_all_contracts(self) -> list[dict]:
        """
        Build complete contract list:
        1. Start with hardcoded known quarterlies (reliable for expired)
        2. Merge in any active contracts from the API (catches new listings)
        3. Filter to our date range
        """
        # Start with known contracts
        contracts = list(KNOWN_QUARTERLY_CONTRACTS)
        known_names = {c["name"] for c in contracts}

        # Add any active contracts not already in the list
        active = self.get_active_futures()
        for ac in active:
            if ac["name"] not in known_names:
                contracts.append(ac)
                logger.info(f"    Added active contract not in hardcoded list: {ac['name']}")

        # Filter to our date range
        range_start = dt.combine(START_DATE, dt.min.time())
        range_end = dt.combine(END_DATE, dt.max.time())

        filtered = []
        for c in contracts:
            expiry_dt = dt.strptime(c["expiry"], "%Y-%m-%d")
            listed_dt = dt.strptime(c["listed_approx"], "%Y-%m-%d")

            # Include if contract overlaps with our collection range
            if expiry_dt >= range_start and listed_dt <= range_end:
                filtered.append(c)

        filtered.sort(key=lambda x: x["expiry"])
        logger.info(f"  Total contracts to collect: {len(filtered)}")
        for c in filtered:
            logger.info(f"    {c['name']}  (expires {c['expiry']})")
        return filtered

    # ── OHLCV — tries live API first, falls back to history endpoint ──

    def get_ohlcv(self, instrument_name: str, resolution: int = 60,
                  start: Optional[dt] = None, end: Optional[dt] = None) -> pd.DataFrame:
        """
        Fetch OHLCV candles for a tradeable instrument.

        Tries the live API first (works for active contracts).
        Falls back to history.deribit.com (works for expired contracts).

        Args:
            instrument_name: e.g. 'BTC-PERPETUAL', 'BTC-28JUN24'
            resolution: candle size in minutes (60 = 1h)
            start/end: datetime range
        """
        if start is None:
            start = dt.combine(START_DATE, dt.min.time())
        if end is None:
            end = dt.combine(END_DATE, dt.max.time())

        # Try live API first, then history endpoint
        for base_url, label in [
            (self.BASE_URL, "live"),
            (self.HISTORY_URL, "history"),
        ]:
            df = self._fetch_ohlcv_from(base_url, instrument_name, resolution, start, end, label)
            if not df.empty:
                return df
            logger.info(f"    {label} API returned no data, trying next...")

        logger.warning(f"    No OHLCV data from any endpoint for {instrument_name}")
        return pd.DataFrame()

    def _fetch_ohlcv_from(self, base_url: str, instrument_name: str,
                          resolution: int, start: dt, end: dt, label: str) -> pd.DataFrame:
        """Fetch OHLCV from a specific Deribit API base URL."""
        url = f"{base_url}/get_tradingview_chart_data"
        all_candles = []
        current_start = ms(start)
        end_ms = ms(end)

        logger.info(f"  Fetching OHLCV ({label}): {instrument_name} "
                     f"({start.date()} → {end.date()}, {resolution}m)")

        while current_start < end_ms:
            params = {
                "instrument_name": instrument_name,
                "start_timestamp": current_start,
                "end_timestamp": end_ms,
                "resolution": resolution,
            }

            try:
                resp = self.session.get(url, params=params)
                data = resp.json()

                if "result" not in data:
                    error = data.get("error", {})
                    msg = error.get("message", "unknown")
                    # "instrument_not_found" on live API is expected for expired
                    if "not_found" in msg.lower() or "expired" in msg.lower():
                        logger.info(f"    {label}: instrument not found (expected for expired)")
                        return pd.DataFrame()
                    logger.warning(f"    {label} API error: {msg}")
                    return pd.DataFrame()

                result = data["result"]
                if result.get("status") == "no_data":
                    break

                ticks = result.get("ticks", [])
                if not ticks:
                    break

                n_candles = len(ticks)
                for i in range(n_candles):
                    all_candles.append({
                        "timestamp": ticks[i],
                        "open": result["open"][i],
                        "high": result["high"][i],
                        "low": result["low"][i],
                        "close": result["close"][i],
                        "volume": result["volume"][i],
                    })

                current_start = ticks[-1] + (resolution * 60 * 1000)
                logger.info(f"    ... {len(all_candles):,} candles "
                            f"(up to {from_ms(ticks[-1]).strftime('%Y-%m-%d %H:%M')})")

                if n_candles < 100:
                    break

            except Exception as e:
                logger.error(f"    Error fetching {instrument_name} ({label}): {e}")
                current_start += 86400000
                continue

        df = pd.DataFrame(all_candles)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["instrument"] = instrument_name
            df = (df.drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                    .reset_index(drop=True))
        logger.info(f"    {label}: {len(df):,} candles for {instrument_name}")
        return df

    # ── Funding Rates ──

    def get_funding_rates(self, instrument_name: str = "BTC-PERPETUAL",
                          start: Optional[dt] = None, end: Optional[dt] = None) -> pd.DataFrame:
        """Fetch historical funding rates for a perpetual contract."""
        if start is None:
            start = dt.combine(START_DATE, dt.min.time())
        if end is None:
            end = dt.combine(END_DATE, dt.max.time())

        # Try both endpoints for funding too
        for base_url, label in [
            (self.BASE_URL, "live"),
            (self.HISTORY_URL, "history"),
        ]:
            df = self._fetch_funding_from(base_url, instrument_name, start, end, label)
            if not df.empty:
                return df

        logger.warning(f"    No funding data from any endpoint for {instrument_name}")
        return pd.DataFrame()

    def _fetch_funding_from(self, base_url: str, instrument_name: str,
                            start: dt, end: dt, label: str) -> pd.DataFrame:
        """Fetch funding rates from a specific Deribit API base URL."""
        url = f"{base_url}/get_funding_rate_history"
        all_rates = []
        end_ms = ms(end)
        chunk_days = 30
        current_start = ms(start)

        logger.info(f"  Fetching funding rates ({label}): {instrument_name} "
                     f"({start.date()} → {end.date()})")

        while current_start < end_ms:
            chunk_end = min(current_start + (chunk_days * 24 * 3600 * 1000), end_ms)

            params = {
                "instrument_name": instrument_name,
                "start_timestamp": current_start,
                "end_timestamp": chunk_end,
            }

            try:
                resp = self.session.get(url, params=params)
                data = resp.json()

                if "result" not in data or not data["result"]:
                    current_start = chunk_end + 1
                    continue

                rates = data["result"]
                all_rates.extend(rates)

                logger.info(f"    ... {len(all_rates):,} funding entries "
                            f"(chunk to {from_ms(chunk_end).strftime('%Y-%m-%d')})")

                current_start = chunk_end + 1

            except Exception as e:
                logger.error(f"    Error fetching funding rates ({label}): {e}")
                current_start = chunk_end + 1
                continue

        df = pd.DataFrame(all_rates)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["venue"] = "deribit"
            if "interest_8h" in df.columns:
                df["funding_rate"] = pd.to_numeric(df["interest_8h"], errors="coerce")
            if "interest_1h" in df.columns:
                df["funding_rate_1h"] = pd.to_numeric(df["interest_1h"], errors="coerce")
            df = (df.drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                    .reset_index(drop=True))
        logger.info(f"    {label}: {len(df):,} funding entries")
        return df

    # ── Orchestration ──

    def collect_all(self):
        """Run full Deribit collection."""
        logger.info("=" * 60)
        logger.info("DERIBIT COLLECTION START")
        logger.info("=" * 60)

        # 1. Spot — skip, use Binance
        logger.info("\n[1/4] BTC Index (Spot) Price")
        logger.info("  SKIPPED — use Binance spot OHLCV for spot/index price")

        # 2. BTC Perpetual OHLCV
        logger.info("\n[2/4] BTC-PERPETUAL OHLCV")
        df_perp = self.get_ohlcv("BTC-PERPETUAL", resolution=60)
        save_parquet(df_perp, self.out_dir / "btc_perpetual_ohlcv_1h.parquet", "BTC Perpetual")

        # 3. BTC Perpetual Funding Rates
        logger.info("\n[3/4] BTC-PERPETUAL Funding Rates")
        df_funding = self.get_funding_rates("BTC-PERPETUAL")
        save_parquet(df_funding, self.out_dir / "btc_perpetual_funding.parquet", "BTC Funding")

        # 4. Quarterly Futures OHLCV — FIXED: hardcoded names + history endpoint
        logger.info("\n[4/4] BTC Quarterly Futures OHLCV")
        contracts = self.get_all_contracts()
        all_futures = []

        for contract in contracts:
            name = contract["name"]
            expiry = contract["expiry"]

            # Collection window: max(listed_approx, START_DATE) → min(expiry, END_DATE)
            c_start = max(
                dt.strptime(contract["listed_approx"], "%Y-%m-%d"),
                dt.combine(START_DATE, dt.min.time()),
            )
            c_end = min(
                dt.strptime(expiry, "%Y-%m-%d") + timedelta(hours=9),  # expires 08:00 UTC
                dt.combine(END_DATE, dt.max.time()),
            )

            if c_start >= c_end:
                logger.info(f"  Skipping {name} — outside collection range")
                continue

            df_fut = self.get_ohlcv(name, resolution=60, start=c_start, end=c_end)
            if not df_fut.empty:
                df_fut["expiry_date"] = expiry
                df_fut["settlement_period"] = "quarter"
                all_futures.append(df_fut)
            else:
                logger.warning(f"No data for {name} — contract may predate "
                               f"history endpoint coverage")

            time.sleep(1)

        if all_futures:
            df_all_futures = pd.concat(all_futures, ignore_index=True)
            # normalize to match Binance schema
            df_all_futures["symbol"] = df_all_futures["instrument"]
            df_all_futures["margin_type"] = "COIN-M"
            df_all_futures["venue"] = "deribit"
            save_parquet(df_all_futures,
                         self.out_dir / "btc_quarterly_futures_ohlcv_1h.parquet",
                         "BTC Quarterly Futures (all contracts)")

            # Save metadata
            contracts_df = pd.DataFrame(contracts)
            save_parquet(contracts_df,
                         self.out_dir / "btc_futures_contracts_metadata.parquet",
                         "Futures contract metadata")

            # Summary stats
            logger.info(f"\n  Quarterly futures summary:")
            logger.info(f"    Contracts with data: {len(all_futures)}/{len(contracts)}")
            logger.info(f"    Total candles: {len(df_all_futures):,}")
            logger.info(f"    Date range: {df_all_futures['datetime'].min()} → "
                         f"{df_all_futures['datetime'].max()}")
            for df_c in all_futures:
                inst = df_c["instrument"].iloc[0]
                logger.info(f"      {inst}: {len(df_c):,} candles "
                            f"({df_c['datetime'].min().date()} → {df_c['datetime'].max().date()})")
        else:
            logger.error("No quarterly futures data collected at all!")

        logger.info("\nDERIBIT COLLECTION COMPLETE")


# ═══════════════════════════════════════════════════════════════
#  HYPERLIQUID COLLECTOR
# ═══════════════════════════════════════════════════════════════

class HyperliquidCollector:
    """
    Collects BTC perpetual funding rate history from Hyperliquid.
    Funding is paid hourly (not 8h like Binance).
    """

    BASE_URL = "https://api.hyperliquid.xyz/info"

    def __init__(self):
        self.session = RateLimitedSession(rps=HYPERLIQUID_RPS, backoff_factor=2.0)
        self.out_dir = RAW_DIR / "hyperliquid"

    def get_funding_rates(self, coin: str = "BTC",
                          start: Optional[dt] = None, end: Optional[dt] = None) -> pd.DataFrame:
        if start is None:
            start = dt.combine(START_DATE, dt.min.time())
        if end is None:
            end = dt.combine(END_DATE, dt.max.time())

        all_rates = []
        current_start = ms(start)
        end_ms = ms(end)

        logger.info(f"  Fetching Hyperliquid funding: {coin} ({start.date()} → {end.date()})")

        while current_start < end_ms:
            payload = {
                "type": "fundingHistory",
                "coin": coin,
                "startTime": current_start,
            }

            try:
                resp = self.session.post(self.BASE_URL, json_data=payload)
                data = resp.json()

                if not data:
                    break

                all_rates.extend(data)

                last_time = data[-1].get("time")
                if last_time is not None:
                    last_time_ms = int(float(str(last_time)))
                    current_start = last_time_ms + 1
                    last_dt_str = from_ms(last_time_ms).strftime('%Y-%m-%d %H:%M')
                else:
                    break

                logger.info(f"    ... {len(all_rates):,} entries (up to {last_dt_str})")

                if len(data) < 500:
                    break

            except Exception as e:
                logger.error(f"    Error: {e}")
                current_start += 86400000 * 7
                continue

        df = pd.DataFrame(all_rates)
        if not df.empty:
            if "time" in df.columns:
                df["time_ms"] = pd.to_numeric(df["time"], errors="coerce")
                df["datetime"] = pd.to_datetime(df["time_ms"], unit="ms", utc=True)
                df["timestamp"] = df["time_ms"].astype("int64")
            if "fundingRate" in df.columns:
                df = df.rename(columns={"fundingRate": "funding_rate"})
            df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
            df["coin"] = coin
            df["venue"] = "hyperliquid"
            df["venue_type"] = "hybrid"
            df = (df.drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                    .reset_index(drop=True))
            df = df[
                (df["datetime"] >= pd.Timestamp(start, tz="UTC")) &
                (df["datetime"] <= pd.Timestamp(end, tz="UTC"))
            ]

        logger.info(f"    Total: {len(df):,} Hyperliquid funding entries")
        return df

    def get_ohlcv(self, coin: str = "BTC", interval: str = "1h",
                  start: Optional[dt] = None, end: Optional[dt] = None) -> pd.DataFrame:
        """
        Fetch BTC perpetual OHLCV candles from Hyperliquid.

        Uses the candleSnapshot endpoint (same /info POST endpoint).
        Max 5000 candles per request, so we paginate forward in time.

        Returns DataFrame with: datetime, open, high, low, close, volume,
                                 num_trades, venue, contract_type
        """
        if start is None:
            start = dt.combine(START_DATE, dt.min.time())
        if end is None:
            end = dt.combine(END_DATE, dt.max.time())

        all_candles = []
        current_start = ms(start)
        end_ms = ms(end)

        # For 1h candles: 5000 candles ≈ 208 days per request
        logger.info(f"  Fetching Hyperliquid OHLCV: {coin} {interval} "
                    f"({start.date()} → {end.date()})")

        while current_start < end_ms:
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": end_ms,
                }
            }

            try:
                resp = self.session.post(self.BASE_URL, json_data=payload)
                data = resp.json()

                if not data:
                    break

                all_candles.extend(data)

                # Advance past the last candle's close time
                last_close = data[-1].get("T")
                if last_close is not None:
                    current_start = int(last_close) + 1
                    last_dt_str = from_ms(int(last_close)).strftime('%Y-%m-%d %H:%M')
                else:
                    break

                logger.info(f"    ... {len(all_candles):,} candles (up to {last_dt_str})")

                # If we got fewer than 5000, we've reached the end
                if len(data) < 5000:
                    break

            except Exception as e:
                logger.error(f"    Error: {e}")
                # Skip forward 7 days on error
                current_start += 86400000 * 7
                continue

        df = pd.DataFrame(all_candles)
        if not df.empty:
            # Hyperliquid fields: t=open_ms, T=close_ms, o, h, l, c, v, n, s, i
            df["datetime"] = pd.to_datetime(df["t"].astype(int), unit="ms", utc=True)
            df["timestamp"] = df["t"].astype("int64")
            for col in ["o", "h", "l", "c", "v"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.rename(columns={
                "o": "open", "h": "high", "l": "low", "c": "close",
                "v": "volume", "n": "num_trades", "s": "coin",
            })
            df["venue"] = "hyperliquid"
            df["venue_type"] = "hybrid"
            df["contract_type"] = "perpetual"
            df = (df.drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                    .reset_index(drop=True))
            df = df[
                (df["datetime"] >= pd.Timestamp(start, tz="UTC")) &
                (df["datetime"] <= pd.Timestamp(end, tz="UTC"))
            ]
            # Keep clean columns
            keep_cols = ["datetime", "timestamp", "open", "high", "low", "close",
                         "volume", "num_trades", "coin", "venue", "venue_type",
                         "contract_type"]
            df = df[[c for c in keep_cols if c in df.columns]]

        logger.info(f"    Total: {len(df):,} Hyperliquid OHLCV candles")
        return df

    def collect_all(self):
        logger.info("=" * 60)
        logger.info("HYPERLIQUID COLLECTION START")
        logger.info("=" * 60)

        df = self.get_funding_rates("BTC")
        save_parquet(df, self.out_dir / "btc_funding_rates.parquet", "Hyperliquid BTC Funding")

        df_ohlcv = self.get_ohlcv("BTC", interval="1h")
        save_parquet(df_ohlcv, self.out_dir / "btc_perpetual_ohlcv_1h.parquet",
                     "Hyperliquid BTC Perp OHLCV")

        logger.info("\nHYPERLIQUID COLLECTION COMPLETE")


# ═══════════════════════════════════════════════════════════════
#  DYDX V4 COLLECTOR
# ═══════════════════════════════════════════════════════════════

class DydxCollector:
    """
    Collects BTC perpetual funding rate history from dYdX v4 Indexer.
    dYdX v4 launched ~Nov 2023. Funding is paid hourly.
    """

    BASE_URL = "https://indexer.dydx.trade/v4"

    def __init__(self):
        self.session = RateLimitedSession(rps=DYDX_RPS, backoff_factor=2.0)
        self.out_dir = RAW_DIR / "dydx"

    def get_funding_rates(self, ticker: str = "BTC-USD",
                          start: Optional[dt] = None, end: Optional[dt] = None) -> pd.DataFrame:
        if start is None:
            start = dt.combine(START_DATE, dt.min.time())
        if end is None:
            end = dt.combine(END_DATE, dt.max.time())

        url = f"{self.BASE_URL}/historicalFunding/{ticker}"
        all_rates = []
        current_before = end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        start_iso = start.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        logger.info(f"Fetching dYdX funding: {ticker} ({start.date()} → {end.date()})")
        logger.info(f"Note: dYdX v4 launched ~Nov 2023, earlier data may not exist")

        while True:
            params = {
                "effectiveBeforeOrAt": current_before,
                "limit": 100,
            }

            try:
                resp = self.session.get(url, params=params)
                data = resp.json()

                rates = data.get("historicalFunding", [])
                if not rates:
                    break

                all_rates.extend(rates)

                oldest = rates[-1].get("effectiveAt", "")
                if not oldest:
                    break

                oldest_dt = pd.to_datetime(oldest)
                if oldest_dt < pd.to_datetime(start_iso):
                    logger.info(f"    Reached start date, stopping")
                    break

                current_before = (oldest_dt - timedelta(seconds=1)).strftime(
                    "%Y-%m-%dT%H:%M:%S.000Z")

                logger.info(f"    ... {len(all_rates):,} entries (back to {oldest[:19]})")

                if len(rates) < 100:
                    break

            except Exception as e:
                logger.error(f"    Error: {e}")
                try:
                    current_dt = pd.to_datetime(current_before)
                    current_before = (current_dt - timedelta(days=7)).strftime(
                        "%Y-%m-%dT%H:%M:%S.000Z")
                except:
                    break
                continue

        df = pd.DataFrame(all_rates)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["effectiveAt"], utc=True)
            df["timestamp"] = df["datetime"].astype("int64") // 10**6
            if "rate" in df.columns:
                df = df.rename(columns={"rate": "funding_rate"})
            df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
            df["ticker"] = ticker
            df["venue"] = "dydx_v4"
            df["venue_type"] = "hybrid"
            df = (df.drop_duplicates(subset=["datetime"])
                    .sort_values("datetime")
                    .reset_index(drop=True))
            df = df[df["datetime"] >= pd.Timestamp(start, tz="UTC")]

        logger.info(f"    Total: {len(df):,} dYdX funding entries")
        return df

    def get_ohlcv(self, ticker: str = "BTC-USD", resolution: str = "1HOUR",
                  start: Optional[dt] = None, end: Optional[dt] = None) -> pd.DataFrame:
        """
        Fetch BTC perpetual OHLCV candles from dYdX v4 Indexer.

        Endpoint: GET /v4/candles/perpetualMarkets/{ticker}
        Params: resolution, fromISO, toISO, limit (max 100 per page)
        Paginates backwards from `end` using toISO.

        Note: dYdX v4 launched ~Nov 2023, no data before that.

        resolution options: 1MIN, 5MINS, 15MINS, 30MINS, 1HOUR, 4HOURS, 1DAY
        """
        if start is None:
            start = dt.combine(START_DATE, dt.min.time())
        if end is None:
            end = dt.combine(END_DATE, dt.max.time())

        url = f"{self.BASE_URL}/candles/perpetualMarkets/{ticker}"
        all_candles = []
        current_to = end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        start_iso = start.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        logger.info(f"  Fetching dYdX OHLCV: {ticker} {resolution} "
                    f"({start.date()} → {end.date()})")
        logger.info(f"    Note: dYdX v4 launched ~Nov 2023, earlier data may not exist")

        while True:
            params = {
                "resolution": resolution,
                "toISO": current_to,
                "limit": 100,
            }

            try:
                resp = self.session.get(url, params=params)
                data = resp.json()

                candles = data.get("candles", [])
                if not candles:
                    break

                all_candles.extend(candles)

                # Candles come newest-first; last element is oldest
                oldest = candles[-1].get("startedAt", "")
                if not oldest:
                    break

                oldest_dt = pd.to_datetime(oldest)
                if oldest_dt < pd.to_datetime(start_iso):
                    logger.info(f"    Reached start date, stopping")
                    break

                # Page backwards: next request ends just before the oldest candle
                current_to = (oldest_dt - timedelta(seconds=1)).strftime(
                    "%Y-%m-%dT%H:%M:%S.000Z")

                logger.info(f"    ... {len(all_candles):,} candles "
                            f"(back to {oldest[:19]})")

                if len(candles) < 100:
                    break

            except Exception as e:
                logger.error(f"    Error: {e}")
                try:
                    current_dt = pd.to_datetime(current_to)
                    current_to = (current_dt - timedelta(days=7)).strftime(
                        "%Y-%m-%dT%H:%M:%S.000Z")
                except:
                    break
                continue

        df = pd.DataFrame(all_candles)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["startedAt"], utc=True)
            df["timestamp"] = df["datetime"].astype("int64") // 10**6
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(df.get("baseTokenVolume",
                                                  pd.Series(dtype=float)),
                                          errors="coerce")
            df["usd_volume"] = pd.to_numeric(df.get("usdVolume",
                                                      pd.Series(dtype=float)),
                                              errors="coerce")
            df["num_trades"] = pd.to_numeric(df.get("trades",
                                                      pd.Series(dtype=float)),
                                              errors="coerce")
            df["open_interest"] = pd.to_numeric(
                df.get("startingOpenInterest", pd.Series(dtype=float)),
                errors="coerce")
            df["ticker"] = ticker
            df["venue"] = "dydx_v4"
            df["venue_type"] = "hybrid"
            df["contract_type"] = "perpetual"
            df = (df.drop_duplicates(subset=["datetime"])
                    .sort_values("datetime")
                    .reset_index(drop=True))
            df = df[df["datetime"] >= pd.Timestamp(start, tz="UTC")]
            # Keep clean columns
            keep_cols = ["datetime", "timestamp", "open", "high", "low", "close",
                         "volume", "usd_volume", "num_trades", "open_interest",
                         "ticker", "venue", "venue_type", "contract_type"]
            df = df[[c for c in keep_cols if c in df.columns]]

        logger.info(f"    Total: {len(df):,} dYdX OHLCV candles")
        return df

    def collect_all(self):
        logger.info("=" * 60)
        logger.info("DYDX V4 COLLECTION START")
        logger.info("=" * 60)

        df = self.get_funding_rates("BTC-USD")
        save_parquet(df, self.out_dir / "btc_funding_rates.parquet", "dYdX BTC Funding")

        df_ohlcv = self.get_ohlcv("BTC-USD", resolution="1HOUR")
        save_parquet(df_ohlcv, self.out_dir / "btc_perpetual_ohlcv_1h.parquet",
                     "dYdX BTC Perp OHLCV")

        logger.info("\nDYDX V4 COLLECTION COMPLETE")


# ═══════════════════════════════════════════════════════════════
#  VALIDATION & SUMMARY
# ═══════════════════════════════════════════════════════════════

def validate_and_summarize():
    """Print summary of all collected data with basic quality checks."""
    logger.info("\n" + "=" * 60)
    logger.info("DATA VALIDATION SUMMARY")
    logger.info("=" * 60)

    parquet_files = list(RAW_DIR.rglob("*.parquet"))
    if not parquet_files:
        logger.warning("No parquet files found!")
        return

    for f in sorted(parquet_files):
        try:
            df = pd.read_parquet(f)
            rel_path = f.relative_to(RAW_DIR)

            logger.info(f"\n  {rel_path}")
            logger.info(f"    Rows: {len(df):,}")
            logger.info(f"    Columns: {list(df.columns)}")

            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                logger.info(f"    Date range: {df['datetime'].min()} → {df['datetime'].max()}")

                if len(df) > 1 and "timestamp" in df.columns:
                    diffs = df["timestamp"].sort_values().diff().dropna()
                    median_gap = diffs.median()
                    max_gap = diffs.max()
                    max_gap_hours = max_gap / (3600 * 1000)
                    logger.info(f"    Median gap: {median_gap/3600000:.1f}h, "
                                f"Max gap: {max_gap_hours:.1f}h")

                    if max_gap_hours > 48:
                        logger.warning(f"    ⚠ Large gap detected: {max_gap_hours:.0f}h")

            if "funding_rate" in df.columns:
                fr = pd.to_numeric(df["funding_rate"], errors="coerce")
                logger.info(f"    Funding rate: mean={fr.mean():.6f}, "
                            f"min={fr.min():.6f}, max={fr.max():.6f}")
                extreme = (fr.abs() > 0.01).sum()
                if extreme > 0:
                    logger.warning(f"    ⚠ {extreme} extreme funding rate entries (|rate| > 1%)")

            if "close" in df.columns:
                logger.info(f"    Price range: ${df['close'].min():,.2f} → "
                            f"${df['close'].max():,.2f}")

            # Per-instrument breakdown for futures files
            if "instrument" in df.columns:
                for inst, grp in df.groupby("instrument"):
                    logger.info(f"      {inst}: {len(grp):,} candles "
                                f"({grp['datetime'].min().date()} → "
                                f"{grp['datetime'].max().date()})")

        except Exception as e:
            logger.error(f"  Error reading {f}: {e}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BTC Futures Curve Trading Data Collector")
    parser.add_argument("--source", choices=["deribit", "hyperliquid", "dydx", "all"],
                        default="all", help="Which source to collect (default: all)")
    parser.add_argument("--start", type=str, default=None,
                        help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip validation summary at end")
    args = parser.parse_args()

    global START_DATE, END_DATE
    if args.start:
        START_DATE = date.fromisoformat(args.start)
    if args.end:
        END_DATE = date.fromisoformat(args.end)

    logger.info(f"Collection range: {START_DATE} → {END_DATE}")
    ensure_dirs()

    if args.source in ("deribit", "all"):
        DeribitCollector().collect_all()

    if args.source in ("hyperliquid", "all"):
        HyperliquidCollector().collect_all()

    if args.source in ("dydx", "all"):
        DydxCollector().collect_all()

    if not args.skip_validation:
        validate_and_summarize()

    logger.info("\nALL DONE. Data saved to: " + str(RAW_DIR.resolve()))


if __name__ == "__main__":
    main()
