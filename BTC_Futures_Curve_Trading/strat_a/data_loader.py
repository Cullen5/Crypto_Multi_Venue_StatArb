"""
BTC Futures Curve Trading — Data Loader
=========================================
Import this in your notebooks:

    from data_loader import DataLoader
    dl = DataLoader("./data/raw")

    # Strategy A: Calendar Spreads
    spot = dl.spot()
    futures = dl.quarterly_futures()
    basis = dl.basis_table()

    # Strategy C: Synthetic Futures from Perp Funding
    funding = dl.funding_rates_all()
    funding_comparison = dl.funding_comparison()
    synthetic = dl.synthetic_term_structure()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class DataLoader:
    """
    Loads and aligns all BTC futures curve trading data.

    Expected directory structure:
        data_dir/
        ├── binance/
        │   ├── btc_spot_ohlcv_1h.parquet
        │   ├── btc_perpetual_ohlcv_1h.parquet
        │   ├── btc_perpetual_funding_rates.parquet
        │   ├── btc_quarterly_futures_ohlcv_1h.parquet
        │   └── btc_quarterly_contracts_summary.parquet
        ├── deribit/
        │   ├── btc_perpetual_ohlcv_1h.parquet
        │   ├── btc_perpetual_funding.parquet
        │   └── btc_quarterly_futures_ohlcv_1h.parquet
        ├── hyperliquid/
        │   ├── btc_funding_rates.parquet
        │   └── btc_perpetual_ohlcv_1h.parquet  (limited history)
        └── dydx/
            ├── btc_funding_rates.parquet
            └── btc_perpetual_ohlcv_1h.parquet
    """

    # All known file mappings: key → list of paths to try (first match wins)
    # This handles both "binance/" subfolder and direct-in-raw layouts.
    FILE_MAP = {
        "binance_spot": [
            "binance/btc_spot_ohlcv_1h.parquet",
            "btc_spot_ohlcv_1h.parquet",
        ],
        "binance_perp": [
            "binance/btc_perpetual_ohlcv_1h.parquet",
            "btc_perpetual_ohlcv_1h.parquet",
        ],
        "binance_funding": [
            "binance/btc_perpetual_funding_rates.parquet",
            "btc_perpetual_funding_rates.parquet",
        ],
        "binance_quarterly": [
            "binance/btc_quarterly_futures_ohlcv_1h.parquet",
            "btc_quarterly_futures_ohlcv_1h.parquet",
        ],
        "binance_contracts": [
            "binance/btc_quarterly_contracts_summary.parquet",
            "btc_quarterly_contracts_summary.parquet",
        ],
        "deribit_perp": [
            "deribit/btc_perpetual_ohlcv_1h.parquet",
            # No flat-dir fallback: btc_perpetual_ohlcv_1h.parquet is dydx
        ],
        "deribit_funding": [
            "deribit/btc_perpetual_funding.parquet",
            "btc_perpetual_funding.parquet",
        ],
        "deribit_quarterly": [
            "deribit/btc_quarterly_futures_ohlcv_1h.parquet",
        ],
        "deribit_contracts": [
            "deribit/btc_quarterly_contracts_summary.parquet",
        ],
        "hyperliquid_funding": [
            "hyperliquid/btc_funding_rates.parquet",
        ],
        "hyperliquid_perp": [
            "hyperliquid/btc_perpetual_ohlcv_1h.parquet",
        ],
        "dydx_funding": [
            "dydx/btc_funding_rates.parquet",
            "btc_funding_rates.parquet",
        ],
        "dydx_perp": [
            "dydx/btc_perpetual_ohlcv_1h.parquet",
            "btc_perpetual_ohlcv_1h.parquet",
        ],
    }

    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = Path(data_dir)
        self._cache = {}
        self._resolved_paths = {}  # key → actual path that exists
        self._validate_paths()

    def _validate_paths(self):
        """Check which data files are available, resolving fallback paths."""
        self.available = {}

        for key, candidates in self.FILE_MAP.items():
            found = False
            for rel_path in candidates:
                full = self.data_dir / rel_path
                if full.exists():
                    self.available[key] = True
                    self._resolved_paths[key] = rel_path
                    found = True
                    break
            if not found:
                self.available[key] = False

        found_keys = [k for k, v in self.available.items() if v]
        missing_keys = [k for k, v in self.available.items() if not v]
        print(f"DataLoader: {self.data_dir}")
        print(f"  Found:   {', '.join(found_keys) if found_keys else '(none)'}")
        if missing_keys:
            print(f"  Missing: {', '.join(missing_keys)}")

    def _resolve(self, key: str) -> str:
        """Get the resolved relative path for a key."""
        if key in self._resolved_paths:
            return self._resolved_paths[key]
        # Fall back to first candidate
        candidates = self.FILE_MAP.get(key, [])
        if candidates:
            return candidates[0]
        raise KeyError(f"Unknown data key: {key}")

    def _load(self, key: str, rel_path: str = None) -> pd.DataFrame:
        """Load a parquet file with caching."""
        if key in self._cache:
            return self._cache[key]

        # Use resolved path if available, otherwise try the provided rel_path
        if rel_path is None:
            rel_path = self._resolve(key)
        path = self.data_dir / rel_path

        if not path.exists():
            # Try resolved path as fallback
            resolved = self._resolve(key)
            path = self.data_dir / resolved
            if not path.exists():
                raise FileNotFoundError(
                    f"{key}: not found. Tried: {self.data_dir / rel_path}, "
                    f"{path}")

        df = pd.read_parquet(path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        # ── Fix mixed μs/ms timestamps ──
        # Some files (e.g. spot 2025 data) have `timestamp` in microseconds
        # instead of milliseconds, which corrupts the `datetime` column.
        df = self._repair_timestamps(df)
        self._cache[key] = df
        return df

    @staticmethod
    def _repair_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and fix rows where the integer `timestamp` column is in
        microseconds (>1e15) instead of the expected milliseconds (1e12–1e13).
        Recomputes `datetime` for affected rows.
        """
        if "timestamp" not in df.columns:
            return df
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        # Microsecond timestamps are ~1e15; millisecond are ~1e12
        bad_mask = ts > 1e15
        if bad_mask.any():
            n_bad = bad_mask.sum()
            # Convert μs → ms for the affected rows
            df = df.copy()
            df.loc[bad_mask, "timestamp"] = ts[bad_mask] // 1000
            # Recompute datetime from the now-fixed timestamps
            if "datetime" in df.columns:
                fixed_dt = pd.to_datetime(
                    df.loc[bad_mask, "timestamp"], unit="ms", utc=True)
                df.loc[bad_mask, "datetime"] = fixed_dt
            print(f"  ⚠ Repaired {n_bad} rows with μs timestamps → ms")
        return df

    def clear_cache(self):
        """Clear cached DataFrames to free memory."""
        self._cache.clear()

    # ═══════════════════════════════════════════════════════════
    #  RAW DATA ACCESSORS
    # ═══════════════════════════════════════════════════════════

    def spot(self) -> pd.DataFrame:
        """BTC spot OHLCV (Binance, 1h)."""
        df = self._load("binance_spot")
        return df.sort_values("datetime").reset_index(drop=True)

    def perpetual(self, venue: str = "binance") -> pd.DataFrame:
        """
        BTC perpetual OHLCV (1h).
        Args:
            venue: 'binance', 'deribit', 'dydx', or 'hyperliquid'
        """
        key_map = {
            "binance": "binance_perp",
            "deribit": "deribit_perp",
            "dydx": "dydx_perp",
            "hyperliquid": "hyperliquid_perp",
        }
        if venue not in key_map:
            raise ValueError(f"Unknown venue: {venue}. "
                             f"Choose from: {list(key_map.keys())}")
        df = self._load(key_map[venue])
        # ── Validate venue matches ──
        # In flat directory layouts the fallback file (e.g.
        # btc_perpetual_ohlcv_1h.parquet) may belong to a different venue.
        # If the file's venue column doesn't match the requested venue,
        # raise FileNotFoundError so callers can fall back to spot.
        if "venue" in df.columns:
            file_venues = set(
                v.lower().replace("_v4", "").replace("_v3", "")
                for v in df["venue"].dropna().unique()
            )
            venue_norm = venue.lower().replace("_v4", "").replace("_v3", "")
            if file_venues and venue_norm not in file_venues:
                raise FileNotFoundError(
                    f"Requested venue '{venue}' but file contains "
                    f"venues {df['venue'].unique().tolist()}.  "
                    f"No dedicated {venue} perpetual OHLCV available."
                )
        return df.sort_values("datetime").reset_index(drop=True)

    def quarterly_futures(self, venue: str = "binance",
                           margin_type: Optional[str] = None) -> pd.DataFrame:
        """
        All BTC quarterly futures OHLCV (1h).

        Args:
            venue: 'binance', 'deribit', or comma-separated like 'binance,deribit'
            margin_type: None (all), 'USDT-M', or 'COIN-M'
        """
        frames = []
        venues = [v.strip() for v in venue.split(",")] if "," in venue else [venue]

        for v in venues:
            if v == "binance":
                df = self._load("binance_quarterly")
            elif v == "deribit":
                df = self._load("deribit_quarterly")
                if "instrument" in df.columns and "symbol" not in df.columns:
                    df["symbol"] = df["instrument"]
                if "margin_type" not in df.columns:
                    df["margin_type"] = "COIN-M"
                if "venue" not in df.columns:
                    df["venue"] = "deribit"
            else:
                raise ValueError(f"Unknown venue: {v}")

            if "venue" not in df.columns:
                df["venue"] = v
            frames.append(df)

        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        if margin_type is not None:
            df = df[df["margin_type"] == margin_type]
        return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    def contracts_summary(self) -> pd.DataFrame:
        """Contract metadata: symbol, first_date, last_date, n_candles, expiry_date."""
        return self._load("binance_contracts")

    def funding_rates(self, venue: str = "binance") -> pd.DataFrame:
        """
        Funding rates for a single venue.

        Returns a DataFrame with 'datetime' and 'funding_rate' columns,
        where funding_rate is the actual per-period rate:
          - Binance: 8h rate reported every 8h
          - Deribit: 1h rate reported every 1h (uses interest_1h)
          - Hyperliquid: 1h rate reported every 1h
          - dYdX: 1h rate reported every 1h

        IMPORTANT — Deribit API returns both interest_8h and interest_1h
        at hourly granularity. interest_8h is the 8h rate extrapolated
        at that instant, NOT a 1h rate. Using it as a 1h rate and summing
        8 of them would overcount by ~8×. We use interest_1h instead.
        """
        key_map = {
            "binance":     "binance_funding",
            "deribit":     "deribit_funding",
            "hyperliquid": "hyperliquid_funding",
            "dydx":        "dydx_funding",
        }
        if venue not in key_map:
            raise ValueError(f"Unknown venue: {venue}. "
                             f"Choose from: {list(key_map.keys())}")
        df = self._load(key_map[venue]).copy()

        # Normalize datetime column
        if "datetime" not in df.columns:
            for alt in ["time", "effectiveAt", "fundingTime"]:
                if alt in df.columns:
                    df["datetime"] = pd.to_datetime(df[alt], utc=True)
                    break

        # Normalize funding_rate column
        if venue == "deribit":
            # Deribit: use interest_1h (true 1h rate), NOT interest_8h
            # interest_8h is the 8h rate snapshot reported every hour —
            # summing 8 of those gives ~8× the real 8h funding.
            if "funding_rate_1h" in df.columns:
                df["funding_rate"] = pd.to_numeric(
                    df["funding_rate_1h"], errors="coerce")
            elif "interest_1h" in df.columns:
                df["funding_rate"] = pd.to_numeric(
                    df["interest_1h"], errors="coerce")
            else:
                # Fallback: divide interest_8h by 8 to approximate 1h rate
                if "funding_rate" in df.columns:
                    df["funding_rate"] = pd.to_numeric(
                        df["funding_rate"], errors="coerce") / 8.0
                elif "interest_8h" in df.columns:
                    df["funding_rate"] = pd.to_numeric(
                        df["interest_8h"], errors="coerce") / 8.0
        else:
            # All other venues: standard column normalization
            if "funding_rate" not in df.columns:
                for alt in ["fundingRate", "rate", "funding"]:
                    if alt in df.columns:
                        df = df.rename(columns={alt: "funding_rate"})
                        break

        return df.sort_values("datetime").reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════
    #  STRATEGY A: CALENDAR SPREADS — Derived Data
    # ═══════════════════════════════════════════════════════════

    def active_contracts_at(self, dt_index: pd.DatetimeIndex) -> pd.DataFrame:
        """For each timestamp, identify which quarterly contracts are active."""
        contracts = self.contracts_summary()
        records = []
        for _, row in contracts.iterrows():
            records.append({
                "symbol": row["symbol"],
                "first_date": pd.to_datetime(row["first_date"], utc=True),
                "last_date": pd.to_datetime(row["last_date"], utc=True),
                "expiry_date": row["expiry_date"],
            })
        return pd.DataFrame(records)

    def basis_table(self, margin_type: str = "USDT-M",
                     venue: str = "binance") -> pd.DataFrame:
        """
        Compute basis (futures - spot) for all quarterly contracts.

        Returns hourly DataFrame with columns:
            datetime, spot_close, symbol, futures_close, expiry_date,
            venue, basis, basis_pct, days_to_expiry, annualized_basis_pct
        """
        spot = self.spot()[["datetime", "close"]].rename(
            columns={"close": "spot_close"})
        futures = self.quarterly_futures(venue=venue, margin_type=margin_type)
        futures["expiry_dt"] = pd.to_datetime(futures["expiry_date"], utc=True)

        merged = futures.merge(spot, on="datetime", how="inner")
        merged["basis"] = merged["close"] - merged["spot_close"]
        merged["basis_pct"] = merged["basis"] / merged["spot_close"] * 100
        merged["days_to_expiry"] = (
            merged["expiry_dt"] - merged["datetime"]
        ).dt.total_seconds() / 86400
        merged["annualized_basis_pct"] = np.where(
            merged["days_to_expiry"] > 0,
            merged["basis_pct"] * (365.25 / merged["days_to_expiry"]),
            np.nan
        )

        cols = [
            "datetime", "spot_close", "symbol", "close", "expiry_date",
            "margin_type", "venue", "days_to_expiry", "basis", "basis_pct",
            "annualized_basis_pct"
        ]
        result = merged[cols].rename(columns={"close": "futures_close"})
        return result.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    def continuous_front_month(self, margin_type: str = "USDT-M",
                                roll_days_before_expiry: int = 5,
                                venue: str = "binance") -> pd.DataFrame:
        """
        Build a continuous front-month futures series by rolling N days
        before expiry.
        """
        basis = self.basis_table(margin_type=margin_type, venue=venue)
        basis = basis[basis["days_to_expiry"] > roll_days_before_expiry].copy()
        idx = basis.groupby("datetime")["days_to_expiry"].idxmin()
        front = basis.loc[idx].sort_values("datetime").reset_index(drop=True)
        return front

    def term_structure_snapshot(self, target_date: str,
                                 margin_type: str = "USDT-M") -> pd.DataFrame:
        """
        Get the term structure at a specific date (all active contracts).
        """
        basis = self.basis_table(margin_type=margin_type)
        target = pd.Timestamp(target_date, tz="UTC")

        snap = basis[basis["datetime"].dt.date == target.date()]
        if snap.empty:
            closest_idx = (basis["datetime"] - target).abs().idxmin()
            snap_date = basis.loc[closest_idx, "datetime"].date()
            snap = basis[basis["datetime"].dt.date == snap_date]
            print(f"  No data for {target_date}, using nearest: {snap_date}")

        snap = snap.sort_values("datetime").groupby("symbol").last().reset_index()
        snap = snap.sort_values("days_to_expiry")
        return snap

    # ═══════════════════════════════════════════════════════════
    #  STRATEGY C: SYNTHETIC FUTURES — Derived Data
    # ═══════════════════════════════════════════════════════════

    def funding_rates_all(self) -> pd.DataFrame:
        """
        Load and normalize funding rates from all available venues into
        a single DataFrame.

        Returns columns:
            datetime, funding_rate, venue, venue_type, frequency
        """
        venue_config = {
            "binance":     {"key": "binance_funding",     "type": "CEX",    "freq": "8h"},
            "deribit":     {"key": "deribit_funding",      "type": "CEX",    "freq": "1h"},
            "hyperliquid": {"key": "hyperliquid_funding",  "type": "hybrid", "freq": "1h"},
            "dydx":        {"key": "dydx_funding",         "type": "hybrid", "freq": "1h"},
        }

        frames = []
        for venue, cfg in venue_config.items():
            if not self.available.get(cfg["key"]):
                continue
            try:
                df = self.funding_rates(venue)
                if "funding_rate" not in df.columns or "datetime" not in df.columns:
                    print(f"  WARNING: {venue} funding missing required columns, "
                          f"has: {list(df.columns)}")
                    continue
                frames.append(pd.DataFrame({
                    "datetime": df["datetime"],
                    "funding_rate": pd.to_numeric(
                        df["funding_rate"], errors="coerce"),
                    "venue": venue,
                    "venue_type": cfg["type"],
                    "frequency": cfg["freq"],
                }))
                print(f"  Loaded {venue} funding: {len(df):,} rows")
            except Exception as e:
                print(f"  WARNING: Failed to load {venue} funding: {e}")

        if not frames:
            raise ValueError(
                "No funding rate data found. Available files: "
                + str({k: v for k, v in self.available.items() if v}))

        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values(["venue", "datetime"]).reset_index(drop=True)

    def funding_comparison(self, resample: str = "8h") -> pd.DataFrame:
        """
        Align funding rates across venues on a common time grid.

        Resamples hourly venues to match Binance's 8h frequency by summing.

        Returns wide DataFrame:
            datetime, binance, deribit, dydx, spread_binance_deribit, ...
        """
        all_funding = self.funding_rates_all()
        venues = all_funding["venue"].unique()

        resampled = {}
        for venue in venues:
            vdf = all_funding[all_funding["venue"] == venue].copy()
            vdf = vdf.set_index("datetime")["funding_rate"]

            freq = all_funding.loc[
                all_funding["venue"] == venue, "frequency"].iloc[0]

            if freq == "1h" and resample == "8h":
                vdf = vdf.resample("8h").sum()
            elif freq == "1h" and resample == "1D":
                vdf = vdf.resample("1D").sum()
            elif freq == "8h" and resample == "1D":
                vdf = vdf.resample("1D").sum()
            elif freq == "8h" and resample == "8h":
                # Snap Binance's slightly irregular 8h timestamps
                # (e.g. 08:00:00.009) to a clean 8h grid so indices
                # align with the resampled hourly venues.
                vdf = vdf.resample("8h").sum()

            resampled[venue] = vdf

        result = pd.DataFrame(resampled)
        result.index.name = "datetime"
        result = result.reset_index()

        from itertools import combinations
        venue_list = [v for v in venues if v in result.columns]
        for va, vb in combinations(venue_list, 2):
            result[f"spread_{va}_{vb}"] = result[va] - result[vb]

        return result.dropna(subset=list(venues), how="all")

    def synthetic_term_structure(self,
                                  horizon_days: list[int] = None) -> pd.DataFrame:
        """
        Construct synthetic futures prices from perpetual funding rates.
        """
        if horizon_days is None:
            horizon_days = [30, 90, 180, 365]

        spot = self.spot()[["datetime", "close"]].rename(
            columns={"close": "spot_close"})
        spot = spot.set_index("datetime").resample("1D").last().dropna()

        all_funding = self.funding_rates_all()
        venues = all_funding["venue"].unique()

        records = []
        for venue in venues:
            vdf = all_funding[all_funding["venue"] == venue].copy()
            vdf = vdf.set_index("datetime")["funding_rate"].resample("1D").sum()
            avg_daily = vdf.rolling(7, min_periods=1).mean()

            aligned = pd.DataFrame({
                "spot_close": spot["spot_close"],
                "avg_daily_funding": avg_daily,
            }).dropna()

            for h in horizon_days:
                for dt_val, row in aligned.iterrows():
                    records.append({
                        "datetime": dt_val,
                        "venue": venue,
                        "spot_close": row["spot_close"],
                        "horizon_days": h,
                        "avg_daily_funding": row["avg_daily_funding"],
                        "annualized_rate_pct": row["avg_daily_funding"] * 365.25 * 100,
                        "synthetic_futures_price": row["spot_close"] * (
                            1 + row["avg_daily_funding"] * h),
                        "implied_basis_pct": row["avg_daily_funding"] * h * 100,
                    })

        return pd.DataFrame(records)

    def synthetic_vs_actual(self,
                             margin_type: str = "USDT-M") -> pd.DataFrame:
        """Compare synthetic futures (from funding) to actual futures prices."""
        front = self.continuous_front_month(margin_type=margin_type)
        actual = front[["datetime", "annualized_basis_pct", "days_to_expiry"]].copy()
        actual = actual.set_index("datetime").resample("1D").last().dropna()

        all_funding = self.funding_rates_all()
        venues = all_funding["venue"].unique()

        records = []
        for venue in venues:
            vdf = all_funding[all_funding["venue"] == venue].copy()
            daily = vdf.set_index("datetime")["funding_rate"].resample("1D").sum()
            annualized = daily.rolling(7, min_periods=1).mean() * 365.25 * 100

            aligned = pd.DataFrame({
                "actual_ann_basis_pct": actual["annualized_basis_pct"],
                "synthetic_ann_rate_pct": annualized,
            }).dropna()

            for dt_val, row in aligned.iterrows():
                records.append({
                    "datetime": dt_val,
                    "venue": venue,
                    "actual_ann_basis_pct": row["actual_ann_basis_pct"],
                    "synthetic_ann_rate_pct": row["synthetic_ann_rate_pct"],
                    "discrepancy_pct": (row["actual_ann_basis_pct"]
                                        - row["synthetic_ann_rate_pct"]),
                })

        return pd.DataFrame(records)

    # ═══════════════════════════════════════════════════════════
    #  REGIME CLASSIFICATION
    # ═══════════════════════════════════════════════════════════

    def classify_regime(self, margin_type: str = "USDT-M") -> pd.DataFrame:
        """
        Classify the futures curve regime over time.

        Regimes (based on front-month annualized basis):
            steep_contango:  >20%
            mild_contango:   5% to 20%
            flat:            -5% to 5%
            backwardation:   <-5%
        """
        front = self.continuous_front_month(margin_type=margin_type)

        def _regime(ann_basis):
            if ann_basis > 20:
                return "steep_contango"
            elif ann_basis > 5:
                return "mild_contango"
            elif ann_basis > -5:
                return "flat"
            else:
                return "backwardation"

        front["regime"] = front["annualized_basis_pct"].apply(_regime)
        return front[["datetime", "spot_close", "annualized_basis_pct",
                       "regime", "symbol"]]

    # ═══════════════════════════════════════════════════════════
    #  CONVENIENCE / SUMMARY
    # ═══════════════════════════════════════════════════════════

    def summary(self):
        """Print a summary of all loaded data."""
        print("\n" + "=" * 65)
        print("DATA SUMMARY")
        print("=" * 65)

        datasets = [
            ("Binance Spot",        "binance_spot"),
            ("Binance Perp",        "binance_perp"),
            ("Binance Funding",     "binance_funding"),
            ("Binance Quarterly",   "binance_quarterly"),
            ("Deribit Perp",        "deribit_perp"),
            ("Deribit Funding",     "deribit_funding"),
            ("Deribit Quarterly",   "deribit_quarterly"),
            ("Hyperliquid Funding", "hyperliquid_funding"),
            ("Hyperliquid Perp",    "hyperliquid_perp"),
            ("dYdX Funding",        "dydx_funding"),
            ("dYdX Perp",           "dydx_perp"),
        ]

        for label, key in datasets:
            if not self.available.get(key):
                print(f"\n  {label}: NOT FOUND")
                continue
            try:
                df = self._load(key)
                n = len(df)
                if "datetime" in df.columns:
                    dt_col = pd.to_datetime(df["datetime"], errors="coerce")
                    dt_col = dt_col.dropna()
                    if dt_col.empty:
                        date_str = "all timestamps invalid"
                    else:
                        # Clamp to valid range to avoid out-of-range errors
                        min_valid = pd.Timestamp("1970-01-01", tz="UTC")
                        max_valid = pd.Timestamp("2100-01-01", tz="UTC")
                        dt_col = dt_col[dt_col.between(min_valid, max_valid)]
                        if dt_col.empty:
                            date_str = "all timestamps out of range"
                        else:
                            dt_min = dt_col.min()
                            dt_max = dt_col.max()
                            date_str = f"{dt_min.date()} → {dt_max.date()}"
                else:
                    date_str = "no datetime column"
                print(f"\n  {label}: {n:,} rows  |  {date_str}")

                if "close" in df.columns:
                    print(f"    Price: ${df['close'].min():,.0f}"
                          f" – ${df['close'].max():,.0f}")
                if "funding_rate" in df.columns:
                    fr = df["funding_rate"]
                    print(f"    Funding: mean={fr.mean():.6f}"
                          f"  min={fr.min():.6f}  max={fr.max():.6f}")
                if "symbol" in df.columns and df["symbol"].nunique() > 1:
                    syms = df["symbol"].unique()
                    print(f"    Contracts: {len(syms)} "
                          f"({', '.join(syms[:5])}...)")
            except Exception as e:
                print(f"\n  {label}: ERROR — {e}")

        print("\n" + "=" * 65)


# ═══════════════════════════════════════════════════════════════
#  QUICK USAGE (when run directly)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    dl = DataLoader("./data/raw")
    dl.summary()
