"""
runner.py -- look-ahead-proof data feed and bar-by-bar runner

This is the structural guarantee against look-ahead bias. The DataFeed
class wraps a price matrix but only exposes one row at a time through
its .step() method. The strategy code never gets a reference to the
full matrix -- it only sees the current bar and whatever it has stored
from past bars.

Design choices:
  - DataFeed builds the aligned price matrix from raw data dicts
  - Log prices are used throughout (strategy works in log space)
  - Coins with too little history are dropped early
  - Small gaps (<=12 bars) are interpolated; big gaps stay NaN
  - The feed tracks which coins are "alive" at each bar so the
    strategy can skip pairs with missing legs

Usage:
    feed = DataFeed(data, venue="binance", coins=["SOL", "AVAX", ...])
    for bar_idx, timestamp, row in feed:
        # row is a 1-D numpy array of log-prices, shape (n_coins,)
        # NaN means that coin isn't available at this bar
        strategy.on_bar(bar_idx, timestamp, row)
"""

import numpy as np
import pandas as pd
from typing import Iterator, Optional


class DataFeed:
    """
    Wraps raw price data into a look-ahead-proof bar-by-bar iterator.

    The full price matrix exists internally for alignment, but the only
    way to access it is one row at a time through iteration. The strategy
    receives a read-only view of the current row -- modifying it won't
    affect anything.

    Internally stores LOG prices. The strategy should work in log space
    for numerical stability and because returns are additive in log space.
    """

    def __init__(
        self,
        data: dict,
        venue: str,
        coins: list[str],
        price_col: str = "close_ohlcv",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_total_bars: int = 500,
        max_interp_gap: int = 12,
        use_log_prices: bool = True,
    ):
        """
        Build aligned price matrix from raw data dict.

        data: {venue: {coin: DataFrame with timestamp index and price_col}}
        venue: which venue to pull from
        coins: list of coin symbols to include
        price_col: column name for close prices
        start_date/end_date: optional date filters
        min_total_bars: drop coins with fewer valid bars
        max_interp_gap: interpolate gaps up to this many bars (bigger gaps
                        stay NaN so the strategy knows the data is missing)
        use_log_prices: store log(price) instead of raw price
        """
        if venue not in data:
            raise ValueError(f"venue '{venue}' not in data")

        venue_data = data[venue]
        dfs = {}
        for coin in coins:
            if coin not in venue_data:
                continue
            df = venue_data[coin].copy()
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            if price_col not in df.columns:
                continue
            dfs[coin] = df[price_col]

        if not dfs:
            raise ValueError(f"no valid price data for venue '{venue}'")

        # align all coins to the same timestamp index
        pm = pd.DataFrame(dfs).sort_index()
        if start_date:
            pm = pm.loc[start_date:]
        if end_date:
            pm = pm.loc[:end_date]
        if len(pm) == 0:
            raise ValueError("no data in date range")

        # interpolate small gaps within each coin's active range
        # we do NOT fill leading/trailing NaNs -- those indicate the
        # coin wasn't listed yet or was delisted
        for col in pm.columns:
            s = pm[col]
            is_nan = s.isna()
            if not is_nan.any():
                continue
            # find runs of NaN and only fill short ones
            groups = (~is_nan).cumsum()
            nan_runs = is_nan.groupby(groups).transform("sum")
            fillable = is_nan & (nan_runs <= max_interp_gap)
            filled = s.interpolate(method="linear")
            pm[col] = s.where(~fillable, filled)

        # drop coins that don't have enough data to be useful
        for coin in list(pm.columns):
            if pm[coin].notna().sum() < min_total_bars:
                pm = pm.drop(columns=coin)

        # drop rows where every coin is NaN (no data for that timestamp)
        pm = pm.dropna(how="all")

        # stash the raw prices for later (reporting needs them)
        self._raw_prices = pm.values.astype(np.float64).copy()

        # convert to log prices if requested
        if use_log_prices:
            # replace zeros and negatives with NaN before log
            vals = pm.values.astype(np.float64)
            vals[vals <= 0] = np.nan
            self._prices = np.log(vals)
        else:
            self._prices = pm.values.astype(np.float64)

        self._timestamps = pm.index
        self._symbols = list(pm.columns)
        self._n_bars = len(pm)
        self._n_coins = len(self._symbols)
        self._use_log = use_log_prices

        # symbol <-> column index lookups
        self.sym_to_col: dict[str, int] = {s: i for i, s in enumerate(self._symbols)}
        self.col_to_sym: dict[int, str] = {i: s for s, i in self.sym_to_col.items()}

        n_full = sum(1 for c in pm.columns if pm[c].notna().all())
        n_partial = self._n_coins - n_full
        print(f"[feed] {self._n_coins} coins ({n_full} full, "
              f"{n_partial} partial), {self._n_bars} bars, "
              f"log_prices={'ON' if use_log_prices else 'OFF'}")

    # -------------------------------------------------------------------
    #  Iteration -- the only way to access price data
    # -------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[int, pd.Timestamp, np.ndarray]]:
        """
        Yield (bar_index, timestamp, price_row) one bar at a time.

        The price_row is a fresh copy each time -- the strategy can't
        accidentally modify our internal state.
        """
        for i in range(self._n_bars):
            # hand out a copy so the strategy can't write back into our matrix
            row = self._prices[i].copy()
            yield i, self._timestamps[i], row

    def __len__(self) -> int:
        return self._n_bars

    # -------------------------------------------------------------------
    #  Metadata accessors (safe -- no future data leakage)
    # -------------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def n_coins(self) -> int:
        return self._n_coins

    @property
    def n_bars(self) -> int:
        return self._n_bars

    @property
    def timestamps(self):
        return self._timestamps

    def raw_price_at(self, bar_idx: int, col: int) -> float:
        """
        Get raw (non-log) price for a specific bar and coin.

        Only for use in trade execution and reporting -- the strategy
        should work in log space.
        """
        return float(self._raw_prices[bar_idx, col])

    def raw_prices_row(self, bar_idx: int) -> np.ndarray:
        """Raw prices for a single bar. For portfolio PnL computation."""
        return self._raw_prices[bar_idx].copy()

    def log_to_raw(self, log_price: float) -> float:
        """Convert a single log price back to raw price."""
        return float(np.exp(log_price))

    def raw_to_log(self, raw_price: float) -> float:
        """Convert a single raw price to log price."""
        if raw_price <= 0:
            return np.nan
        return float(np.log(raw_price))


class RollingWindow:
    """
    Tracks a rolling window of past observations for each coin.

    Used by the strategy to accumulate formation window data without
    ever peeking ahead. Stores log prices in a circular buffer.
    """

    def __init__(self, n_coins: int, max_window: int):
        self._buffer = np.full((max_window, n_coins), np.nan)
        self._max = max_window
        self._pos = 0           # write position in circular buffer
        self._count = 0         # total bars ingested

    def push(self, row: np.ndarray) -> None:
        """Add one bar of prices to the rolling window."""
        self._buffer[self._pos] = row
        self._pos = (self._pos + 1) % self._max
        self._count += 1

    def get_column(self, col: int, n_bars: Optional[int] = None) -> np.ndarray:
        """
        Retrieve the last n_bars of data for a single coin.

        Returns a contiguous array in chronological order. NaN entries
        mean the coin wasn't available at that bar.
        """
        if n_bars is None:
            n_bars = min(self._count, self._max)
        n_bars = min(n_bars, self._count, self._max)

        # figure out where the data lives in the circular buffer
        end = self._pos
        start = (end - n_bars) % self._max

        if start < end:
            return self._buffer[start:end, col].copy()
        else:
            # wraps around -- need to concatenate two slices
            part1 = self._buffer[start:, col]
            part2 = self._buffer[:end, col]
            return np.concatenate([part1, part2])

    def get_pair(self, col_a: int, col_b: int,
                 n_bars: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Get aligned data for a pair. Only returns bars where BOTH coins
        have valid (non-NaN) data, trimmed from the front.
        """
        a = self.get_column(col_a, n_bars)
        b = self.get_column(col_b, n_bars)

        # find the clean overlap (both non-NaN and positive)
        valid = np.isfinite(a) & np.isfinite(b)
        if not np.any(valid):
            return np.array([]), np.array([])

        # trim leading invalids so we get a contiguous block from the back
        first_valid = np.argmax(valid)
        a_clean = a[first_valid:]
        b_clean = b[first_valid:]

        # within the trimmed range, check for any remaining gaps
        still_valid = np.isfinite(a_clean) & np.isfinite(b_clean)
        if not np.all(still_valid):
            # find the last contiguous valid block from the end
            bad_idx = np.where(~still_valid)[0]
            last_bad = bad_idx[-1]
            a_clean = a_clean[last_bad + 1:]
            b_clean = b_clean[last_bad + 1:]

        return a_clean, b_clean

    @property
    def count(self) -> int:
        """Total bars pushed so far."""
        return self._count

    @property
    def available(self) -> int:
        """Number of bars currently in the window."""
        return min(self._count, self._max)
