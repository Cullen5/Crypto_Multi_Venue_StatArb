"""
strategy.py -- Kalman pairs trading engine (look-ahead proof)

Pipeline per bar:
  1. Vol tracker update
  2. Push into rolling window
  3. Refit if due (screen, Kalman init, BH-FDR)
  4. Force-close positions with missing data
  5. Batch Kalman update -> signals
  6. Exit cascade (stop, z-hard-stop, max-hold, beta-break, stale, signal, rebalance)
  7. Entry logic (z-threshold, regime filter, confidence sizing, sector limits)
  8. Liquidation check
"""

import numpy as np
import pandas as pd
import time as _time
import warnings
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from .models import CompletedTrade
from .portfolio import Portfolio
from .margin import MarginConfig
from .runner import DataFeed, RollingWindow
from .kalman import KalmanBatch
from .screening import screen_pairs, rank_and_select, fast_half_life
from .costs import get_venue_costs
from .metrics import compute_all_metrics, print_summary
from .volatility import (
    EWMAVolTracker, vol_adjusted_notionals,
    implied_correlation, regime_confidence,
)

warnings.filterwarnings("ignore")



_MIN_HOLD_BARS = 8           # don't exit before this (noise guard)
_HL_MIN_MULT = 0.2           # min hold floor = half_life * this
_REBALANCE_MIN_INTERVAL = 96 # don't rebalance more than once per day
_PAIR_COOLDOWN_BARS = 672    # blacklist pair after stop-loss (~7 days)
_BETA_BREAK_PCT = 0.10       # beta drift >= 10% → structural break, close + cooldown
_BETA_BREAK_COOLDOWN = 672   # 7 days at 96 bars/day
_BETA_MAX_CV = 0.40          # max coeff of variation across refits
_SNAPSHOT_EVERY = 4          # record equity every N bars
_LOG_INTERVAL = 2880         # print progress every N bars
_BARS_PER_DAY = 96           # 15-min bars
_VOL_HALFLIFE = 672.0        # EWMA vol halflife (~7 days @ 15min)
_VOL_MIN_HISTORY = 96        # min bars before vol is valid (~1 day)
_Z_WINDOW = 2000             # rolling window for z-score mean/std
_Z_MIN_BARS = 50             # min bars before z-scores are valid
_N_JOBS = -1                 # parallelism for screening
_WARMUP_BARS = 1000          # bars to initialize Kalman state



@dataclass
class BacktestResults:
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    metrics: dict
    pair_stats: Optional[pd.DataFrame] = None
    spread_data: Optional[dict] = None
    config: dict = field(default_factory=dict)

    def summary(self):
        print_summary(self.metrics, title="Pairs Backtest")



class PairsEngine:
    """Stateful pairs trading engine. Feed one bar at a time via on_bar()."""

    def __init__(
        self,
        feed: DataFeed,
        sectors: dict[str, list[str]],
        venue: str = "binance",
        initial_capital: float = 100_000,
        # --- walk-forward ---
        formation_window: int = 4320,
        refit_interval: int = 2880,
        screen_lookback: int = 2000,
        # --- screening ---
        max_p: float = 0.05,
        fdr_q: float = 0.05,
        min_return_corr: float = 0.40,
        # --- kalman ---
        default_delta: float = 1e-4,
        R: float = 0.1,
        # --- signals ---
        entry_z: float = 2.0,
        exit_z: float = 0.75,
        z_hard_stop: float = 3.5,
        # --- risk ---
        max_open: int = 20,
        position_pct: float = 0.10,
        pct_stop: float = 0.06,
        max_sector_pairs: int = 6,
        # --- pair selection / ranking ---
        max_pairs: int = 30,
        max_coin_appearances: int = 4,
        hl_optimal_low: float = 48.0,
        hl_optimal_high: float = 192.0,
        # --- hold limits ---
        hl_max_mult: float = 6.0,
        # --- rebalancing ---
        rebalance_beta_drift: float = 0.05,
        # --- confidence sizing ---
        confidence_power: float = 2.5,
        # --- regime filter ---
        rho_high: float = 0.55,
        rho_low: float = 0.10,
        # --- stale trade exit ---
        stale_z_progress: float = 0.30,
        # --- leverage / margin ---
        max_leverage: float = 1.0,
        maintenance_margin_rate: float = 0.05,
        liquidation_fee_rate: float = 0.02,
        margin_interest_rate: float = 0.0003,
        interest_interval_bars: int = 4,
    ):
        self.feed = feed
        self.sectors = sectors
        self.venue = venue
        self.initial_capital = initial_capital

        self.formation_window = formation_window
        self.refit_interval = refit_interval
        self.screen_lookback = screen_lookback
        self.max_p = max_p
        self.fdr_q = fdr_q
        self.min_return_corr = min_return_corr
        self.default_delta = default_delta
        self.R = R
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.z_hard_stop = z_hard_stop
        self.max_open = max_open
        self.position_pct = position_pct
        self.pct_stop = pct_stop
        self.max_sector_pairs = max_sector_pairs
        self.max_pairs = max_pairs
        self.max_coin_appearances = max_coin_appearances
        self.hl_optimal_low = hl_optimal_low
        self.hl_optimal_high = hl_optimal_high
        self.hl_max_mult = hl_max_mult
        self.rebalance_beta_drift = rebalance_beta_drift
        self.confidence_power = confidence_power
        self.rho_high = rho_high
        self.rho_low = rho_low
        self.stale_z_progress = stale_z_progress
        self.max_leverage = max_leverage

        self._margin_config = MarginConfig(
            max_leverage=max_leverage,
            maintenance_margin_rate=maintenance_margin_rate,
            liquidation_fee_rate=liquidation_fee_rate,
            margin_interest_rate=margin_interest_rate,
            interest_interval_bars=interest_interval_bars,
        )

        if screen_lookback > formation_window:
            print(f"  [warn] screen_lookback ({screen_lookback}) > formation_window "
                  f"({formation_window}). Window auto-sized to "
                  f"{max(formation_window, screen_lookback)}.")

        self.vc = get_venue_costs(venue)
        self.entry_cost_rate = self.vc.taker_fee + self.vc.avg_slippage
        self.exit_cost_rate = self.vc.maker_fee + self.vc.avg_slippage

        self.S2C = feed.sym_to_col
        self.C2S = feed.col_to_sym

        self.active_sectors = {}
        for sn, sc in sectors.items():
            avail = [c for c in sc if c in self.S2C]
            if len(avail) >= 2:
                self.active_sectors[sn] = avail

        self.portfolio = Portfolio(initial_capital, margin_config=self._margin_config)

        self.kb = KalmanBatch(
            default_delta=default_delta, R=R,
            z_window=_Z_WINDOW, z_min_bars=_Z_MIN_BARS,
            use_mle=True)

        self.vol_tracker = EWMAVolTracker(
            n_assets=feed.n_coins,
            halflife_bars=_VOL_HALFLIFE,
            min_history=_VOL_MIN_HISTORY)

        window_size = max(formation_window, screen_lookback, _WARMUP_BARS)
        self.window = RollingWindow(feed.n_coins, window_size)

        self.pair_meta: dict[str, dict] = {}
        self.pair_col_a: dict[str, int] = {}
        self.pair_col_b: dict[str, int] = {}
        self.active_sids: set[str] = set()
        self.last_refit = -refit_interval
        self.last_known = np.full(feed.n_coins, np.nan)

        self._last_rebalance: dict[str, int] = {}
        self._pair_cooldown: dict[str, int] = {}
        self._beta_history: dict[str, list[float]] = {}

        self.spread_history: dict[str, dict] = {}

        self._regime_passed = 0
        self._regime_rejected = 0
        self._rebalances = 0
        self._beta_breaks = 0
        self._beta_break_cooldown: dict[str, int] = {}
        self._cooldown_rejected = 0
        self._beta_stability_rejected = 0
        self._stale_exits = 0
        self._liquidations = 0
        self._interest_total = 0.0
        self._margin_capped_entries = 0

    @staticmethod
    def _sid(sym_a: str, sym_b: str) -> str:
        return f"{sym_a}__{sym_b}"

    def _get_raw_prices(self, pair_id: str, bar_idx: int):
        """Get raw (non-log) prices for a pair at the current bar."""
        ca = self.pair_col_a.get(pair_id)
        cb = self.pair_col_b.get(pair_id)
        if ca is None or cb is None:
            return None, None
        pa = self.feed.raw_price_at(bar_idx, ca)
        pb = self.feed.raw_price_at(bar_idx, cb)
        if np.isnan(pa) or np.isnan(pb) or pa <= 0 or pb <= 0:
            lk_a = self.last_known[ca]
            lk_b = self.last_known[cb]
            if np.isnan(pa) or pa <= 0:
                pa = self.feed.log_to_raw(lk_a) if np.isfinite(lk_a) else np.nan
            if np.isnan(pb) or pb <= 0:
                pb = self.feed.log_to_raw(lk_b) if np.isfinite(lk_b) else np.nan
        if np.isnan(pa) or np.isnan(pb):
            return None, None
        return float(pa), float(pb)

    def _close_position(self, pair_id: str, bar_idx: int,
                         reason: str, z: float = 0.0, sv: float = 0.0):
        """Close a spread position and record it."""
        pa, pb = self._get_raw_prices(pair_id, bar_idx)
        if pa is None:
            pos = self.portfolio.open_positions.get(pair_id)
            if pos is None:
                return
            pa, pb = pos.entry_price_a, pos.entry_price_b
        self.portfolio.close_spread(
            pair_id, bar_idx, pa, pb, z, reason, self.exit_cost_rate,
            exit_spread=sv,
            exit_beta=self.kb.get_beta(pair_id),
            exit_beta_P11=self.kb.get_P11(pair_id))

        if reason in ("stop_loss", "z_stop"):
            self._pair_cooldown[pair_id] = bar_idx

    def _check_beta_drift(self, sid: str, bar_idx: int) -> float:
        """Return fractional beta drift from entry (0.0 if too soon to check)."""
        pos = self.portfolio.open_positions.get(sid)
        if pos is None:
            return 0.0

        last = self._last_rebalance.get(sid, pos.entry_bar)
        if bar_idx - last < _REBALANCE_MIN_INTERVAL:
            return 0.0

        current_beta = self.kb.get_beta(sid)
        entry_beta = pos.beta
        if abs(entry_beta) < 1e-8:
            return 0.0

        return abs(current_beta - entry_beta) / abs(entry_beta)

    def _refit(self, bar_idx: int, timestamp, log_row: np.ndarray):
        """Screen for cointegrated pairs using the rolling window."""
        t_refit = _time.perf_counter()

        clean_syms = []
        for sym, col in self.S2C.items():
            data = self.window.get_column(col, self.screen_lookback)
            n_valid = np.sum(np.isfinite(data))
            if n_valid >= self.screen_lookback:
                clean_syms.append(sym)

        if len(clean_syms) < 2:
            return

        price_dict = {}
        for sym in clean_syms:
            col = self.S2C[sym]
            arr = self.window.get_column(col, self.screen_lookback)
            valid_mask = np.isfinite(arr)
            if not np.any(valid_mask):
                continue
            first_valid = np.argmax(valid_mask)
            arr = arr[first_valid:]
            if len(arr) >= self.screen_lookback:
                price_dict[sym] = arr

        sector_subset = {
            sn: [c for c in sc if c in price_dict]
            for sn, sc in self.active_sectors.items()
        }
        sector_subset = {k: v for k, v in sector_subset.items() if len(v) >= 2}

        if not sector_subset:
            return

        cands = screen_pairs(
            price_dict, sector_subset,
            max_p=self.max_p,
            min_return_corr=self.min_return_corr,
            fdr_q=self.fdr_q,
            n_jobs=_N_JOBS, verbose=True)

        cands = rank_and_select(
            cands,
            n_select=self.max_pairs,
            max_per_sector=self.max_sector_pairs,
            max_per_coin=self.max_coin_appearances,
            hl_optimal_low=self.hl_optimal_low,
            hl_optimal_high=self.hl_optimal_high,
            verbose=True)

        new_active = set()
        for c in cands:
            sid = self._sid(c["sym_a"], c["sym_b"])
            new_active.add(sid)

        for sid in list(self.active_sids - new_active):
            if sid in self.portfolio.open_positions:
                self._close_position(sid, bar_idx, "forced")

        for c in cands:
            sid = self._sid(c["sym_a"], c["sym_b"])
            ca, cb = self.S2C[c["sym_a"]], self.S2C[c["sym_b"]]

            pa, pb = self.window.get_pair(ca, cb, self.formation_window)
            if len(pa) < self.screen_lookback:
                new_active.discard(sid)
                continue

            self.kb.add(sid, ca, cb, pa, pb,
                        c["hedge_ratio"], _WARMUP_BARS,
                        half_life=c["half_life"])
            self.pair_col_a[sid] = ca
            self.pair_col_b[sid] = cb
            self.pair_meta[sid] = {
                "sym_a": c["sym_a"], "sym_b": c["sym_b"],
                "sector": c["sector"],
                "half_life": c["half_life"],
                "p_value": c["p_value"],
                "hedge_ratio": c["hedge_ratio"],
                "composite_score": c.get("composite_score", 0.0),
                "return_corr": c.get("return_corr", 0.0),
            }

            if sid not in self._beta_history:
                self._beta_history[sid] = []
            self._beta_history[sid].append(c["hedge_ratio"])
            if len(self._beta_history[sid]) > 5:
                self._beta_history[sid] = self._beta_history[sid][-5:]

            if sid not in self.spread_history:
                self.spread_history[sid] = {
                    "timestamps": [], "spreads": [], "z_scores": [],
                    "bar_indices": [], "betas": [], "implied_rhos": [],
                    "prices_a": [], "prices_b": [],
                    "sym_a": c["sym_a"], "sym_b": c["sym_b"],
                }

        self.active_sids = new_active
        self.kb._dirty = True

        elapsed_r = _time.perf_counter() - t_refit
        print(f"  [{timestamp.date() if hasattr(timestamp, 'date') else timestamp}] "
              f"REFIT: {len(cands)} pairs from "
              f"{len(clean_syms)}/{self.feed.n_coins} coins | {elapsed_r:.1f}s")

    def on_bar(self, bar_idx: int, timestamp, log_row: np.ndarray) -> bool:
        """Process a single bar. Returns True if any trade was executed."""
        traded = False
        raw_row = self.feed.raw_prices_row(bar_idx)

        self.vol_tracker.update(raw_row)
        self.window.push(log_row)

        valid = np.isfinite(log_row)
        self.last_known[valid] = log_row[valid]

        if self._margin_config.is_leveraged:
            eq_for_interest = self.portfolio.equity_from_row(
                raw_row, self.pair_col_a, self.pair_col_b)
            interest = self.portfolio.margin.apply_interest(
                bar_idx, eq_for_interest)
            if interest > 0:
                self.portfolio._total_costs += interest
                self._interest_total += interest

        if (bar_idx >= self.screen_lookback
                and bar_idx - self.last_refit >= self.refit_interval):
            self.last_refit = bar_idx
            self._refit(bar_idx, timestamp, log_row)

        for pid in list(self.portfolio.open_positions.keys()):
            ca = self.pair_col_a.get(pid)
            cb = self.pair_col_b.get(pid)
            if (ca is None or cb is None
                    or np.isnan(raw_row[ca]) or np.isnan(raw_row[cb])
                    or raw_row[ca] <= 0 or raw_row[cb] <= 0):
                self._close_position(pid, bar_idx, "forced")
                traded = True

        signals = self.kb.update(self.active_sids, log_row)

        for sid, (sv, z, spread_std, P11) in signals.items():
            meta = self.pair_meta.get(sid)
            if meta is None:
                continue

            if sid in self.spread_history:
                self.spread_history[sid]["timestamps"].append(timestamp)
                self.spread_history[sid]["spreads"].append(sv)
                self.spread_history[sid]["z_scores"].append(z)
                self.spread_history[sid]["bar_indices"].append(bar_idx)
                self.spread_history[sid]["betas"].append(
                    self.kb.get_beta(sid))
                ca = self.pair_col_a.get(sid)
                cb = self.pair_col_b.get(sid)
                if ca is not None and cb is not None:
                    va = self.vol_tracker.get_vol(ca)
                    vb = self.vol_tracker.get_vol(cb)
                    rho = implied_correlation(self.kb.get_beta(sid), va, vb)
                    self.spread_history[sid]["implied_rhos"].append(
                        rho if np.isfinite(rho) else np.nan)
                    self.spread_history[sid]["prices_a"].append(raw_row[ca])
                    self.spread_history[sid]["prices_b"].append(raw_row[cb])
                else:
                    self.spread_history[sid]["implied_rhos"].append(np.nan)
                    self.spread_history[sid]["prices_a"].append(np.nan)
                    self.spread_history[sid]["prices_b"].append(np.nan)

            if sid in self.portfolio.open_positions:
                pos = self.portfolio.open_positions[sid]
                pa, pb = self._get_raw_prices(sid, bar_idx)
                if pa is None:
                    continue

                pnl_pct = pos.pnl_pct(pa, pb)
                bars_held = bar_idx - pos.entry_bar

                hl = pos.half_life if pos.half_life > 0 else 96.0
                pair_min_hold = max(_MIN_HOLD_BARS, int(_HL_MIN_MULT * hl))
                pair_max_hold = int(self.hl_max_mult * hl)

                if abs(z) >= self.z_hard_stop:
                    self._close_position(sid, bar_idx, "z_stop", z, sv)
                    traded = True
                    continue

                if pnl_pct <= -self.pct_stop:
                    self._close_position(sid, bar_idx, "stop_loss", z, sv)
                    traded = True
                    continue

                if pair_max_hold > 0 and bars_held >= pair_max_hold:
                    self._close_position(sid, bar_idx, "max_hold", z, sv)
                    traded = True
                    continue

                drift = self._check_beta_drift(sid, bar_idx)

                if drift >= _BETA_BREAK_PCT:
                    self._close_position(sid, bar_idx, "beta_break", z, sv)
                    self._beta_break_cooldown[sid] = bar_idx
                    self._beta_breaks += 1
                    traded = True
                    continue

                if drift >= self.rebalance_beta_drift:
                    pa_raw, pb_raw = self._get_raw_prices(sid, bar_idx)
                    if pa_raw is not None:
                        ca = self.pair_col_a.get(sid)
                        cb = self.pair_col_b.get(sid)
                        if ca is not None and cb is not None:
                            new_beta = self.kb.get_beta(sid)
                            vol_a = self.vol_tracker.get_vol(ca)
                            vol_b = self.vol_tracker.get_vol(cb)
                            self.portfolio.rebalance_spread(
                                sid, bar_idx, pa_raw, pb_raw,
                                new_beta, vol_a, vol_b,
                                self.entry_cost_rate)
                            self._last_rebalance[sid] = bar_idx
                            self._rebalances += 1

                if bars_held < pair_min_hold:
                    continue

                if bars_held >= hl:
                    if abs(pos.entry_z) > 1e-6:
                        z_progress = 1.0 - (abs(z) / abs(pos.entry_z))
                    else:
                        z_progress = 0.0
                    if z_progress < self.stale_z_progress:
                        self._close_position(sid, bar_idx, "stale", z, sv)
                        self._stale_exits += 1
                        traded = True
                        continue

                if pos.direction > 0 and z >= -self.exit_z:
                    self._close_position(sid, bar_idx, "signal", z, sv)
                    traded = True
                    continue
                if pos.direction < 0 and z <= self.exit_z:
                    self._close_position(sid, bar_idx, "signal", z, sv)
                    traded = True
                    continue

            elif sid in self.active_sids:
                if len(self.portfolio.open_positions) >= self.max_open:
                    continue

                if abs(z) < self.entry_z or abs(z) >= self.z_hard_stop:
                    continue

                last_stop = self._pair_cooldown.get(sid, -999999)
                if bar_idx - last_stop < _PAIR_COOLDOWN_BARS:
                    self._cooldown_rejected += 1
                    continue

                last_beta_break = self._beta_break_cooldown.get(sid, -999999)
                if bar_idx - last_beta_break < _BETA_BREAK_COOLDOWN:
                    self._cooldown_rejected += 1
                    continue

                bh = self._beta_history.get(sid, [])
                if len(bh) >= 3:
                    bh_arr = np.array(bh)
                    bh_mean = np.mean(np.abs(bh_arr))
                    if bh_mean > 1e-6:
                        bh_cv = np.std(bh_arr) / bh_mean
                        if bh_cv > _BETA_MAX_CV:
                            self._beta_stability_rejected += 1
                            continue

                ca = self.pair_col_a.get(sid)
                cb = self.pair_col_b.get(sid)
                if ca is None or cb is None:
                    continue

                pa_raw, pb_raw = self._get_raw_prices(sid, bar_idx)
                if pa_raw is None:
                    continue

                beta = self.kb.get_beta(sid)
                direction = -1 if z > 0 else 1

                sector = meta["sector"]
                sec_counts = self.portfolio.sector_count()
                if sec_counts.get(sector, 0) >= self.max_sector_pairs:
                    continue

                eq = self.portfolio.equity_from_row(
                    raw_row, self.pair_col_a, self.pair_col_b)

                eff_pct = self.position_pct
                confidence = 1.0
                if abs(beta) > 1e-8:
                    beta_unc = np.sqrt(max(P11, 0.0))
                    rel_unc = beta_unc / abs(beta)
                    confidence = float(np.clip(1.0 - rel_unc, 0.2, 1.0))
                    if confidence >= 0.6:
                        confidence_sized = 1.0
                    else:
                        confidence_sized = (confidence / 0.6) ** self.confidence_power
                    eff_pct *= confidence_sized

                notional = eq * eff_pct

                if self._margin_config.is_leveraged:
                    notional *= self.max_leverage
                    can_open, max_notional = self.portfolio.margin.can_open(
                        notional, eq)
                    if not can_open or max_notional < self.vc.min_trade_usd * 2:
                        self._margin_capped_entries += 1
                        continue
                    if max_notional < notional:
                        self._margin_capped_entries += 1
                        notional = max_notional

                if notional < self.vc.min_trade_usd * 2:
                    continue

                vol_a = self.vol_tracker.get_vol(ca)
                vol_b = self.vol_tracker.get_vol(cb)

                implied_rho = implied_correlation(beta, vol_a, vol_b)
                regime_conf = 1.0

                if np.isfinite(implied_rho):
                    regime_conf = regime_confidence(
                        implied_rho, rho_high=self.rho_high,
                        rho_low=self.rho_low)
                    if regime_conf <= 0.0:
                        self._regime_rejected += 1
                        continue
                    self._regime_passed += 1

                    notional *= regime_conf
                    if notional < self.vc.min_trade_usd * 2:
                        continue

                quality = confidence * regime_conf

                notional_a, notional_b = vol_adjusted_notionals(
                    notional, beta, vol_a, vol_b)

                pos = self.portfolio.open_spread(
                    pair_id=sid,
                    sym_a=meta["sym_a"], sym_b=meta["sym_b"],
                    sector=meta["sector"],
                    direction=direction, beta=beta,
                    bar_idx=bar_idx,
                    price_a=pa_raw, price_b=pb_raw,
                    z_score=z,
                    notional_a=notional_a, notional_b=notional_b,
                    total_notional=notional,
                    cost_rate=self.entry_cost_rate,
                    half_life=meta.get("half_life", 0.0),
                )
                if pos is not None:
                    pos.entry_spread = sv
                    pos.entry_beta_P11 = P11
                    pos.entry_innov_S = spread_std
                    pos.entry_confidence = confidence
                    pos.entry_equity = eq
                    pos.entry_implied_rho = implied_rho
                    pos.entry_regime_conf = regime_conf
                    pos.entry_adaptive_z = self.entry_z
                    pos.entry_quality = quality
                    pos.entry_z_peak = self.kb.get_z_peak(sid)
                    traded = True

        if (self._margin_config.is_leveraged
                and self.portfolio.open_positions):
            eq_liq = self.portfolio.equity_from_row(
                raw_row, self.pair_col_a, self.pair_col_b)
            if self.portfolio.margin.check_liquidation(eq_liq):
                self.portfolio.force_liquidation(
                    bar_idx, timestamp, raw_row,
                    self.pair_col_a, self.pair_col_b,
                    self.exit_cost_rate)
                self._liquidations += 1
                traded = True

        return traded


def run_pairs(
    data: dict,
    sectors: dict[str, list[str]],
    venue: str = "binance",
    initial_capital: float = 100_000,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Optional[str] = None,
    btc_symbol: str = "BTC",
    price_col: str = "close_ohlcv",
    use_log_prices: bool = True,
    # all tunable params forwarded to PairsEngine
    **kwargs,
) -> BacktestResults:
    """Run Kalman pairs trading strategy with look-ahead-proof bar-by-bar loop."""
    all_coins = sorted(set(c for coins in sectors.values() for c in coins))

    feed = DataFeed(
        data, venue, all_coins,
        price_col=price_col,
        start_date=start_date, end_date=end_date,
        use_log_prices=use_log_prices,
    )

    engine = PairsEngine(
        feed=feed, sectors=sectors, venue=venue,
        initial_capital=initial_capital,
        **kwargs,
    )

    config = {
        "venue": venue, "initial_capital": initial_capital,
        "start_date": start_date, "end_date": end_date,
        "sectors": dict(sectors),
        "n_coins": feed.n_coins, "n_bars": feed.n_bars,
    }
    config.update(kwargs)

    leverage_active = engine._margin_config.is_leveraged

    n_bars = feed.n_bars

    lev_str = (f" | leverage={engine.max_leverage:.1f}x"
               if leverage_active else "")
    print(f"\n  Running: {n_bars} bars, {feed.n_coins} coins | "
          f"venue={venue} | capital=${initial_capital:,.0f}{lev_str}")

    t0 = _time.perf_counter()

    for bar_idx, timestamp, log_row in feed:
        if bar_idx == 0:
            engine.window.push(log_row)
            valid = np.isfinite(log_row)
            engine.last_known[valid] = log_row[valid]
            engine.vol_tracker.update(feed.raw_prices_row(bar_idx))
            continue

        traded = engine.on_bar(bar_idx, timestamp, log_row)

        raw_row = feed.raw_prices_row(bar_idx)
        do_snap = (traded
                   or bar_idx % _SNAPSHOT_EVERY == 0
                   or bar_idx >= n_bars - 2)
        if do_snap:
            engine.portfolio.snapshot(
                bar_idx, timestamp, raw_row,
                engine.pair_col_a, engine.pair_col_b)

        if bar_idx % _LOG_INTERVAL == 0 and bar_idx > engine.screen_lookback:
            eq = engine.portfolio.equity_from_row(
                raw_row, engine.pair_col_a, engine.pair_col_b)
            n_open = len(engine.portfolio.open_positions)
            elapsed = _time.perf_counter() - t0
            bps = bar_idx / max(elapsed, 0.001)
            pct = bar_idx / n_bars * 100
            ret = (eq / initial_capital - 1) * 100

            regime_total = engine._regime_passed + engine._regime_rejected
            regime_pct = (engine._regime_rejected / regime_total * 100
                          if regime_total > 0 else 0)

            print(f"  [{timestamp.date() if hasattr(timestamp, 'date') else timestamp}] "
                  f"{bar_idx}/{n_bars} ({pct:.0f}%) | "
                  f"eq: ${eq:,.0f} ({ret:+.1f}%) | "
                  f"open: {n_open} | "
                  f"regime: {engine._regime_rejected}/{regime_total} ({regime_pct:.0f}%) | "
                  f"rebal: {engine._rebalances} | "
                  f"beta_breaks: {engine._beta_breaks} | "
                  f"{bps:.0f} bars/s")

    for pid in list(engine.portfolio.open_positions.keys()):
        engine._close_position(pid, n_bars - 1, "end")

    elapsed = _time.perf_counter() - t0
    regime_total = engine._regime_passed + engine._regime_rejected

    print(f"\n  Done: {n_bars} bars in {elapsed:.1f}s "
          f"({n_bars / max(elapsed, .001):.0f} bars/s)")
    if regime_total > 0:
        print(f"  Regime filter: {engine._regime_rejected}/{regime_total} "
              f"({engine._regime_rejected / regime_total * 100:.0f}%) rejected")
    print(f"  Rebalances: {engine._rebalances} | "
          f"Beta breaks: {engine._beta_breaks}")
    if engine._cooldown_rejected > 0:
        print(f"  Cooldown rejected: {engine._cooldown_rejected}")
    if engine._beta_stability_rejected > 0:
        print(f"  Beta stability rejected: {engine._beta_stability_rejected}")
    if engine._stale_exits > 0:
        print(f"  Stale trade exits: {engine._stale_exits}")
    if leverage_active:
        print(f"  Leverage: {engine.max_leverage:.1f}x | "
              f"Liquidations: {engine._liquidations} | "
              f"Interest paid: ${engine._interest_total:,.2f} | "
              f"Margin-capped entries: {engine._margin_capped_entries}")

    eq_df = engine.portfolio.equity_df()
    tr_df = engine.portfolio.trade_log_df()
    metrics = compute_all_metrics(eq_df, tr_df)
    if leverage_active:
        metrics.update(engine.portfolio.margin.summary_fields())
    pair_stats = _build_pair_stats(engine.portfolio.completed_trades,
                                   pair_meta=engine.pair_meta)

    btc_col = engine.S2C.get(btc_symbol)
    btc_series = None
    if btc_col is not None:
        snap_bars = eq_df["bar"].values if "bar" in eq_df.columns else None
        if snap_bars is not None and len(snap_bars) > 0:
            btc_at_snaps = np.array([
                feed.raw_price_at(int(b), btc_col) for b in snap_bars])
            btc_series = pd.Series(
                btc_at_snaps, index=eq_df["timestamp"].values, name="BTC")

    result = BacktestResults(
        equity_curve=eq_df, trade_log=tr_df, metrics=metrics,
        pair_stats=pair_stats, spread_data=engine.spread_history,
        config=config)
    result.summary()

    if pair_stats is not None and len(pair_stats) > 0:
        _print_pair_rankings(pair_stats)

    if tr_df is not None and len(tr_df) > 10:
        _print_win_quality(tr_df)

    if output_dir is not None:
        from .reporting import generate_reports
        generate_reports(
            result, btc_series=btc_series,
            spread_data=engine.spread_history,
            output_dir=output_dir,
            bars_per_day=_BARS_PER_DAY)

    return result


def _build_pair_stats(trades: list[CompletedTrade],
                      pair_meta: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """Aggregate completed trades by pair."""
    if not trades:
        return None

    by_pair = defaultdict(list)
    for t in trades:
        by_pair[t.pair_id].append(t)

    rows = []
    for pid, pair_trades in by_pair.items():
        pnls = np.array([t.net_pnl for t in pair_trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        avg_hold = np.mean([t.bars_held for t in pair_trades])
        bars_per_year = _BARS_PER_DAY * 365
        trades_per_year = bars_per_year / max(avg_hold, 1)
        if len(pnls) >= 3 and np.std(pnls) > 1e-10:
            sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(trades_per_year)
        else:
            sharpe = 0.0

        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - peak
        max_dd = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        total_pnl = float(pnls.sum())
        ann_return = total_pnl * (trades_per_year / max(len(pnls), 1))
        calmar = ann_return / max_dd if max_dd > 1e-10 else 0.0

        total_notional = sum(t.notional for t in pair_trades)
        total_cost = sum(t.total_cost for t in pair_trades)

        meta = pair_meta.get(pid, {}) if pair_meta else {}

        rows.append({
            "pair_id": pid,
            "sym_a": pair_trades[0].sym_a,
            "sym_b": pair_trades[0].sym_b,
            "sector": pair_trades[0].sector,
            "composite_score": meta.get("composite_score", 0.0),
            "return_corr": meta.get("return_corr", 0.0),
            "n_trades": len(pnls),
            "total_pnl": round(total_pnl, 2),
            "sharpe": round(sharpe, 2),
            "calmar": round(calmar, 2),
            "win_rate": round(len(wins) / len(pnls), 3) if len(pnls) > 0 else 0,
            "avg_win": round(float(wins.mean()), 2) if len(wins) > 0 else 0,
            "avg_loss": round(float(losses.mean()), 2) if len(losses) > 0 else 0,
            "profit_factor": round(abs(wins.sum() / losses.sum()), 2)
                if len(losses) > 0 and losses.sum() != 0 else float("inf"),
            "avg_bars_held": round(float(avg_hold), 1),
            "avg_half_life": round(float(np.mean([
                t.half_life for t in pair_trades])), 1),
            "total_cost": round(total_cost, 2),
            "cost_pct": round(total_cost / total_notional * 100, 2)
                if total_notional > 0 else 0,
        })

    df = pd.DataFrame(rows).sort_values("total_pnl", ascending=False)
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"
    return df


def _print_pair_rankings(pair_stats: pd.DataFrame, n: int = 10):
    has_score = "composite_score" in pair_stats.columns
    print(f"\n{'='*100}")
    print(f"  TOP {min(n, len(pair_stats))} PAIRS (by P/L)")
    print(f"{'='*100}")
    hdr = (f"  {'Pair':30s} {'PnL':>10s} {'Sharpe':>8s} {'Calmar':>8s} {'WR':>6s} "
           f"{'Trades':>7s} {'PF':>6s} {'AvgHold':>8s}")
    if has_score:
        hdr += f" {'Score':>6s} {'Corr':>5s}"
    print(hdr)
    print(f"  {'-'*95}")
    for _, r in pair_stats.head(n).iterrows():
        pair_name = f"{r['sym_a']} / {r['sym_b']}"
        pf = f"{r['profit_factor']:.1f}" if r['profit_factor'] < 100 else "inf"
        line = (f"  {pair_name:30s} ${r['total_pnl']:>9,.0f} {r['sharpe']:>8.2f} "
                f"{r['calmar']:>8.2f} {r['win_rate']:>5.0%} "
                f"{r['n_trades']:>7d} {pf:>6s} {r['avg_bars_held']:>7.0f}b")
        if has_score:
            line += f" {r.get('composite_score', 0):>6.3f} {r.get('return_corr', 0):>5.2f}"
        print(line)

    if len(pair_stats) > n:
        print(f"\n{'='*100}")
        print(f"  BOTTOM {min(n, len(pair_stats))} PAIRS (by P/L)")
        print(f"{'='*100}")
        print(hdr)
        print(f"  {'-'*95}")
        for _, r in pair_stats.tail(n).iterrows():
            pair_name = f"{r['sym_a']} / {r['sym_b']}"
            pf = f"{r['profit_factor']:.1f}" if r['profit_factor'] < 100 else "inf"
            line = (f"  {pair_name:30s} ${r['total_pnl']:>9,.0f} {r['sharpe']:>8.2f} "
                    f"{r['calmar']:>8.2f} {r['win_rate']:>5.0%} "
                    f"{r['n_trades']:>7d} {pf:>6s} {r['avg_bars_held']:>7.0f}b")
            if has_score:
                line += f" {r.get('composite_score', 0):>6.3f} {r.get('return_corr', 0):>5.2f}"
            print(line)
    print()


def _print_win_quality(trade_log: pd.DataFrame):
    df = trade_log.copy()
    for col in ["entry_z", "exit_z", "gross_pnl", "total_cost",
                "net_pnl", "pnl_pct", "notional", "bars_held",
                "entry_adaptive_z", "entry_z_peak", "entry_confidence",
                "entry_quality"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sig = df[df["exit_reason"] == "signal"]
    if len(sig) < 5:
        return

    sig_wins = sig[sig["net_pnl"] > 0]
    sig_losses = sig[sig["net_pnl"] <= 0]
    non_sig = df[df["exit_reason"] != "signal"]

    abs_entry_z = sig["entry_z"].abs()
    abs_exit_z = sig["exit_z"].abs()
    z_travel = abs_entry_z - abs_exit_z

    gross_wins = sig[sig["gross_pnl"] > 0]["gross_pnl"].sum()
    total_cost_sig = sig["total_cost"].sum()
    cost_drag = total_cost_sig / gross_wins * 100 if gross_wins > 0 else 0

    non_sig_loss = non_sig["net_pnl"].sum()
    sig_total = sig["net_pnl"].sum()
    net_total = df["net_pnl"].sum()

    print(f"\n{'='*80}")
    print(f"  WIN QUALITY DIAGNOSTIC")
    print(f"{'='*80}")

    print(f"\n  SIGNAL EXITS ({len(sig)} trades, {len(sig_wins)} wins, {len(sig_losses)} losses)")
    print(f"  {'─'*60}")
    print(f"  Median entry |z|:    {abs_entry_z.median():.2f}")
    print(f"  Median exit |z|:     {abs_exit_z.median():.2f}")
    print(f"  Median z-travel:     {z_travel.median():.2f}σ")
    print(f"  Mean z-travel:       {z_travel.mean():.2f}σ")
    print()
    print(f"  Median win:          ${sig_wins['net_pnl'].median():.1f}" if len(sig_wins) > 0 else "")
    print(f"  Mean win:            ${sig_wins['net_pnl'].mean():.1f}" if len(sig_wins) > 0 else "")
    print(f"  Median loss:         ${sig_losses['net_pnl'].median():.1f}" if len(sig_losses) > 0 else "")
    print(f"  Median notional:     ${sig['notional'].median():,.0f}")
    print(f"  Median P/L %:        {sig['pnl_pct'].median()*100:.3f}%")
    print()
    print(f"  Gross wins (signal): ${gross_wins:,.0f}")
    print(f"  Costs (signal):      ${total_cost_sig:,.0f}")
    print(f"  Cost drag:           {cost_drag:.0f}% of gross wins")
    print(f"  Net signal P/L:      ${sig_total:,.0f}")

    print(f"\n  NON-SIGNAL EXITS ({len(non_sig)} trades)")
    print(f"  {'─'*60}")
    for reason in sorted(non_sig["exit_reason"].unique()):
        sub = non_sig[non_sig["exit_reason"] == reason]
        print(f"  {reason:15s}  n={len(sub):>4d}  "
              f"total=${sub['net_pnl'].sum():>9,.0f}  "
              f"avg=${sub['net_pnl'].mean():>7,.0f}")
    print(f"  {'─'*60}")
    print(f"  Non-signal total:    ${non_sig_loss:,.0f}")
    print(f"  NET P/L:             ${net_total:,.0f}")

    print(f"\n  DIAGNOSIS")
    print(f"  {'─'*60}")
    issues = []
    if z_travel.median() < 0.5:
        issues.append(f"  ⚠ Z-travel too small ({z_travel.median():.2f}σ). "
                      f"Entry too shallow or exit too early.")
    if cost_drag > 50:
        issues.append(f"  ⚠ Cost drag {cost_drag:.0f}%. "
                      f"Costs eating >{cost_drag:.0f}% of gross wins.")
    if abs_entry_z.median() < 1.3:
        issues.append(f"  ⚠ Median entry |z|={abs_entry_z.median():.2f}. "
                      f"Entering too close to the mean.")
    if len(sig_wins) > 0 and sig_wins["net_pnl"].median() < 10:
        issues.append(f"  ⚠ Median win ${sig_wins['net_pnl'].median():.0f}. "
                      f"Not enough spread per trade.")
    if abs(non_sig_loss) > sig_total and sig_total > 0:
        issues.append(f"  ⚠ Non-signal losses (${non_sig_loss:,.0f}) exceed "
                      f"signal gains (${sig_total:,.0f}).")
    if not issues:
        issues.append("  ✓ No obvious issues detected.")
    for issue in issues:
        print(issue)
    print()
