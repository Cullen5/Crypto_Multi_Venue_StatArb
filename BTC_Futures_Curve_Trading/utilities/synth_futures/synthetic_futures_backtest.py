"""
synthetic_futures_backtest.py — Synthetic Futures from Perp Funding
====================================================================
Strategy C: Cross-Venue Perpetual Funding Arbitrage
Per case study section 3.2.

Trade logic (funding rate differential):
  The strategy exploits persistent funding rate differentials across
  perpetual venues. When one venue's funding is significantly higher
  than another's, we:

  Long funding spread (Binance funding > Hyperliquid/dYdX funding):
    → Short Binance perp (RECEIVE high funding)
    → Long Hyperliquid/dYdX perp (PAY low funding)
    → Delta-neutral: BTC exposure cancels out
    → Collect net funding differential every 8h

  Short funding spread (Hyperliquid/dYdX funding > Binance funding):
    → Long Binance perp (pay low funding)
    → Short Hyperliquid/dYdX perp (receive high funding)
    → Same mechanics, reversed

Enhancements on top of base case study spec:
  1. Rolling z-score on funding spread for timing
  2. Regime-aware sizing: scale by spread persistence
  3. Multi-venue: scan all available venue pairs
  4. Costs via costs.py venue model (including gas for on-chain venues)

P&L (delta-neutral funding arb):
  Net P&L = Σ (funding_received - funding_paid) × notional - costs
  Per period: notional × (short_venue_funding - long_venue_funding)

Usage:
    from data_loader import DataLoader
    from synthetic_futures_backtest import run_synthetic_futures, SyntheticConfig

    dl = DataLoader("./data/raw")
    result = run_synthetic_futures(dl)
    result.summary()
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from utilities.options_calendar_spread.costs import get_venue_costs, VenueCosts


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

@dataclass
class SyntheticConfig:
    """
    Hyperparameters for Strategy C: Synthetic Futures from Perp Funding.
    Per case study section 3.2 Strategy C.

    Case study spec:
      - Short high-funding perp, long low-funding perp, collect differential
      - Entry: when funding differential is meaningful (covers costs)
      - Exit: spread converges or flips
      - Risk: max 2x leverage, basis stop at 5%, 50% margin cushion
      - Venue limits: Binance 50%, Hyperliquid 15%, dYdX 5%
      - Costs: Binance maker 0.01%/taker 0.04%, dYdX maker 0%/taker 0.05%
    """

    # ── Capital ──
    initial_capital: float = 100_000

    # ── Walk-forward ──
    train_end: str = "2023-06-30"
    test_start: str = "2023-07-01"

    # ── Venue pairs ──
    # binance-deribit DROPPED: only 1.3% ann spread (less than RT costs).
    # hyperliquid excluded: no historical perp price data.
    # The real opportunity is dYdX (21% ann funding) vs Binance/Deribit (6-8%).
    venue_pairs: tuple = (
        ("binance", "dydx"),
        ("deribit", "dydx"),
        ("deribit", "binance"),
    )

    # ── Funding resampling ──
    resample_freq: str = "8h"
    funding_lookback: int = 7      # days for rolling avg funding (smoothing)

    # ═════════════════════════════════════
    #  ENTRY / EXIT — calibrated to diagnostic data
    # ═════════════════════════════════════
    # Mean spreads: binance-dydx 9.3%, deribit-dydx 9.8%.
    # >60% of days have |spread| > 8%.
    # RT costs ~0.25%. At 8% ann spread = 0.022%/day, break-even ~11 days.
    # Entry at 8% is selective enough to avoid noise, loose enough to
    # capture the persistent dYdX premium.
    spread_entry_ann_pct: float = 8.0
    # Exit only when spread is truly exhausted
    spread_exit_ann_pct: float = 1.0
    # Exit if spread flips sign (now paying instead of receiving)
    spread_flip_exit: bool = True

    # ── Z-score (exit safety valve only) ──
    z_lookback: int = 60
    z_min_lookback: int = 20
    z_entry_boost: float = 0.0     # not used for entry
    z_full_size: float = 2.5
    z_stop: float = -2.5           # only exit on extreme z reversal

    # ── Position management ──
    base_position_pct: float = 0.50      # deploy up to venue limit per pair
    max_positions: int = 2               # 2 pairs (both use dydx leg)
    max_hold_days: int = 120             # funding arbs can persist
    # Case study: "Exit if basis moves against position by >5%"
    basis_stop_pct: float = -5.0
    stop_loss_cum_funding_pct: float = -5.0
    min_trade_usd: float = 5_000

    # Minimum raw 8h spread (anti-noise)
    spread_min_raw: float = 0.0001       # ~0.01% per 8h ≈ ~4.6% ann

    # ── Regime sizing multipliers ──
    regime_size_steep_contango: float = 1.0
    regime_size_mild_contango: float = 1.0
    regime_size_flat: float = 0.60
    regime_size_backwardation: float = 0.30

    # ── Venue exposure limits ──
    max_exposure_pct: dict = None

    # ── Re-entry cooldown ──
    cooldown_periods: int = 3      # 1 day cooldown (3 × 8h)

    def __post_init__(self):
        if self.max_exposure_pct is None:
            # With only 2 pairs both using dydx, dydx limit is the binding
            # constraint. At 25% per pair → up to 50% total dydx exposure
            # when both active. At ~10% net funding on deployed capital
            # this targets ~5% portfolio return (beats rf=4%).
            self.max_exposure_pct = {
                "binance": 0.50,
                "hyperliquid": 0.15,
                "dydx": 0.25,
                "deribit": 0.30,
            }

    def venue_costs(self, venue: str) -> VenueCosts:
        return get_venue_costs(venue)

    @property
    def round_trip_cost_rate(self) -> dict:
        """Round-trip cost per venue pair (entry + exit on both legs)."""
        costs = {}
        for short_v, long_v in self.venue_pairs:
            svc = self.venue_costs(short_v)
            lvc = self.venue_costs(long_v)
            # Entry: taker on both legs; Exit: maker on both legs
            entry = (svc.taker_fee + svc.avg_slippage) + (lvc.taker_fee + lvc.avg_slippage)
            exit_ = (svc.maker_fee + svc.avg_slippage) + (lvc.maker_fee + lvc.avg_slippage)
            gas = (svc.gas_cost_usd + lvc.gas_cost_usd) * 2  # entry + exit
            costs[(short_v, long_v)] = {
                "entry_rate": entry,
                "exit_rate": exit_,
                "round_trip_rate": entry + exit_,
                "gas_usd": gas,
            }
        return costs

    def regime_multiplier(self, regime: str) -> float:
        return {
            "steep_contango": self.regime_size_steep_contango,
            "mild_contango": self.regime_size_mild_contango,
            "flat": self.regime_size_flat,
            "backwardation": self.regime_size_backwardation,
        }.get(regime, 0.3)

    def to_dict(self) -> dict:
        d = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, tuple):
                v = list(v)
            d[k] = v
        d["round_trip_costs"] = {
            f"{a}-{b}": v for (a, b), v in self.round_trip_cost_rate.items()
        }
        return d


# ═══════════════════════════════════════════════════════════════
#  TRADE / POSITION MODELS
# ═══════════════════════════════════════════════════════════════

@dataclass
class Trade:
    trade_id: str
    venue_pair: str              # e.g. "binance-hyperliquid"
    short_venue: str             # venue where we short (receive funding)
    long_venue: str              # venue where we long (pay funding)
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    days_held: float
    direction: str               # "long_spread" or "short_spread"
    entry_spread_ann_pct: float  # annualised funding spread at entry
    exit_spread_ann_pct: float
    entry_z: float
    exit_z: float
    entry_spot: float
    exit_spot: float
    exit_reason: str
    regime_at_entry: str
    regime_multiplier: float
    z_size_scalar: float
    notional: float
    # Funding P&L
    cum_funding_received: float  # total funding received (short venue)
    cum_funding_paid: float      # total funding paid (long venue)
    gross_funding_pnl: float     # received - paid
    # Basis P&L (mark-to-market from price divergence between venues)
    basis_pnl: float
    gross_pnl: float             # funding + basis
    cost: float
    net_pnl: float
    pnl_pct: float
    avg_net_funding_ann_pct: float  # average annualised net funding collected

    def to_dict(self) -> dict:
        d = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, pd.Timestamp):
                v = str(v.date())
            elif isinstance(v, float):
                v = round(v, 4)
            d[k] = v
        return d


@dataclass
class OpenPosition:
    trade_id: str
    venue_pair: str
    short_venue: str
    long_venue: str
    entry_date: pd.Timestamp
    direction: str
    entry_spread_ann_pct: float
    entry_z: float
    entry_spot: float
    regime_at_entry: str
    regime_multiplier: float
    z_size_scalar: float
    notional: float
    # Tracking cumulative funding
    cum_funding_received: float = 0.0
    cum_funding_paid: float = 0.0
    # Track entry cost already deducted
    entry_cost: float = 0.0
    # Track cumulative basis P&L (mark-to-market from venue price divergence)
    cum_basis_pnl: float = 0.0
    # Per-venue perp close tracking for basis accrual
    last_short_close: float = 0.0
    last_long_close: float = 0.0

    def accrue_funding(self, short_funding: float, long_funding: float):
        """
        Accrue one period's funding.
        Short venue: we receive funding if positive, pay if negative.
        Long venue: we pay funding if positive, receive if negative.

        The net_funding_pnl property handles sign correctly: when
        short_funding is negative (we pay) it reduces cum_funding_received,
        and when long_funding is negative (we receive) it reduces
        cum_funding_paid. The difference still gives correct net P&L.
        """
        # Short position: if funding > 0, longs pay shorts → we RECEIVE
        self.cum_funding_received += self.notional * short_funding
        # Long position: if funding > 0, longs pay shorts → we PAY
        self.cum_funding_paid += self.notional * long_funding

    def accrue_basis(self, short_perp_close: float, long_perp_close: float,
                     prev_short_close: float, prev_long_close: float):
        """
        Compute actual basis P&L from per-venue perp price divergence.

        We are short venue A perp and long venue B perp. If both perps
        moved identically, basis P&L = 0 (delta-neutral). In practice
        they diverge, creating real P&L.

        Short leg P&L: notional × (prev_short - current_short) / prev_short
          (we're short, so profit when price falls)
        Long leg P&L:  notional × (current_long - prev_long) / prev_long
          (we're long, so profit when price rises)

        Net basis P&L = short_leg_pnl + long_leg_pnl
        If both move up by same %, these cancel to zero (delta-neutral).
        """
        if (prev_short_close > 0 and prev_long_close > 0
                and short_perp_close > 0 and long_perp_close > 0):
            short_ret = (short_perp_close - prev_short_close) / prev_short_close
            long_ret = (long_perp_close - prev_long_close) / prev_long_close
            # Short leg: we profit when short_venue price falls
            short_leg_pnl = -short_ret * self.notional
            # Long leg: we profit when long_venue price rises
            long_leg_pnl = long_ret * self.notional
            self.cum_basis_pnl += short_leg_pnl + long_leg_pnl

    @property
    def net_funding_pnl(self) -> float:
        return self.cum_funding_received - self.cum_funding_paid

    @property
    def cum_net_funding_pct(self) -> float:
        if self.notional == 0:
            return 0.0
        return self.net_funding_pnl / self.notional * 100


# ═══════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(
    equity_series: pd.Series,
    trades: list[Trade],
    spot_series: pd.Series = None,
) -> dict:
    if len(equity_series) < 2:
        return {"error": "insufficient data"}

    returns = equity_series.pct_change().dropna()
    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
    n_days = (equity_series.index[-1] - equity_series.index[0]).days
    ann_factor = 365.25 / max(n_days, 1)

    # ── Portfolio-level Sharpe from daily equity returns ──
    periods_per_year = len(returns) / max(n_days / 365.25, 1e-6)
    rf_per_period = 0.04 / periods_per_year

    excess = returns - rf_per_period
    sharpe = (float(excess.mean() / excess.std() * np.sqrt(periods_per_year))
              if excess.std() > 1e-10 else 0.0)

    # Sortino: downside deviation (uses same dynamic periods_per_year)
    downside_sq = np.minimum(excess.values, 0.0) ** 2
    downside_dev = np.sqrt(np.mean(downside_sq))
    sortino = (float(excess.mean() / downside_dev * np.sqrt(periods_per_year))
               if downside_dev > 1e-10 else 0.0)

    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min()
    ann_return = (1 + total_return) ** ann_factor - 1
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    # ── Statistical significance ──
    # Sharpe SE ≈ sqrt((1 + 0.5*SR²) / n_obs)
    n_obs = len(returns)
    n_trades = len(trades) if trades else 0
    sharpe_se = np.sqrt((1 + 0.5 * sharpe ** 2) / max(n_obs, 1))

    # ── Capital efficiency: return on deployed capital ──
    # total_notional_days uses calendar days (t.days_held), so 365.25 is
    # the correct annualiser here (not periods_per_year).
    if n_trades > 0:
        total_notional_days = sum(t.notional * max(t.days_held, 1)
                                  for t in trades)
        initial_equity = equity_series.iloc[0]
        total_equity_days = initial_equity * max(n_days, 1)
        capital_utilisation = (total_notional_days / total_equity_days
                               if total_equity_days > 0 else 0.0)
        # Return on deployed capital (annualised)
        total_pnl_val = sum(t.net_pnl for t in trades)
        if total_notional_days > 0:
            return_on_deployed_ann = (total_pnl_val / total_notional_days
                                      * 365.25 * 100)
        else:
            return_on_deployed_ann = 0.0
    else:
        capital_utilisation = 0.0
        return_on_deployed_ann = 0.0

    pnls = np.array([t.net_pnl for t in trades]) if trades else np.array([0.0])
    n_trades = len(trades)
    if n_trades > 0:
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        win_rate = len(wins) / n_trades
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        pf = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else np.inf
        total_cost = sum(t.cost for t in trades)
        total_funding_received = sum(t.cum_funding_received for t in trades)
        total_funding_paid = sum(t.cum_funding_paid for t in trades)
        total_basis_pnl = sum(t.basis_pnl for t in trades)
        avg_net_funding_ann = np.mean([t.avg_net_funding_ann_pct for t in trades])
    else:
        win_rate = avg_win = avg_loss = total_cost = 0.0
        total_funding_received = total_funding_paid = avg_net_funding_ann = 0.0
        total_basis_pnl = 0.0
        pf = 0.0

    regime_counts, regime_pnl = {}, {}
    venue_pair_pnl = {}
    for t in trades:
        r = t.regime_at_entry
        regime_counts[r] = regime_counts.get(r, 0) + 1
        regime_pnl[r] = regime_pnl.get(r, 0.0) + t.net_pnl
        vp = t.venue_pair
        venue_pair_pnl[vp] = venue_pair_pnl.get(vp, 0.0) + t.net_pnl

    # ── BTC correlation ──
    # Correlation of daily strategy returns vs BTC spot returns.
    btc_corr = np.nan
    if spot_series is not None and len(spot_series) > 5:
        spot_rets = spot_series.pct_change().dropna()
        # Align on common index
        common = returns.index.intersection(spot_rets.index)
        if len(common) > 5:
            btc_corr = float(returns.loc[common].corr(spot_rets.loc[common]))

    # ── Average funding cost (annualised, as % of avg notional deployed) ──
    # Uses 365.25 directly since total_notional_days is in calendar days.
    if n_trades > 0:
        total_notional_days = sum(t.notional * max(t.days_held, 1)
                                  for t in trades)
        avg_funding_cost_ann = (
            (total_funding_paid / total_notional_days) * 365.25 * 100
            if total_notional_days > 0 else 0.0
        )
        avg_funding_income_ann = (
            (total_funding_received / total_notional_days) * 365.25 * 100
            if total_notional_days > 0 else 0.0
        )
    else:
        avg_funding_cost_ann = avg_funding_income_ann = 0.0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(ann_return * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sharpe_se": round(float(sharpe_se), 3) if not np.isnan(sharpe_se) else "N/A",
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar_ratio": round(calmar, 3),
        "capital_utilisation_pct": round(capital_utilisation * 100, 1),
        "return_on_deployed_ann_pct": round(return_on_deployed_ann, 2),
        "stat_warning": (f"Low confidence: {n_trades} trades / {n_obs} daily obs. "
                         f"Sharpe 95% CI: [{sharpe - 1.96*sharpe_se:.1f}, "
                         f"{sharpe + 1.96*sharpe_se:.1f}]")
                        if n_trades < 30 else "",
        "btc_correlation": round(float(btc_corr), 3) if not np.isnan(btc_corr) else "N/A",
        "n_trades": n_trades,
        "win_rate": round(win_rate, 3),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(float(pf), 2) if np.isfinite(pf) else "inf",
        "total_pnl": round(float(pnls.sum()), 2),
        "total_cost": round(total_cost, 2),
        "cost_drag_pct": round(total_cost / max(equity_series.iloc[0], 1) * 100, 2),
        "total_funding_received": round(total_funding_received, 2),
        "total_funding_paid": round(total_funding_paid, 2),
        "net_funding_pnl": round(total_funding_received - total_funding_paid, 2),
        "total_basis_pnl": round(total_basis_pnl, 2),
        "avg_net_funding_ann_pct": round(avg_net_funding_ann, 2),
        "avg_funding_cost_ann_pct": round(avg_funding_cost_ann, 2),
        "avg_funding_income_ann_pct": round(avg_funding_income_ann, 2),
        "avg_days_held": round(np.mean([t.days_held for t in trades]), 1) if trades else 0,
        "n_days_traded": n_days,
        "regime_trades": regime_counts,
        "regime_pnl": {k: round(v, 2) for k, v in regime_pnl.items()},
        "venue_pair_pnl": {k: round(v, 2) for k, v in venue_pair_pnl.items()},
    }


def print_summary(metrics: dict, title: str = "Strategy C"):
    _pct_keys = {"total_return_pct", "annualized_return_pct", "max_drawdown_pct",
                 "cost_drag_pct", "avg_net_funding_ann_pct",
                 "capital_utilisation_pct", "return_on_deployed_ann_pct"}
    _dollar_keys = {"total_basis_pnl", "net_funding_pnl", "total_pnl",
                    "total_cost", "total_funding_received", "total_funding_paid"}
    _rate_keys = {"win_rate"}
    _skip_keys = {"stat_warning"}

    print(f"\n{'═' * 62}")
    print(f"  {title}")
    print(f"{'═' * 62}")
    for k, v in metrics.items():
        if k in _skip_keys:
            continue
        label = k.replace("_", " ").title()
        if isinstance(v, dict):
            print(f"  {label}:")
            for kk, vv in v.items():
                print(f"    {kk:24s} {str(vv):>10s}")
        elif k in _rate_keys and isinstance(v, (int, float)):
            print(f"  {label:32s} {v * 100:>11.1f}%")
        elif k in _pct_keys and isinstance(v, (int, float)):
            print(f"  {label:32s} {v:>11.2f}%")
        elif isinstance(v, float):
            print(f"  {label:32s} {v:>12.2f}")
        else:
            print(f"  {label:32s} {str(v):>12s}")

    # Print statistical warning at the end if present
    if metrics.get("stat_warning"):
        print(f"  {'─' * 58}")
        print(f"  ⚠ {metrics['stat_warning']}")

    print(f"{'═' * 62}\n")


# ═══════════════════════════════════════════════════════════════
#  RESULTS CONTAINER
# ═══════════════════════════════════════════════════════════════

@dataclass
class BacktestResults:
    strategy: str
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    metrics: dict
    config: dict
    daily_data: pd.DataFrame = None

    def summary(self):
        print_summary(self.metrics, title=self.strategy)

    def plot_equity(self, ax=None, benchmark_spot=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 5))
        eq = self.equity_curve.set_index("date")["equity"]
        ax.plot(eq.index, eq.values, label="Synthetic Futures Arb",
                linewidth=1.5, color="#2ca02c")
        if benchmark_spot is not None:
            bm = benchmark_spot / benchmark_spot.iloc[0] * eq.iloc[0]
            ax.plot(bm.index, bm.values, label="BTC Buy & Hold",
                    alpha=0.5, linewidth=1, color="#7f7f7f")
        ax.set_ylabel("Equity ($)")
        ax.set_title("Strategy C: Synthetic Futures from Perp Funding — Equity")
        ax.legend(); ax.grid(alpha=0.3)
        return ax

    def plot_drawdown(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 3))
        eq = self.equity_curve.set_index("date")["equity"]
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax.fill_between(dd.index, dd.values, 0, alpha=0.5, color="#d62728")
        ax.set_ylabel("Drawdown (%)"); ax.set_title("Strategy C — Drawdown")
        ax.grid(alpha=0.3)
        return ax

    def plot_funding_spread(self, ax=None):
        """Plot annualised funding spread with trade entry/exit shading."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 5))
        if self.daily_data is not None and "spread_ann_pct" in self.daily_data.columns:
            dd = self.daily_data
            for pair in dd["venue_pair"].unique():
                pdf = dd[dd["venue_pair"] == pair]
                ax.plot(pdf["date"], pdf["spread_ann_pct"], linewidth=0.8,
                        alpha=0.7, label=f"{pair} ann. spread (%)")
            ax.axhline(0, color="black", linewidth=0.5)
            _cfg = self.config
            ax.axhline(_cfg.get("spread_entry_ann_pct", 10), color="#2ca02c",
                        linestyle="--", alpha=0.5, label="Entry threshold")
            ax.axhline(-_cfg.get("spread_entry_ann_pct", 10), color="#2ca02c",
                        linestyle="--", alpha=0.5)
            ax.axhline(_cfg.get("spread_exit_ann_pct", 3), color="#ff7f0e",
                        linestyle=":", alpha=0.5, label="Exit threshold")
        if not self.trade_log.empty:
            for _, t in self.trade_log.iterrows():
                entry_d = pd.to_datetime(t["entry_date"])
                exit_d = pd.to_datetime(t["exit_date"])
                color = "#2ca02c" if t.get("net_pnl", 0) > 0 else "#d62728"
                ax.axvspan(entry_d, exit_d, alpha=0.12, color=color)
        ax.set_ylabel("Annualised Funding Spread (%)")
        ax.set_title("Strategy C — Funding Spread & Trades")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        return ax

    def plot_cumulative_funding(self, ax=None):
        """Plot cumulative net funding collected across all positions."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 4))
        # cum_net_funding lives in the equity_curve, not daily_data
        if self.equity_curve is not None and "cum_net_funding" in self.equity_curve.columns:
            dd = self.equity_curve.drop_duplicates("date").set_index("date")
            ax.plot(dd.index, dd["cum_net_funding"], linewidth=1.2,
                    color="#9467bd", label="Cum. Net Funding ($)")
            ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Cumulative Net Funding ($)")
        ax.set_title("Strategy C — Cumulative Funding Collected")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        return ax

    def plot_regime_pnl(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        regime_pnl = self.metrics.get("regime_pnl", {})
        if regime_pnl:
            regimes = list(regime_pnl.keys())
            pnls = [regime_pnl[r] for r in regimes]
            colors = ["#2ca02c" if p > 0 else "#d62728" for p in pnls]
            ax.bar(regimes, pnls, color=colors, alpha=0.7)
            ax.set_ylabel("Net P&L ($)"); ax.set_title("P&L by Regime")
            ax.grid(alpha=0.3, axis="y")
        return ax

    def plot_venue_pair_pnl(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        vp_pnl = self.metrics.get("venue_pair_pnl", {})
        if vp_pnl:
            pairs = list(vp_pnl.keys())
            pnls = [vp_pnl[p] for p in pairs]
            colors = ["#2ca02c" if p > 0 else "#d62728" for p in pnls]
            ax.bar(pairs, pnls, color=colors, alpha=0.7)
            ax.set_ylabel("Net P&L ($)"); ax.set_title("P&L by Venue Pair")
            ax.grid(alpha=0.3, axis="y")
        return ax


# ═══════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

def _classify_regime(ann_basis: float) -> str:
    if pd.isna(ann_basis):
        return "flat"
    if ann_basis > 20:
        return "steep_contango"
    elif ann_basis > 5:
        return "mild_contango"
    elif ann_basis > -5:
        return "flat"
    else:
        return "backwardation"


def _prepare_funding_data(
    dl, cfg: SyntheticConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the funding spread dataset for all venue pairs.

    CRITICAL: Only produces rows where BOTH venues in a pair have:
      1. Funding rate data
      2. Perpetual OHLCV price data (close)

    This means dYdX pairs only start when dYdX perp price data begins
    (~Nov 2023), and Hyperliquid is excluded entirely (no price data).

    Returns:
        spread_data:  DataFrame with daily signals, including per-venue
                      perp close prices for basis P&L calculation
        funding_periods:  Granular (8h) funding data for accrual, with
                          per-venue perp close prices interpolated
    """
    # ── Load aligned funding comparison ──
    funding_comp = dl.funding_comparison(resample=cfg.resample_freq)

    # ── Load per-venue perp close prices via DataLoader ──
    # Use dl.perpetual(venue) which handles path resolution + fallbacks
    venue_perp_prices = {}
    venue_is_spot_proxy = set()  # track which venues fell back to spot

    # Load spot as a universal fallback for venues without perp OHLCV
    spot_fallback_df = None
    try:
        _spot = dl.spot()
        spot_fallback_df = _spot[["datetime", "close"]].rename(
            columns={"close": "perp_close"})
    except Exception:
        pass

    for venue in ["binance", "deribit", "dydx", "hyperliquid"]:
        try:
            perp_df = dl.perpetual(venue)
            venue_perp_prices[venue] = perp_df[["datetime", "close"]].rename(
                columns={"close": "perp_close"})
            print(f"  {venue.capitalize()} perp prices: {len(perp_df):,} rows "
                  f"({perp_df['datetime'].min().date()} → "
                  f"{perp_df['datetime'].max().date()})")
        except (FileNotFoundError, KeyError, Exception) as e:
            if spot_fallback_df is not None:
                venue_perp_prices[venue] = spot_fallback_df.copy()
                venue_is_spot_proxy.add(venue)
                print(f"  {venue.capitalize()} perp prices: using SPOT as proxy "
                      f"({len(spot_fallback_df):,} rows) — {e}")
            else:
                print(f"  WARNING: No {venue.capitalize()} perp OHLCV and "
                      f"no spot fallback: {e}")

    if venue_is_spot_proxy:
        print(f"  ⚠ Venues using spot proxy: {venue_is_spot_proxy}")
        print(f"    Basis P&L between two spot-proxied legs will be ~zero.")
        print(f"    Real per-venue perp data needed for accurate basis tracking.")

    # ── Resample perp prices to daily close ──
    venue_daily_close = {}
    for venue, pdf in venue_perp_prices.items():
        pdf = pdf.copy()
        pdf["date"] = pdf["datetime"].dt.normalize().dt.tz_localize(None)
        daily = (pdf.sort_values("datetime")
                 .groupby("date")["perp_close"]
                 .last()
                 .reset_index()
                 .rename(columns={"perp_close": f"{venue}_perp_close"}))
        venue_daily_close[venue] = daily
        print(f"  {venue} daily perp close: {len(daily)} days")

    # ── Load spot for regime classification ──
    try:
        front = dl.continuous_front_month(margin_type="USDT-M")
        front["date"] = front["datetime"].dt.normalize().dt.tz_localize(None)
        regime_daily = (front.sort_values("datetime")
                        .groupby("date")
                        .agg(ann_basis=("annualized_basis_pct", "last"),
                             spot_close=("spot_close", "last"))
                        .reset_index())
    except Exception:
        regime_daily = pd.DataFrame(columns=["date", "ann_basis", "spot_close"])

    # ── Extend spot_close coverage beyond USDT-M contract range ──
    # USDT-M quarterly contracts may not cover the full strategy period
    # (e.g. last USDT-M expires Jun 2025 but funding data runs to Dec 2025).
    # Fill gaps from raw spot data so the benchmark and regime are complete.
    try:
        spot_df = dl.spot()
        spot_df["date"] = spot_df["datetime"].dt.normalize().dt.tz_localize(None)
        spot_daily_full = (spot_df.sort_values("datetime")
                           .groupby("date")
                           .agg(spot_close=("close", "last"))
                           .reset_index())
        if regime_daily.empty:
            regime_daily = spot_daily_full.copy()
            regime_daily["ann_basis"] = 0.0
        else:
            # Find dates missing from regime_daily
            existing_dates = set(regime_daily["date"])
            missing = spot_daily_full[~spot_daily_full["date"].isin(existing_dates)].copy()
            if len(missing) > 0:
                missing["ann_basis"] = 0.0  # no futures curve data → flat regime
                regime_daily = pd.concat([regime_daily, missing], ignore_index=True)
                regime_daily = regime_daily.sort_values("date").reset_index(drop=True)
                print(f"  Extended regime_daily with {len(missing)} days of"
                      f" spot-only data (no USDT-M contracts)")
    except Exception as e:
        print(f"  WARNING: Could not extend spot coverage: {e}")
        if regime_daily.empty:
            raise ValueError("No spot or front-month data available for regime")

    regime_daily["regime"] = regime_daily["ann_basis"].apply(_classify_regime)

    # ── Also resample perp prices to 8h for intraday basis accrual ──
    venue_8h_close = {}
    for venue, pdf in venue_perp_prices.items():
        pdf = pdf.copy()
        pdf = pdf.set_index("datetime").sort_index()
        resampled = pdf["perp_close"].resample("8h").last().dropna().reset_index()
        resampled = resampled.rename(columns={
            "perp_close": f"{venue}_perp_close"})
        venue_8h_close[venue] = resampled

    # ── Build spread series for each venue pair ──
    all_spreads = []
    all_periods = []

    for short_v, long_v in cfg.venue_pairs:
        if short_v not in funding_comp.columns or long_v not in funding_comp.columns:
            print(f"  WARNING: Skipping pair {short_v}-{long_v}, "
                  f"missing from funding data")
            continue

        if short_v not in venue_perp_prices or long_v not in venue_perp_prices:
            print(f"  WARNING: Skipping pair {short_v}-{long_v}, "
                  f"missing perp price data")
            continue

        pair_key = f"{short_v}-{long_v}"
        pdf = funding_comp[["datetime", short_v, long_v]].dropna().copy()
        pdf["date"] = pdf["datetime"].dt.normalize()
        if pdf["date"].dt.tz is not None:
            pdf["date"] = pdf["date"].dt.tz_localize(None)

        # ── Merge 8h perp prices for both venues ──
        for venue in [short_v, long_v]:
            v8h = venue_8h_close[venue].copy()
            v8h_col = f"{venue}_perp_close"
            # Ensure matching datetime resolution for merge_asof
            pdf["datetime"] = pdf["datetime"].astype("datetime64[ns, UTC]")
            v8h["datetime"] = v8h["datetime"].astype("datetime64[ns, UTC]")
            # Merge on nearest 8h timestamp
            pdf = pd.merge_asof(
                pdf.sort_values("datetime"),
                v8h.sort_values("datetime"),
                on="datetime",
                direction="nearest",
                tolerance=pd.Timedelta("4h"),
            )

        # ── Drop rows where either venue's perp price is missing ──
        short_col = f"{short_v}_perp_close"
        long_col = f"{long_v}_perp_close"
        before_count = len(pdf)
        pdf = pdf.dropna(subset=[short_col, long_col])
        after_count = len(pdf)
        if before_count > after_count:
            print(f"  {pair_key}: dropped {before_count - after_count} rows "
                  f"missing perp prices ({after_count} remaining)")

        if pdf.empty:
            print(f"  WARNING: {pair_key} has no overlapping price data, skipping")
            continue

        print(f"  {pair_key}: {len(pdf)} periods with full data "
              f"({pdf['datetime'].min().date()} → {pdf['datetime'].max().date()})")

        # Raw spread: short_venue_funding - long_venue_funding
        pdf["short_funding"] = pdf[short_v]
        pdf["long_funding"] = pdf[long_v]
        pdf["spread_raw"] = pdf[short_v] - pdf[long_v]

        # Annualise
        periods_per_year = (365.25 * 24) / 8  # = 1095.75
        pdf["spread_ann_pct"] = pdf["spread_raw"] * periods_per_year * 100

        # Adaptive rolling window: use available data length, min 3 days
        pair_data_len = len(pdf)
        effective_lookback = min(cfg.funding_lookback * 3, max(pair_data_len // 4, 9))

        pdf["spread_smooth"] = pdf["spread_raw"].rolling(
            effective_lookback, min_periods=max(effective_lookback // 3, 3)).mean()
        pdf["spread_smooth_ann_pct"] = pdf["spread_smooth"] * periods_per_year * 100

        # Z-score with adaptive window
        z_lb = min(cfg.z_lookback, max(pair_data_len // 3, cfg.z_min_lookback + 1))
        z_min = min(cfg.z_min_lookback, max(z_lb // 3, 5))
        pdf["spread_mean"] = pdf["spread_raw"].rolling(z_lb, min_periods=z_min).mean()
        pdf["spread_std"] = pdf["spread_raw"].rolling(z_lb, min_periods=z_min).std()
        pdf["spread_std"] = pdf["spread_std"].clip(lower=1e-6)
        pdf["z_score"] = (pdf["spread_raw"] - pdf["spread_mean"]) / pdf["spread_std"]

        pdf["venue_pair"] = pair_key
        pdf["short_venue"] = short_v
        pdf["long_venue"] = long_v

        # Store granular data for funding + basis accrual
        period_cols = [
            "datetime", "date", "venue_pair", "short_venue", "long_venue",
            "short_funding", "long_funding", "spread_raw", "spread_ann_pct",
            "spread_smooth_ann_pct", "z_score",
            short_col, long_col,
        ]
        all_periods.append(pdf[[c for c in period_cols if c in pdf.columns]].copy())

        # Daily aggregation for signal generation
        daily = (pdf.sort_values("datetime")
                 .groupby("date")
                 .agg(
                     spread_raw=("spread_raw", "mean"),
                     spread_ann_pct=("spread_smooth_ann_pct", "last"),
                     z_score=("z_score", "last"),
                     short_funding_sum=("short_funding", "sum"),
                     long_funding_sum=("long_funding", "sum"),
                     n_periods=("spread_raw", "count"),
                     **{short_col: (short_col, "last"),
                        long_col: (long_col, "last")},
                 )
                 .reset_index())

        daily["venue_pair"] = pair_key
        daily["short_venue"] = short_v
        daily["long_venue"] = long_v

        # Merge regime + spot
        daily = daily.merge(
            regime_daily[["date", "regime", "spot_close"]],
            on="date", how="left",
        )
        daily["regime"] = daily["regime"].fillna("flat")
        daily["spot_close"] = daily["spot_close"].ffill()

        all_spreads.append(daily)

    if not all_spreads:
        raise ValueError("No valid venue pairs found with both funding AND perp price data")

    spread_data = pd.concat(all_spreads, ignore_index=True).sort_values(
        ["venue_pair", "date"]).reset_index(drop=True)
    funding_periods = pd.concat(all_periods, ignore_index=True).sort_values(
        ["venue_pair", "datetime"]).reset_index(drop=True)

    return spread_data, funding_periods


# ═══════════════════════════════════════════════════════════════
#  MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════

def run_synthetic_futures(
    dl,
    cfg: SyntheticConfig = None,
    test_only: bool = True,
) -> BacktestResults:
    """
    Run the synthetic futures / funding arb backtest per Strategy C.

    Parameters
    ----------
    dl : DataLoader
    cfg : SyntheticConfig (defaults if None)
    test_only : if True, only trade in test period (train for z-score warmup)
    """
    if cfg is None:
        cfg = SyntheticConfig()

    print(f"\n{'═' * 62}")
    print(f"  STRATEGY C: Synthetic Futures from Perp Funding")
    print(f"{'═' * 62}")
    print(f"  Venue pairs: {[f'{a}-{b}' for a, b in cfg.venue_pairs]}")
    print(f"  Entry: |ann spread| > {cfg.spread_entry_ann_pct}%")
    print(f"  Exit:  |ann spread| < {cfg.spread_exit_ann_pct}%"
          f"{', flip exit' if cfg.spread_flip_exit else ''}")
    print(f"  Z-score: lookback={cfg.z_lookback}, "
          f"entry_boost={cfg.z_entry_boost}, z_stop={cfg.z_stop}")
    print(f"  Position: {cfg.base_position_pct:.0%} base × regime × z_scalar, "
          f"max {cfg.max_positions}")

    # Print costs per pair
    rt_costs = cfg.round_trip_cost_rate
    for pair_key, costs in rt_costs.items():
        gas = costs['gas_usd']
        gas_str = f", gas=${gas:.2f}" if gas > 0 else ""
        print(f"  Costs ({pair_key[0]}-{pair_key[1]}): "
              f"{costs['round_trip_rate']:.4%} RT{gas_str}")

    # ── Prepare data ──
    spread_data, funding_periods = _prepare_funding_data(dl, cfg)

    print(f"\n  Full range: {spread_data['date'].min().date()} → "
          f"{spread_data['date'].max().date()}")
    print(f"  Venue pairs: {spread_data['venue_pair'].nunique()}")

    # Filter to trading period
    if test_only:
        test_ts = pd.Timestamp(cfg.test_start)
        trade_mask = spread_data["date"] >= test_ts
    else:
        trade_mask = spread_data["z_score"].notna()

    trade_data = spread_data[trade_mask].copy().reset_index(drop=True)
    if trade_data.empty:
        raise ValueError("No trading data after filtering.")

    trade_dates = sorted(trade_data["date"].unique())
    print(f"  Trading: {trade_dates[0].date()} → "
          f"{trade_dates[-1].date()} ({len(trade_dates)} days)")

    # Regime distribution
    regime_counts = trade_data.groupby("regime")["date"].nunique()
    for regime, cnt in regime_counts.items():
        print(f"    {regime}: {cnt} days ({cnt / len(trade_dates):.0%})")

    # Days above entry threshold (any pair)
    for pair in trade_data["venue_pair"].unique():
        pdf = trade_data[trade_data["venue_pair"] == pair]
        n_above = (pdf["spread_ann_pct"].abs() > cfg.spread_entry_ann_pct).sum()
        print(f"  {pair}: {n_above} days |ann spread| > "
              f"{cfg.spread_entry_ann_pct}% ({n_above/len(pdf):.0%})")

    # ── Backtest loop (daily) ──
    equity = cfg.initial_capital
    positions: list[OpenPosition] = []
    completed: list[Trade] = []
    equity_hist = []
    trade_counter = 0
    cooldown_map: dict[str, pd.Timestamp] = {}   # venue_pair → last exit date
    cum_net_funding_total = 0.0

    for date in trade_dates:
        day_data = trade_data[trade_data["date"] == date]
        # Get funding periods for this date (for accrual)
        day_periods = funding_periods[funding_periods["date"] == date]

        # Get spot price for today (for basis accrual)
        if not day_data.empty:
            day_spot = day_data.iloc[0].get("spot_close", np.nan)
        else:
            day_spot = np.nan

        # ═══════════════════════════
        #  ACCRUE FUNDING on open positions
        # ═══════════════════════════
        for pos in positions:
            pair_periods = day_periods[day_periods["venue_pair"] == pos.venue_pair]
            for _, period in pair_periods.iterrows():
                if pos.direction == "long_spread":
                    # We are: short the short_venue, long the long_venue
                    # Short venue: receive funding if positive
                    pos.accrue_funding(period["short_funding"], period["long_funding"])
                else:
                    # Reversed: short the long_venue, long the short_venue
                    pos.accrue_funding(period["long_funding"], period["short_funding"])

            # Accrue basis P&L using actual per-venue perp close prices
            pair_periods_for_basis = day_periods[
                day_periods["venue_pair"] == pos.venue_pair]
            if not pair_periods_for_basis.empty:
                short_col = f"{pos.short_venue}_perp_close"
                long_col = f"{pos.long_venue}_perp_close"
                if short_col in pair_periods_for_basis.columns and \
                   long_col in pair_periods_for_basis.columns:
                    for _, bp in pair_periods_for_basis.iterrows():
                        cur_short = bp.get(short_col, np.nan)
                        cur_long = bp.get(long_col, np.nan)
                        prev_short = pos.last_short_close
                        prev_long = pos.last_long_close
                        if (not pd.isna(cur_short) and not pd.isna(cur_long)
                                and prev_short > 0 and prev_long > 0):
                            pos.accrue_basis(cur_short, cur_long,
                                             prev_short, prev_long)
                        # Update price tracking
                        if not pd.isna(cur_short) and cur_short > 0:
                            pos.last_short_close = cur_short
                        if not pd.isna(cur_long) and cur_long > 0:
                            pos.last_long_close = cur_long

        # ═══════════════════════════
        #  CHECK EXITS
        # ═══════════════════════════
        for pos in list(positions):
            days_held = (date - pos.entry_date).days

            # Current spread for this venue pair
            pair_row = day_data[day_data["venue_pair"] == pos.venue_pair]
            if pair_row.empty:
                # No data for this pair today → hold
                continue

            pr = pair_row.iloc[0]
            cur_spread_ann = pr["spread_ann_pct"]
            cur_z = pr.get("z_score", np.nan)
            cur_spot = pr.get("spot_close", pos.entry_spot)
            exit_reason = None

            # 1. CONVERGENCE EXIT: spread narrowed below threshold
            if abs(cur_spread_ann) < cfg.spread_exit_ann_pct:
                exit_reason = "converged"

            # 2. SPREAD FLIP: spread changed sign
            if exit_reason is None and cfg.spread_flip_exit:
                if pos.direction == "long_spread" and cur_spread_ann < 0:
                    exit_reason = "spread_flip"
                elif pos.direction == "short_spread" and cur_spread_ann > 0:
                    exit_reason = "spread_flip"

            # 3. BASIS STOP (case study: "Exit if basis moves against position by >5%")
            if exit_reason is None and pos.notional > 0:
                basis_pct = pos.cum_basis_pnl / pos.notional * 100
                if basis_pct < cfg.basis_stop_pct:
                    exit_reason = "basis_stop"

            # 4. Cumulative funding stop: net funding went badly negative
            if exit_reason is None:
                if pos.cum_net_funding_pct < cfg.stop_loss_cum_funding_pct:
                    exit_reason = "funding_stop"

            # 4. Z-score regime flip
            if exit_reason is None and not pd.isna(cur_z):
                if pos.direction == "long_spread" and cur_z < cfg.z_stop:
                    exit_reason = "z_regime_flip"
                elif pos.direction == "short_spread" and cur_z > -cfg.z_stop:
                    exit_reason = "z_regime_flip"

            # 5. Max hold
            if exit_reason is None and days_held >= cfg.max_hold_days:
                exit_reason = "max_hold"

            # ── Execute exit ──
            if exit_reason is not None:
                pair_tuple = (pos.short_venue, pos.long_venue)
                pair_costs = rt_costs.get(pair_tuple, {
                    "round_trip_rate": 0.002, "gas_usd": 0.0})
                # Fixed: only exit cost here (entry cost already deducted)
                exit_rate = pair_costs.get("exit_rate",
                                           pair_costs["round_trip_rate"] / 2)
                exit_cost = (pos.notional * exit_rate
                             + pair_costs["gas_usd"] / 2)

                # Fixed: use accumulated basis P&L instead of hardcoded zero
                basis_pnl = pos.cum_basis_pnl

                # Total cost = entry (already deducted) + exit
                total_cost = pos.entry_cost + exit_cost

                gross = pos.net_funding_pnl + basis_pnl
                net = gross - exit_cost  # only subtract exit cost from equity
                equity += net
                cum_net_funding_total += pos.net_funding_pnl
                cooldown_map[pos.venue_pair] = date

                # Average annualised net funding
                if days_held > 0:
                    avg_daily_net = pos.net_funding_pnl / (days_held * pos.notional)
                    avg_ann_net = avg_daily_net * 365.25 * 100
                else:
                    avg_ann_net = 0.0

                completed.append(Trade(
                    trade_id=pos.trade_id,
                    venue_pair=pos.venue_pair,
                    short_venue=pos.short_venue,
                    long_venue=pos.long_venue,
                    entry_date=pos.entry_date,
                    exit_date=date,
                    days_held=days_held,
                    direction=pos.direction,
                    entry_spread_ann_pct=pos.entry_spread_ann_pct,
                    exit_spread_ann_pct=cur_spread_ann,
                    entry_z=pos.entry_z,
                    exit_z=cur_z if not pd.isna(cur_z) else 0,
                    entry_spot=pos.entry_spot,
                    exit_spot=cur_spot if not pd.isna(cur_spot) else pos.entry_spot,
                    exit_reason=exit_reason,
                    regime_at_entry=pos.regime_at_entry,
                    regime_multiplier=pos.regime_multiplier,
                    z_size_scalar=pos.z_size_scalar,
                    notional=pos.notional,
                    cum_funding_received=pos.cum_funding_received,
                    cum_funding_paid=pos.cum_funding_paid,
                    gross_funding_pnl=pos.net_funding_pnl,
                    basis_pnl=basis_pnl,
                    gross_pnl=gross,
                    cost=total_cost,  # report full round-trip cost
                    net_pnl=gross - total_cost,  # net after all costs
                    pnl_pct=(gross - total_cost) / pos.notional if pos.notional > 0 else 0,
                    avg_net_funding_ann_pct=avg_ann_net,
                ))
                positions.remove(pos)

        # ═══════════════════════════
        #  CHECK ENTRIES — enter ALL valid pairs (carry trade: be deployed)
        # ═══════════════════════════
        if len(positions) < cfg.max_positions:
            open_pairs = {p.venue_pair for p in positions}

            for _, pr in day_data.iterrows():
                if len(positions) >= cfg.max_positions:
                    break

                pair_key = pr["venue_pair"]
                if pair_key in open_pairs:
                    continue

                # Cooldown
                last_exit = cooldown_map.get(pair_key)
                if last_exit is not None:
                    cooldown_days = cfg.cooldown_periods / 3
                    if (date - last_exit).days < cooldown_days:
                        continue

                spread_ann = pr["spread_ann_pct"]
                z = pr.get("z_score", np.nan)

                if pd.isna(spread_ann):
                    continue

                # Direction and entry check
                direction = None
                if spread_ann > cfg.spread_entry_ann_pct:
                    direction = "long_spread"
                elif spread_ann < -cfg.spread_entry_ann_pct:
                    direction = "short_spread"

                if direction is None:
                    continue

                # Minimum raw spread (anti-noise)
                spread_raw = pr.get("spread_raw", 0)
                if abs(spread_raw) < cfg.spread_min_raw:
                    continue

                # ── Position sizing (case study: venue limits + regime) ──
                trade_counter += 1
                regime = pr.get("regime", "flat")
                regime_mult = cfg.regime_multiplier(regime)

                sv = pr["short_venue"]
                lv = pr["long_venue"]

                # Size = base × regime, capped by venue limit
                max_exp = min(
                    cfg.max_exposure_pct.get(sv, 0.50),
                    cfg.max_exposure_pct.get(lv, 0.50),
                )
                effective_pct = min(cfg.base_position_pct * regime_mult, max_exp)

                unrealized_for_sizing = sum(
                    p.net_funding_pnl + p.cum_basis_pnl for p in positions)
                mtm_equity = equity + unrealized_for_sizing
                notional = mtm_equity * effective_pct

                if notional >= cfg.min_trade_usd:
                    pair_tuple = (sv, lv)
                    pair_costs = rt_costs.get(pair_tuple, {
                        "round_trip_rate": 0.002, "gas_usd": 0.0})
                    entry_rate = pair_costs.get(
                        "entry_rate",
                        pair_costs["round_trip_rate"] / 2)
                    entry_cost = (notional * entry_rate
                                  + pair_costs["gas_usd"] / 2)
                    equity -= entry_cost

                    entry_spot = pr.get("spot_close", 0)
                    short_col = f"{sv}_perp_close"
                    long_col = f"{lv}_perp_close"
                    entry_short_close = pr.get(short_col, entry_spot)
                    entry_long_close = pr.get(long_col, entry_spot)

                    new_pos = OpenPosition(
                        trade_id=f"C_{trade_counter:04d}",
                        venue_pair=pair_key,
                        short_venue=sv,
                        long_venue=lv,
                        entry_date=date,
                        direction=direction,
                        entry_spread_ann_pct=spread_ann,
                        entry_z=z if not pd.isna(z) else 0,
                        entry_spot=entry_spot,
                        regime_at_entry=regime,
                        regime_multiplier=regime_mult,
                        z_size_scalar=1.0,  # not used for sizing
                        notional=notional,
                        entry_cost=entry_cost,
                        last_short_close=entry_short_close,
                        last_long_close=entry_long_close,
                    )
                    positions.append(new_pos)
                    open_pairs.add(pair_key)

        # ═══════════════════════════
        #  EQUITY SNAPSHOT
        # ═══════════════════════════
        unrealized = sum(
            p.net_funding_pnl + p.cum_basis_pnl for p in positions)

        # Get regime from any row for this date
        if not day_data.empty:
            day_regime = day_data.iloc[0].get("regime", "flat")
        else:
            day_regime = "flat"

        equity_hist.append({
            "date": date,
            "equity": equity + unrealized,
            "n_open": len(positions),
            "regime": day_regime,
            "spot_close": day_spot if not pd.isna(day_spot) else np.nan,
            "cum_net_funding": cum_net_funding_total + sum(
                p.net_funding_pnl for p in positions),
        })

    # ═══════════════════════════
    #  CLOSE REMAINING
    # ═══════════════════════════
    if positions:
        last_date = trade_dates[-1]
        for pos in list(positions):
            days_held = (last_date - pos.entry_date).days

            pair_row = trade_data[
                (trade_data["date"] == last_date) &
                (trade_data["venue_pair"] == pos.venue_pair)
            ]
            if not pair_row.empty:
                pr = pair_row.iloc[0]
                cur_spread_ann = pr["spread_ann_pct"]
                cur_z = pr.get("z_score", 0)
                cur_spot = pr.get("spot_close", pos.entry_spot)
            else:
                cur_spread_ann = pos.entry_spread_ann_pct
                cur_z = 0
                cur_spot = pos.entry_spot

            pair_tuple = (pos.short_venue, pos.long_venue)
            pair_costs = rt_costs.get(pair_tuple, {
                "round_trip_rate": 0.002, "gas_usd": 0.0})
            # Only exit cost (entry already deducted)
            exit_rate = pair_costs.get("exit_rate",
                                       pair_costs["round_trip_rate"] / 2)
            exit_cost = (pos.notional * exit_rate
                         + pair_costs["gas_usd"] / 2)
            total_cost = pos.entry_cost + exit_cost

            basis_pnl = pos.cum_basis_pnl
            gross = pos.net_funding_pnl + basis_pnl
            net = gross - exit_cost  # only subtract exit cost from equity
            equity += net
            cum_net_funding_total += pos.net_funding_pnl

            if days_held > 0:
                avg_daily_net = pos.net_funding_pnl / (days_held * pos.notional)
                avg_ann_net = avg_daily_net * 365.25 * 100
            else:
                avg_ann_net = 0.0

            completed.append(Trade(
                trade_id=pos.trade_id,
                venue_pair=pos.venue_pair,
                short_venue=pos.short_venue,
                long_venue=pos.long_venue,
                entry_date=pos.entry_date,
                exit_date=last_date,
                days_held=days_held,
                direction=pos.direction,
                entry_spread_ann_pct=pos.entry_spread_ann_pct,
                exit_spread_ann_pct=cur_spread_ann,
                entry_z=pos.entry_z,
                exit_z=cur_z,
                entry_spot=pos.entry_spot,
                exit_spot=cur_spot,
                exit_reason="end",
                regime_at_entry=pos.regime_at_entry,
                regime_multiplier=pos.regime_multiplier,
                z_size_scalar=pos.z_size_scalar,
                notional=pos.notional,
                cum_funding_received=pos.cum_funding_received,
                cum_funding_paid=pos.cum_funding_paid,
                gross_funding_pnl=pos.net_funding_pnl,
                basis_pnl=basis_pnl,
                gross_pnl=gross,
                cost=total_cost,
                net_pnl=gross - total_cost,
                pnl_pct=(gross - total_cost) / pos.notional if pos.notional > 0 else 0,
                avg_net_funding_ann_pct=avg_ann_net,
            ))

    # Correct final equity snapshot to reflect force-close costs
    if equity_hist:
        equity_hist[-1]["equity"] = equity

    # ═══════════════════════════
    #  BUILD RESULTS
    # ═══════════════════════════
    eq_df = pd.DataFrame(equity_hist)
    eq_series = eq_df.set_index("date")["equity"]
    spot_series = eq_df.set_index("date")["spot_close"].dropna()
    metrics = compute_metrics(eq_series, completed, spot_series=spot_series)
    trade_df = (pd.DataFrame([t.to_dict() for t in completed])
                if completed else pd.DataFrame())

    if completed:
        reasons = {}
        for t in completed:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        metrics["exit_reasons"] = reasons

    print(f"\n  Trades: {len(completed)}")
    if not eq_df.empty:
        final_eq = eq_series.iloc[-1]
        print(f"  Final equity: ${final_eq:,.0f} "
              f"({(final_eq / cfg.initial_capital - 1) * 100:+.1f}%)")
    if completed:
        print(f"  Exit reasons: "
              f"{', '.join(f'{k}={v}' for k, v in sorted(reasons.items()))}")
        avg_funding = np.mean([t.avg_net_funding_ann_pct for t in completed])
        print(f"  Avg net funding collected: {avg_funding:.1f}% ann")
        print(f"  Total funding received: ${sum(t.cum_funding_received for t in completed):,.2f}")
        print(f"  Total funding paid:     ${sum(t.cum_funding_paid for t in completed):,.2f}")
        print(f"  Total basis P&L:        ${sum(t.basis_pnl for t in completed):,.2f}")
        print(f"  Total costs:            ${sum(t.cost for t in completed):,.2f}")

    result = BacktestResults(
        strategy="C: Synthetic Futures from Perp Funding",
        equity_curve=eq_df,
        trade_log=trade_df,
        metrics=metrics,
        config=cfg.to_dict(),
        daily_data=spread_data,
    )
    result.summary()

    # auto-plot
    try:
        import matplotlib.pyplot as plt

        spot_for_plot = None
        if "spot_close" in eq_df.columns:
            spot_s = eq_df.set_index("date")["spot_close"].dropna()
            if len(spot_s) > 0:
                spot_for_plot = spot_s

        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        fig.suptitle("Strategy C: Synthetic Futures from Perp Funding",
                      fontsize=14, fontweight="bold")
        result.plot_equity(ax=axes[0], benchmark_spot=spot_for_plot)
        result.plot_drawdown(ax=axes[1])
        result.plot_funding_spread(ax=axes[2])
        result.plot_cumulative_funding(ax=axes[3])
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

    return result


# ═══════════════════════════════════════════════════════════════
#  ANALYSIS HELPERS
# ═══════════════════════════════════════════════════════════════

def regime_performance(result: BacktestResults) -> pd.DataFrame:
    eq = result.equity_curve.copy()
    eq["return"] = eq["equity"].pct_change()
    rows = []
    for regime, grp in eq.groupby("regime"):
        rets = grp["return"].dropna()
        if len(rets) < 3:
            continue
        rows.append({
            "regime": regime,
            "n_days": len(rets),
            "pct_of_time": round(len(rets) / len(eq) * 100, 1),
            "mean_daily_ret_bps": round(rets.mean() * 10000, 2),
            "sharpe": round(rets.mean() / rets.std() * np.sqrt(365.25), 2)
                     if rets.std() > 0 else 0,
        })
    return pd.DataFrame(rows).sort_values("n_days", ascending=False)


def venue_pair_analysis(result: BacktestResults) -> pd.DataFrame:
    if result.trade_log.empty:
        return pd.DataFrame()
    rows = []
    for pair, grp in result.trade_log.groupby("venue_pair"):
        pnls = grp["net_pnl"].astype(float).values
        rows.append({
            "venue_pair": pair,
            "n_trades": len(grp),
            "total_pnl": round(pnls.sum(), 2),
            "avg_pnl": round(pnls.mean(), 2),
            "win_rate": round((pnls > 0).mean(), 3),
            "avg_days_held": round(grp["days_held"].astype(float).mean(), 1),
            "avg_entry_spread_ann": round(
                grp["entry_spread_ann_pct"].astype(float).mean(), 1),
            "avg_net_funding_ann": round(
                grp["avg_net_funding_ann_pct"].astype(float).mean(), 1),
            "total_funding_received": round(
                grp["cum_funding_received"].astype(float).sum(), 2),
            "total_funding_paid": round(
                grp["cum_funding_paid"].astype(float).sum(), 2),
        })
    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)


def funding_decomposition(result: BacktestResults) -> pd.DataFrame:
    """Break down P&L into funding vs costs for each trade."""
    if result.trade_log.empty:
        return pd.DataFrame()
    tl = result.trade_log.copy()
    tl["gross_funding_pnl"] = tl["gross_funding_pnl"].astype(float)
    tl["cost"] = tl["cost"].astype(float)
    tl["net_pnl"] = tl["net_pnl"].astype(float)
    return tl[["trade_id", "venue_pair", "direction", "days_held",
               "gross_funding_pnl", "cost", "net_pnl",
               "avg_net_funding_ann_pct"]]


def parameter_sensitivity(
    dl, param_name: str, values: list,
    base_cfg: SyntheticConfig = None,
) -> pd.DataFrame:
    if base_cfg is None:
        base_cfg = SyntheticConfig()
    import io, contextlib

    # Properties / computed keys to strip before reconstructing config
    computed_keys = {"round_trip_costs"}
    rows = []
    for v in values:
        override = {k: v_ for k, v_ in base_cfg.to_dict().items()
                    if k not in computed_keys}
        override[param_name] = v
        # venue_pairs needs to be a tuple
        if "venue_pairs" in override and isinstance(override["venue_pairs"], list):
            override["venue_pairs"] = tuple(
                tuple(p) if isinstance(p, list) else p
                for p in override["venue_pairs"]
            )
        c = SyntheticConfig(**override)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            res = run_synthetic_futures(dl, c, test_only=True)
        m = res.metrics
        rows.append({
            param_name: v,
            "sharpe": m.get("sharpe_ratio", 0),
            "return_pct": m.get("total_return_pct", 0),
            "max_dd_pct": m.get("max_drawdown_pct", 0),
            "n_trades": m.get("n_trades", 0),
            "win_rate": m.get("win_rate", 0),
            "pnl": m.get("total_pnl", 0),
            "avg_net_funding_ann": m.get("avg_net_funding_ann_pct", 0),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  BENCHMARKS
# ═══════════════════════════════════════════════════════════════

def _compute_benchmark_metrics(
    equity: pd.Series, spot: pd.Series, label: str,
) -> dict:
    """Compute standard metrics for a benchmark equity curve."""
    if len(equity) < 2:
        return {"strategy": label, "error": "insufficient data"}

    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_days = (equity.index[-1] - equity.index[0]).days
    ann_factor = 365.25 / max(n_days, 1)
    ann_return = (1 + total_return) ** ann_factor - 1

    PERIODS_PER_YEAR = 365.25
    rf_per_period = 0.04 / PERIODS_PER_YEAR
    excess = returns - rf_per_period

    sharpe = (excess.mean() / excess.std() * np.sqrt(PERIODS_PER_YEAR)
              if excess.std() > 1e-10 else 0.0)

    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()

    # BTC correlation
    btc_corr = np.nan
    if spot is not None and len(spot) > 5:
        spot_rets = spot.pct_change().dropna()
        common = returns.index.intersection(spot_rets.index)
        if len(common) > 5:
            btc_corr = float(returns.loc[common].corr(spot_rets.loc[common]))

    return {
        "strategy": label,
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(ann_return * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "btc_correlation": round(float(btc_corr), 3) if not np.isnan(btc_corr) else "N/A",
    }


def build_benchmarks(
    dl,
    result: BacktestResults,
    cfg: SyntheticConfig = None,
) -> pd.DataFrame:
    """
    Build the three benchmark equity curves required by the case study
    and compare against the strategy result.

    Benchmarks:
        1. Buy-and-hold BTC spot
        2. Naive perp hold: long Binance perp, pay funding, no optimisation
        3. Perp hold + taker costs: same as (2) but with realistic entry drag

    All benchmarks start with the same initial capital and date range as
    the strategy result.

    Parameters
    ----------
    dl : DataLoader
    result : BacktestResults from run_synthetic_futures
    cfg : SyntheticConfig (for initial capital; defaults if None)

    Returns
    -------
    DataFrame with one row per strategy/benchmark, columns for all metrics.

    Usage:
        result = run_synthetic_futures(dl)
        bench = build_benchmarks(dl, result)
        print(bench.to_string(index=False))
    """
    if cfg is None:
        cfg = SyntheticConfig()

    eq_df = result.equity_curve.copy()
    dates = pd.to_datetime(eq_df["date"])
    start_date, end_date = dates.min(), dates.max()
    initial = cfg.initial_capital

    # ── Load spot price series ──
    # Primary: use front-month for spot_close (most accurate, tick-aligned)
    # Fallback: use raw spot data for dates beyond USDT-M contract range
    try:
        front = dl.continuous_front_month(margin_type="USDT-M")
        front["date"] = front["datetime"].dt.normalize().dt.tz_localize(None)
        spot_daily = (front.sort_values("datetime")
                      .groupby("date")["spot_close"].last())
    except Exception:
        spot_daily = pd.Series(dtype=float)

    # Extend with raw spot for dates not covered by USDT-M contracts
    try:
        spot_df = dl.spot()
        spot_df["date"] = spot_df["datetime"].dt.normalize().dt.tz_localize(None)
        spot_raw_daily = (spot_df.sort_values("datetime")
                          .groupby("date")["close"].last())
        if spot_daily.empty:
            spot_daily = spot_raw_daily
        else:
            missing_dates = spot_raw_daily.index.difference(spot_daily.index)
            if len(missing_dates) > 0:
                spot_daily = pd.concat([spot_daily, spot_raw_daily.loc[missing_dates]])
                spot_daily = spot_daily.sort_index()
    except Exception:
        pass

    # Filter to strategy date range
    mask = (spot_daily.index >= start_date) & (spot_daily.index <= end_date)
    spot_daily = spot_daily[mask].dropna()

    if spot_daily.empty:
        print("  WARNING: No spot data available for benchmark construction")
        return pd.DataFrame()

    # ── Load Binance funding rates ──
    try:
        funding_comp = dl.funding_comparison(resample=cfg.resample_freq)
        if "binance" in funding_comp.columns:
            bin_funding = funding_comp.set_index("datetime")["binance"]
            bin_funding.index = bin_funding.index.normalize()
            if bin_funding.index.tz is not None:
                bin_funding.index = bin_funding.index.tz_localize(None)
            # Sum to daily (multiple 8h periods per day)
            bin_funding_daily = bin_funding.groupby(bin_funding.index).sum()
            mask_f = ((bin_funding_daily.index >= start_date) &
                      (bin_funding_daily.index <= end_date))
            bin_funding_daily = bin_funding_daily[mask_f]
        else:
            bin_funding_daily = pd.Series(0.0, index=spot_daily.index)
    except Exception:
        bin_funding_daily = pd.Series(0.0, index=spot_daily.index)

    # ── Benchmark 1: Buy-and-hold BTC spot ──
    spot_equity = spot_daily / spot_daily.iloc[0] * initial

    # ── Benchmark 2: Naive perp hold (long Binance perp, pay funding) ──
    # P&L per day = spot_return - funding_paid
    # "Naive roll" from the case study means: hold perp with no optimisation
    # of venue selection or timing. Just pay whatever Binance charges.
    naive_dates = sorted(set(spot_daily.index) & set(bin_funding_daily.index))
    if naive_dates:
        naive_spot = spot_daily.loc[naive_dates]
        naive_funding = bin_funding_daily.loc[naive_dates].fillna(0)
        spot_returns = naive_spot.pct_change().fillna(0)
        # Long pays funding when positive, receives when negative
        naive_returns = spot_returns - naive_funding
        naive_equity = initial * (1 + naive_returns).cumprod()
    else:
        naive_equity = spot_equity.copy()

    # ── Benchmark 3: Perp hold + realistic entry/exit costs ──
    binance_costs = cfg.venue_costs("binance")
    entry_drag = binance_costs.taker_fee + binance_costs.avg_slippage
    perp_hold_equity = naive_equity * (1 - entry_drag)

    # ── Strategy equity ──
    strat_equity = eq_df.set_index("date")["equity"]

    # ── Compute metrics for each ──
    rows = []

    # Strategy (pull from existing metrics, add strategy label)
    strat_m = {}
    for k in ["total_return_pct", "annualized_return_pct", "sharpe_ratio",
              "max_drawdown_pct", "btc_correlation",
              "avg_funding_cost_ann_pct"]:
        strat_m[k] = result.metrics.get(k, "N/A")
    strat_m["strategy"] = "Strategy C (optimised, multi-venue)"
    rows.append(strat_m)

    # Benchmark 1: BTC spot
    rows.append(
        _compute_benchmark_metrics(spot_equity, spot_daily, "BTC Buy & Hold"))

    # Benchmark 2: Naive perp hold
    bm_naive = _compute_benchmark_metrics(
        naive_equity, spot_daily, "Naive Perp Hold (Binance)")
    if len(bin_funding_daily) > 0:
        avg_naive_funding = float(bin_funding_daily.mean()) * 365.25 * 100
        bm_naive["avg_funding_cost_ann_pct"] = round(avg_naive_funding, 2)
    rows.append(bm_naive)

    # Benchmark 3: Perp hold + costs
    bm_perp = _compute_benchmark_metrics(
        perp_hold_equity, spot_daily, "Perp Hold + Taker Costs")
    if "avg_funding_cost_ann_pct" in bm_naive:
        bm_perp["avg_funding_cost_ann_pct"] = bm_naive["avg_funding_cost_ann_pct"]
    rows.append(bm_perp)

    comp_df = pd.DataFrame(rows)

    # Order columns per case study spec
    display_cols = [
        "strategy",
        "total_return_pct",
        "annualized_return_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "btc_correlation",
        "avg_funding_cost_ann_pct",
    ]
    display_cols = [c for c in display_cols if c in comp_df.columns]
    return comp_df[display_cols].copy()


def venue_profitability(result: BacktestResults) -> pd.DataFrame:
    """
    Venue-specific profitability breakdown with per-venue Sharpe,
    return, and funding metrics — as required by the case study.

    Usage:
        result = run_synthetic_futures(dl)
        vp = venue_profitability(result)
        print(vp.to_string(index=False))
    """
    if result.trade_log.empty:
        return pd.DataFrame()

    tl = result.trade_log.copy()
    for col in ["net_pnl", "gross_funding_pnl", "basis_pnl", "cost",
                "days_held", "notional", "cum_funding_received",
                "cum_funding_paid", "avg_net_funding_ann_pct",
                "entry_spread_ann_pct"]:
        if col in tl.columns:
            tl[col] = tl[col].astype(float)

    eq_df = result.equity_curve.copy()
    initial = eq_df["equity"].iloc[0]

    rows = []
    for pair, grp in tl.groupby("venue_pair"):
        pnls = grp["net_pnl"].values
        n = len(grp)
        total_pnl = pnls.sum()
        total_notional_days = (
            grp["notional"] * grp["days_held"].clip(lower=1)).sum()

        # Estimate Sharpe from trade-level daily returns
        if total_notional_days > 0:
            avg_daily_ret = total_pnl / total_notional_days
            trade_daily_rets = pnls / (
                grp["notional"].values
                * np.maximum(grp["days_held"].values, 1))
            daily_std = (trade_daily_rets.std()
                         if len(trade_daily_rets) > 1 else 1e-6)
            venue_sharpe = (avg_daily_ret / max(daily_std, 1e-10)
                            * np.sqrt(365.25))
        else:
            venue_sharpe = 0.0

        total_funding_rcvd = grp["cum_funding_received"].sum()
        total_funding_paid = grp["cum_funding_paid"].sum()
        total_basis = (grp["basis_pnl"].sum()
                       if "basis_pnl" in grp.columns else 0)
        total_cost = grp["cost"].sum()

        avg_funding_cost_ann = (
            (total_funding_paid / total_notional_days) * 365.25 * 100
            if total_notional_days > 0 else 0.0)

        rows.append({
            "venue_pair": pair,
            "n_trades": n,
            "total_pnl": round(total_pnl, 2),
            "return_on_initial_pct": round(total_pnl / initial * 100, 2),
            "sharpe_estimate": round(venue_sharpe, 2),
            "win_rate": round((pnls > 0).mean(), 3),
            "avg_pnl_per_trade": round(pnls.mean(), 2),
            "avg_days_held": round(grp["days_held"].mean(), 1),
            "avg_entry_spread_ann_pct": round(
                grp["entry_spread_ann_pct"].mean(), 1),
            "avg_net_funding_ann_pct": round(
                grp["avg_net_funding_ann_pct"].mean(), 1),
            "avg_funding_cost_ann_pct": round(avg_funding_cost_ann, 2),
            "total_funding_received": round(total_funding_rcvd, 2),
            "total_funding_paid": round(total_funding_paid, 2),
            "total_basis_pnl": round(total_basis, 2),
            "total_cost": round(total_cost, 2),
        })

    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)


# ═══════════════════════════════════════════════════════════════
#  FULL PERFORMANCE REPORT
# ═══════════════════════════════════════════════════════════════

def full_performance_report(
    dl,
    result: BacktestResults,
    cfg: SyntheticConfig = None,
    plot: bool = True,
) -> dict:
    """
    Run all required analyses from the case study spec and print
    a formatted report. Call this on your BacktestResults.

    Covers every item from the case study performance requirements:
      - Total return, annualized
      - Sharpe ratio
      - Max drawdown
      - Correlation to BTC spot
      - Average funding cost
      - Venue-specific profitability
      - Benchmark comparisons (BTC spot, naive perp, perp + costs)
      - Regime breakdown
      - P&L decomposition (funding vs basis vs costs)

    Usage:
        from synthetic_futures_backtest import (
            run_synthetic_futures, full_performance_report
        )

        result = run_synthetic_futures(dl)
        report = full_performance_report(dl, result)

        # Access individual tables:
        report["benchmarks"]            # comparison DataFrame
        report["venue_profitability"]   # per-venue breakdown
        report["regime_performance"]    # per-regime Sharpe
        report["funding_decomposition"] # per-trade P&L breakdown

    Returns dict with all analysis DataFrames.
    """
    if cfg is None:
        cfg = SyntheticConfig()

    print("\n" + "═" * 70)
    print("  FULL PERFORMANCE REPORT — Strategy C: Synthetic Futures")
    print("═" * 70)

    # ── 1. Strategy metrics ──
    print("\n┌─────────────────────────────────────┐")
    print("│  1. STRATEGY PERFORMANCE METRICS    │")
    print("└─────────────────────────────────────┘")
    result.summary()

    # ── 2. Benchmark comparison ──
    print("\n┌─────────────────────────────────────┐")
    print("│  2. BENCHMARK COMPARISON            │")
    print("└─────────────────────────────────────┘")
    bench_df = build_benchmarks(dl, result, cfg)
    if not bench_df.empty:
        print(bench_df.to_string(index=False))
    else:
        print("  Could not build benchmarks (missing data)")

    # ── 3. Venue-specific profitability ──
    print("\n┌─────────────────────────────────────┐")
    print("│  3. VENUE-SPECIFIC PROFITABILITY    │")
    print("└─────────────────────────────────────┘")
    venue_df = venue_profitability(result)
    if not venue_df.empty:
        display_cols = [
            "venue_pair", "n_trades", "total_pnl", "sharpe_estimate",
            "win_rate", "avg_days_held", "avg_net_funding_ann_pct",
            "avg_funding_cost_ann_pct", "total_basis_pnl", "total_cost",
        ]
        display_cols = [c for c in display_cols if c in venue_df.columns]
        print(venue_df[display_cols].to_string(index=False))
    else:
        print("  No trades to analyse")

    # ── 4. Regime performance ──
    print("\n┌─────────────────────────────────────┐")
    print("│  4. REGIME PERFORMANCE              │")
    print("└─────────────────────────────────────┘")
    regime_df = regime_performance(result)
    if not regime_df.empty:
        print(regime_df.to_string(index=False))
    else:
        print("  No regime data")

    # ── 5. P&L decomposition ──
    print("\n┌─────────────────────────────────────┐")
    print("│  5. P&L DECOMPOSITION               │")
    print("└─────────────────────────────────────┘")
    m = result.metrics
    print(f"  Gross funding P&L:   ${m.get('net_funding_pnl', 0):>12,.2f}")
    print(f"  Basis P&L (drag):    ${m.get('total_basis_pnl', 0):>12,.2f}")
    print(f"  Transaction costs:  -${m.get('total_cost', 0):>12,.2f}")
    print(f"  ─────────────────────────────────")
    print(f"  Net P&L:             ${m.get('total_pnl', 0):>12,.2f}")

    # ── 6. Plots ──
    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            fig.suptitle(
                "Strategy C: Synthetic Futures from Perp Funding",
                fontsize=14, fontweight="bold")

            # Equity with BTC benchmark
            spot_for_plot = None
            eq_df = result.equity_curve
            if "spot_close" in eq_df.columns:
                spot_s = eq_df.set_index("date")["spot_close"].dropna()
                if len(spot_s) > 0:
                    spot_for_plot = spot_s
            result.plot_equity(ax=axes[0], benchmark_spot=spot_for_plot)

            # Drawdown
            result.plot_drawdown(ax=axes[1])

            # Funding spread with trades
            result.plot_funding_spread(ax=axes[2])

            plt.tight_layout()
            plt.savefig(
                "strategy_c_report.png", dpi=150, bbox_inches="tight")
            print("\n  Saved plots to strategy_c_report.png")
        except ImportError:
            print("\n  matplotlib not available, skipping plots")

    print("\n" + "═" * 70)

    return {
        "benchmarks": bench_df,
        "venue_profitability": venue_df,
        "regime_performance": regime_df,
        "funding_decomposition": funding_decomposition(result),
    }
