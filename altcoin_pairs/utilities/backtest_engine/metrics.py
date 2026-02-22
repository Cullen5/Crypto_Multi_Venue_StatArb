"""
metrics.py -- performance metrics from equity curve and trade log
"""

import numpy as np
import pandas as pd


def _resample_daily(equity_curve: pd.DataFrame, eq_col: str = "total_equity"):
    """
    Resample equity to daily frequency for Sharpe/Sortino.

    Sub-daily returns inflate Sharpe because:
      - many zero-return periods dilute std
      - sqrt(N) annualization assumes i.i.d., which fails for
        strategies that are frequently flat
    """
    if "timestamp" not in equity_curve.columns or len(equity_curve) < 10:
        return equity_curve[eq_col].values, None

    ts = pd.to_datetime(equity_curve["timestamp"])
    eq = equity_curve[eq_col].values
    eq_series = pd.Series(eq, index=ts)
    eq_series = eq_series[~eq_series.index.duplicated(keep="last")].sort_index()
    eq_daily = eq_series.resample("1D").last().dropna()

    if len(eq_daily) < 3:
        return eq, None
    return eq_daily.values, eq_daily.index


def compute_all_metrics(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    risk_free_rate: float = 0.04,
    periods_per_year: float = 365,
) -> dict:
    """
    Compute performance metrics from equity curve and trade log.

    Works with both old-style (leg-level) and new-style (spread-level) trade logs.
    """
    if len(equity_curve) < 2:
        return {"error": "not enough data"}

    eq_col = "total_equity"
    if eq_col not in equity_curve.columns:
        # try spread_equity from old format
        if "spread_equity" in equity_curve.columns:
            eq_col = "spread_equity"
        else:
            return {"error": f"no equity column found"}

    equity_daily, _ = _resample_daily(equity_curve, eq_col)
    equity_raw = equity_curve[eq_col].values

    returns = np.diff(equity_daily) / equity_daily[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return {"error": "not enough returns"}

    ann_factor = np.sqrt(periods_per_year)
    rf_per_period = risk_free_rate / periods_per_year
    metrics = {}

    # returns
    metrics["total_return"] = (equity_daily[-1] / equity_daily[0]) - 1

    if "timestamp" in equity_curve.columns and len(equity_curve) >= 2:
        t0 = pd.Timestamp(equity_curve["timestamp"].iloc[0])
        t1 = pd.Timestamp(equity_curve["timestamp"].iloc[-1])
        years = max((t1 - t0).total_seconds() / (365.25 * 86400), 1e-6)
        metrics["annualized_return"] = (1 + metrics["total_return"]) ** (1 / years) - 1
    else:
        metrics["annualized_return"] = (
            (1 + metrics["total_return"]) ** (periods_per_year / len(returns)) - 1)

    metrics["daily_return_std"] = np.std(returns)
    metrics["annualized_vol"] = metrics["daily_return_std"] * ann_factor

    # risk-adjusted
    excess = returns - rf_per_period
    std_excess = np.std(excess)
    metrics["sharpe_ratio"] = (
        (np.mean(excess) / std_excess) * ann_factor if std_excess > 1e-10 else 0.0)

    downside = excess[excess < 0]
    ds_std = np.std(downside) if len(downside) > 1 else 1e-10
    metrics["sortino_ratio"] = (
        (np.mean(excess) / ds_std) * ann_factor if ds_std > 1e-10 else 0.0)

    # drawdown
    running_max = np.maximum.accumulate(equity_raw)
    drawdowns = (equity_raw - running_max) / running_max
    metrics["max_drawdown"] = float(np.min(drawdowns))
    metrics["avg_drawdown"] = (
        float(np.mean(drawdowns[drawdowns < 0]))
        if np.any(drawdowns < 0) else 0.0)

    metrics["calmar_ratio"] = (
        metrics["annualized_return"] / abs(metrics["max_drawdown"])
        if metrics["max_drawdown"] < -0.001 else 0.0)

    # trade-level
    if len(trade_log) > 0:
        # new format: each row is a completed spread trade with net_pnl
        if "net_pnl" in trade_log.columns:
            pnls = trade_log["net_pnl"].values
        elif "pnl" in trade_log.columns:
            # old format: aggregate by trade_group_id if available
            if "trade_group_id" in trade_log.columns:
                close_mask = trade_log["action"].isin(
                    ["CLOSE", "STOP_LOSS", "FORCED_CLOSE"])
                closes = trade_log[close_mask]
                pnls = closes.groupby("trade_group_id")["pnl"].sum().values
            else:
                pnls = trade_log["pnl"].dropna().values
        else:
            pnls = np.array([])

        if len(pnls) > 0:
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            metrics["num_trades"] = len(pnls)
            metrics["win_rate"] = len(wins) / len(pnls)
            metrics["avg_win"] = float(np.mean(wins)) if len(wins) > 0 else 0
            metrics["avg_loss"] = float(np.mean(losses)) if len(losses) > 0 else 0
            metrics["profit_factor"] = (
                abs(np.sum(wins) / np.sum(losses))
                if np.sum(losses) != 0 else float("inf"))
            metrics["avg_trade_pnl"] = float(np.mean(pnls))
            metrics["best_trade"] = float(np.max(pnls))
            metrics["worst_trade"] = float(np.min(pnls))

        # costs
        if "total_cost" in trade_log.columns:
            total_costs = trade_log["total_cost"].sum()
            metrics["total_costs"] = total_costs
            metrics["cost_drag"] = total_costs / equity_raw[0]

        # holding periods
        if "bars_held" in trade_log.columns:
            bh = trade_log["bars_held"]
            metrics["avg_bars_held"] = float(bh.mean())
            metrics["median_bars_held"] = float(bh.median())

    # misc
    metrics["num_days"] = len(returns)
    if "timestamp" in equity_curve.columns:
        metrics["start_date"] = equity_curve["timestamp"].iloc[0]
        metrics["end_date"] = equity_curve["timestamp"].iloc[-1]
    metrics["initial_equity"] = equity_raw[0]
    metrics["final_equity"] = equity_raw[-1]

    return metrics


def print_summary(metrics: dict, title: str = "Backtest Results"):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    sections = {
        "Returns": ["total_return", "annualized_return", "annualized_vol"],
        "Risk-Adjusted": ["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
        "Drawdown": ["max_drawdown", "avg_drawdown"],
        "Trading": ["num_trades", "win_rate", "avg_win", "avg_loss",
                     "profit_factor", "avg_trade_pnl"],
        "Costs": ["total_costs", "cost_drag"],
        "Holding": ["avg_bars_held", "median_bars_held"],
        "Leverage": ["max_leverage_setting", "liquidation_events",
                      "peak_margin_usage", "peak_actual_leverage",
                      "total_interest_paid"],
    }

    for section, keys in sections.items():
        present = [k for k in keys if k in metrics]
        if not present:
            continue
        print(f"\n  {section}:")
        for key in present:
            val = metrics[key]
            if isinstance(val, float):
                if any(k in key for k in ["sharpe", "sortino", "calmar", "factor"]):
                    print(f"    {key:30s} {val:>12.3f}")
                elif "rate" in key or "return" in key or "drag" in key or "drawdown" in key:
                    print(f"    {key:30s} {val:>12.4%}")
                else:
                    print(f"    {key:30s} {val:>12.2f}")
            elif isinstance(val, int):
                print(f"    {key:30s} {val:>12d}")
            else:
                print(f"    {key:30s} {str(val):>12s}")

    if "start_date" in metrics:
        print(f"\n  Period: {metrics['start_date']} → {metrics['end_date']}")
    if "initial_equity" in metrics:
        print(f"  Capital: ${metrics['initial_equity']:,.0f} → ${metrics['final_equity']:,.0f}")
    print(f"{'='*60}\n")
