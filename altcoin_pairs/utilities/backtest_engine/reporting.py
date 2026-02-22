"""
reporting.py -- trade log export and performance charts
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def generate_reports(
    results,
    btc_series: Optional[pd.Series] = None,
    spread_data: Optional[dict] = None,
    output_dir: str = "./reports",
    bars_per_day: int = 96,
    rolling_sharpe_window_days: int = 90,
):
    """Generate all reports (CSVs + charts) from backtest results."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    eq_df = results.equity_curve
    tr_df = results.trade_log
    pair_stats = results.pair_stats

    if tr_df is not None and len(tr_df) > 0:
        trade_path = outdir / "trade_log.csv"
        tr_df.to_csv(trade_path, index=False)
        print(f"  [report] Trade log: {trade_path} ({len(tr_df)} trades)")

    if pair_stats is not None and len(pair_stats) > 0:
        pair_path = outdir / "pair_rankings.csv"
        pair_stats.to_csv(pair_path)
        print(f"  [report] Pair rankings: {pair_path} ({len(pair_stats)} pairs)")

    if not _HAS_MPL:
        print("  [report] matplotlib not available -- skipping charts")
        return

    if len(eq_df) < 10:
        print("  [report] not enough equity data for charts")
        return

    sharpe_path = outdir / "rolling_sharpe.png"
    plot_rolling_sharpe(eq_df, sharpe_path,
                        window_days=rolling_sharpe_window_days)
    print(f"  [report] Rolling Sharpe: {sharpe_path}")

    eq_vs_btc_path = outdir / "equity_vs_btc.png"
    plot_equity_vs_btc(eq_df, btc_series, eq_vs_btc_path)
    print(f"  [report] Equity vs BTC: {eq_vs_btc_path}")

    if tr_df is not None and len(tr_df) > 0 and "exit_reason" in tr_df.columns:
        plot_pnl_by_reason(tr_df, outdir)
        print(f"  [report] PnL by reason: {outdir}/pnl_*_by_reason.png")

    if (spread_data is not None and pair_stats is not None
            and len(pair_stats) > 0):
        _plot_top_bottom_spreads(pair_stats, spread_data, tr_df, outdir)

    if tr_df is not None and len(tr_df) > 20:
        plot_win_quality_diagnostic(tr_df, outdir)
        print(f"  [report] Win quality: {outdir}/diag_*.png")

    print()


# rolling sharpe

def plot_rolling_sharpe(
    eq_df: pd.DataFrame, save_path,
    window_days: int = 30,
    risk_free_rate: float = 0.04,
):
    """Rolling Sharpe from daily-resampled equity."""
    eq_col = "total_equity"
    if eq_col not in eq_df.columns:
        return

    ts = pd.to_datetime(eq_df["timestamp"])
    eq = pd.Series(eq_df[eq_col].values, index=ts)
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()
    eq_daily = eq.resample("1D").last().dropna()
    if len(eq_daily) < window_days + 5:
        return

    daily_returns = eq_daily.pct_change().dropna()
    rf_daily = risk_free_rate / 365
    excess = daily_returns - rf_daily
    rolling_mean = excess.rolling(window=window_days).mean()
    rolling_std = excess.rolling(window=window_days).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(365)
    rolling_sharpe = rolling_sharpe.dropna()
    if len(rolling_sharpe) < 5:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    dates = rolling_sharpe.index
    values = rolling_sharpe.values

    ax.plot(dates, values, color="#333333", linewidth=1.0, alpha=0.8)
    ax.fill_between(dates, values, 0,
                     where=values >= 0, color="#2ca02c", alpha=0.2)
    ax.fill_between(dates, values, 0,
                     where=values < 0, color="#d62728", alpha=0.2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axhline(1.0, color="#2ca02c", linestyle=":", linewidth=0.7, alpha=0.4)
    ax.axhline(-1.0, color="#d62728", linestyle=":", linewidth=0.7, alpha=0.4)

    overall_sharpe = (
        (excess.mean() / excess.std()) * np.sqrt(365)
        if excess.std() > 1e-10 else 0.0)
    ax.axhline(overall_sharpe, color="#1f77b4", linestyle="-",
               linewidth=1.5, alpha=0.6,
               label=f"Overall Sharpe = {overall_sharpe:.2f}")

    ax.set_title(f"Rolling Sharpe Ratio ({window_days}-day window)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio (annualized)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


# equity vs BTC benchmark

def plot_equity_vs_btc(
    eq_df: pd.DataFrame,
    btc_series: Optional[pd.Series],
    save_path,
):
    """Strategy cumulative return vs BTC buy-and-hold."""
    eq_col = "total_equity"
    if eq_col not in eq_df.columns or "timestamp" not in eq_df.columns:
        return

    ts = pd.to_datetime(eq_df["timestamp"].values)
    equity = eq_df[eq_col].values.astype(float)
    strat_return = (equity / equity[0] - 1) * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts, strat_return, color="#1f77b4", linewidth=1.5,
            label="Pairs Strategy", zorder=3)
    ax.fill_between(ts, strat_return, 0,
                     where=strat_return >= 0, color="#1f77b4",
                     alpha=0.08, interpolate=True)
    ax.fill_between(ts, strat_return, 0,
                     where=strat_return < 0, color="#d62728",
                     alpha=0.08, interpolate=True)

    if btc_series is not None and len(btc_series) > 10:
        btc_ts = pd.to_datetime(btc_series.index)
        btc_vals = btc_series.values.astype(float)
        valid = np.isfinite(btc_vals)
        if valid.sum() > 10:
            btc_ts, btc_vals = btc_ts[valid], btc_vals[valid]
            btc_return = (btc_vals / btc_vals[0] - 1) * 100
            ax.plot(btc_ts, btc_return, color="#ff7f0e", linewidth=1.3,
                    alpha=0.7, label="BTC Buy & Hold", zorder=2)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    final_strat = strat_return[-1]
    ax.annotate(f"{final_strat:+.1f}%",
                xy=(ts[-1], final_strat),
                xytext=(10, 0), textcoords="offset points",
                fontsize=10, fontweight="bold", color="#1f77b4", va="center")

    ax.set_title("Strategy P/L vs BTC Buy & Hold",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="best", fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


# P/L breakdown by exit reason (2 charts)

def plot_pnl_by_reason(trade_log: pd.DataFrame, save_dir):
    """Two separate bar charts: total P/L and avg P/L by exit reason."""
    if "exit_reason" not in trade_log.columns:
        return
    pnl_col = "net_pnl" if "net_pnl" in trade_log.columns else "pnl"
    if pnl_col not in trade_log.columns:
        return

    grouped = trade_log.groupby("exit_reason").agg(
        total_pnl=(pnl_col, "sum"),
        n_trades=(pnl_col, "count"),
        avg_pnl=(pnl_col, "mean"),
        win_rate=(pnl_col, lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0),
    ).sort_values("total_pnl", ascending=True)

    save_dir = Path(save_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in grouped["total_pnl"]]
    bars = ax.barh(grouped.index, grouped["total_pnl"], color=colors,
                   edgecolor="white", linewidth=0.5)
    ax.set_title("Total P/L by Exit Reason", fontsize=12, fontweight="bold")
    ax.set_xlabel("Total Net P/L ($)")
    ax.axvline(0, color="gray", linewidth=0.7)
    for bar, (_, row) in zip(bars, grouped.iterrows()):
        x_pos = bar.get_width()
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"  n={int(row['n_trades'])}",
                va="center", fontsize=9, color="#555555")
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_dir / "pnl_total_by_reason.png", dpi=150,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_avg = ["#2ca02c" if v >= 0 else "#d62728" for v in grouped["avg_pnl"]]
    ax.barh(grouped.index, grouped["avg_pnl"], color=colors_avg,
            edgecolor="white", linewidth=0.5)
    ax.set_title("Avg P/L per Trade by Exit Reason",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Avg Net P/L ($)")
    ax.axvline(0, color="gray", linewidth=0.7)
    for i, (_, row) in enumerate(grouped.iterrows()):
        ax.text(row["avg_pnl"], i,
                f"  WR={row['win_rate']:.0%}",
                va="center", fontsize=9, color="#555555")
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_dir / "pnl_avg_by_reason.png", dpi=150,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


# win quality diagnostics (6 charts)

def plot_win_quality_diagnostic(trade_log: pd.DataFrame, save_dir):
    """Save 6 individual diagnostic charts to save_dir."""
    if not _HAS_MPL or len(trade_log) < 10:
        return

    save_dir = Path(save_dir)

    df = trade_log.copy()
    for col in ["entry_z", "exit_z", "gross_pnl", "total_cost",
                "net_pnl", "pnl_pct", "notional", "bars_held",
                "entry_adaptive_z", "entry_z_peak", "entry_confidence",
                "entry_quality"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sig = df[df["exit_reason"] == "signal"].copy()
    all_exits = df.copy()

    if len(sig) < 5:
        return

    sig["abs_entry_z"] = sig["entry_z"].abs()
    sig["abs_exit_z"] = sig["exit_z"].abs()
    sig["z_travel"] = sig["abs_entry_z"] - sig["abs_exit_z"]
    winners = sig[sig["net_pnl"] > 0]
    losers = sig[sig["net_pnl"] <= 0]

    def _save(fig, name):
        fig.savefig(save_dir / name, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)

    # entry |z| distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    bins_z = np.linspace(0, sig["abs_entry_z"].quantile(0.98), 30)
    ax.hist(winners["abs_entry_z"], bins=bins_z, alpha=0.6, color="#2ca02c",
            label=f"Winners (n={len(winners)})", density=True)
    ax.hist(losers["abs_entry_z"], bins=bins_z, alpha=0.6, color="#d62728",
            label=f"Losers (n={len(losers)})", density=True)
    if "entry_adaptive_z" in sig.columns:
        median_adapt = sig["entry_adaptive_z"].median()
        ax.axvline(median_adapt, color="#ff7f0e", linestyle="--", linewidth=1.5,
                   label=f"Median adaptive thresh: {median_adapt:.2f}")
    ax.set_xlabel("|Entry Z-Score|")
    ax.set_ylabel("Density")
    ax.set_title("Entry |Z| Distribution (Signal Exits)")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    _save(fig, "diag_entry_z_dist.png")

    # z-travel distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    z_travel_win = winners["z_travel"]
    z_travel_loss = losers["z_travel"]
    bins_t = np.linspace(
        min(sig["z_travel"].quantile(0.02), -0.5),
        sig["z_travel"].quantile(0.98), 30)
    ax.hist(z_travel_win, bins=bins_t, alpha=0.6, color="#2ca02c",
            label=f"Winners: \u03bc={z_travel_win.mean():.2f}\u03c3", density=True)
    ax.hist(z_travel_loss, bins=bins_t, alpha=0.6, color="#d62728",
            label=f"Losers: \u03bc={z_travel_loss.mean():.2f}\u03c3", density=True)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.7)
    ax.set_xlabel("Z-Travel (|entry_z| \u2212 |exit_z|)")
    ax.set_ylabel("Density")
    ax.set_title("Z-Travel Captured")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    _save(fig, "diag_z_travel.png")

    # gross P/L vs cost
    if "gross_pnl" in sig.columns and "total_cost" in sig.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        gross = sig["gross_pnl"].values
        cost = sig["total_cost"].values
        colors = ["#2ca02c" if g > c else "#d62728"
                  for g, c in zip(gross, cost)]
        ax.scatter(cost, gross, c=colors, alpha=0.3, s=8, edgecolors="none")
        max_val = max(abs(gross).max(), abs(cost).max()) * 1.1
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8,
                alpha=0.5, label="Break-even")
        total_gross = sig[sig["gross_pnl"] > 0]["gross_pnl"].sum()
        total_cost_val = sig["total_cost"].sum()
        cost_drag = total_cost_val / total_gross * 100 if total_gross > 0 else 0
        ax.text(0.05, 0.95, f"Cost drag: {cost_drag:.0f}% of gross wins",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
        ax.set_xlabel("Transaction Cost ($)")
        ax.set_ylabel("Gross P/L ($)")
        ax.set_title("Gross P/L vs Transaction Cost")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)
        plt.tight_layout()
        _save(fig, "diag_gross_vs_cost.png")

    # net P/L % distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    pnl_pcts = sig["pnl_pct"].dropna() * 100
    p5, p95 = pnl_pcts.quantile(0.001), pnl_pcts.quantile(0.999)
    bins_pnl = np.linspace(p5, p95, 40)
    ax.hist(pnl_pcts, bins=bins_pnl, color="#555555", alpha=0.7,
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.axvline(pnl_pcts.median(), color="#1f77b4", linestyle="--",
               linewidth=1.5, label=f"Median: {pnl_pcts.median():.3f}%")
    ax.axvline(pnl_pcts.mean(), color="#ff7f0e", linestyle="--",
               linewidth=1.5, label=f"Mean: {pnl_pcts.mean():.3f}%")
    stats_text = (f"Median win: ${sig[sig['net_pnl']>0]['net_pnl'].median():.1f}\n"
                  f"Mean win: ${sig[sig['net_pnl']>0]['net_pnl'].mean():.1f}\n"
                  f"Median notional: ${sig['notional'].median():,.0f}")
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax.set_xlabel("Net P/L (%)")
    ax.set_ylabel("Count")
    ax.set_title("Signal Exit P/L Distribution")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    _save(fig, "diag_pnl_dist.png")

    # entry |z| vs P/L
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter_z = sig["abs_entry_z"].values
    scatter_pnl = (sig["pnl_pct"] * 100).values
    colors_5 = ["#2ca02c" if p > 0 else "#d62728" for p in scatter_pnl]
    ax.scatter(scatter_z, scatter_pnl, c=colors_5, alpha=0.3, s=10,
               edgecolors="none")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.7)
    z_bins = np.linspace(scatter_z.min(), scatter_z.max(), 10)
    bin_idx = np.digitize(scatter_z, z_bins)
    for b in range(1, len(z_bins)):
        mask = bin_idx == b
        if mask.sum() >= 3:
            ax.scatter(z_bins[b-1] + (z_bins[1]-z_bins[0])/2,
                       np.mean(scatter_pnl[mask]),
                       color="#1f77b4", s=60, zorder=5, marker="D",
                       edgecolors="black", linewidth=0.5)
    ax.set_xlabel("|Entry Z-Score|")
    ax.set_ylabel("Net P/L (%)")
    ax.set_title("Entry |Z| vs P/L")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    _save(fig, "diag_z_vs_pnl.png")

    # holding period vs P/L
    fig, ax = plt.subplots(figsize=(10, 6))
    for reason in all_exits["exit_reason"].unique():
        subset = all_exits[all_exits["exit_reason"] == reason]
        color = "#2ca02c" if reason == "signal" else "#d62728"
        alpha = 0.15 if reason == "signal" else 0.4
        ax.scatter(subset["bars_held"], subset["pnl_pct"] * 100,
                   alpha=alpha, s=8, color=color, label=reason,
                   edgecolors="none")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.7)
    ax.set_xlabel("Bars Held")
    ax.set_ylabel("P/L (%)")
    ax.set_title("Holding Period vs P/L")
    ax.legend(fontsize=8, framealpha=0.7, ncol=2, loc="lower left")
    ax.grid(True, alpha=0.15)
    sig_pnl_range = (sig["pnl_pct"] * 100).quantile([0.01, 0.99])
    all_pnl_range = (all_exits["pnl_pct"] * 100).quantile([0.01, 0.99])
    y_lo = min(sig_pnl_range.iloc[0], all_pnl_range.iloc[0]) * 1.2
    y_hi = max(sig_pnl_range.iloc[1], all_pnl_range.iloc[1]) * 1.2
    ax.set_ylim(y_lo, y_hi)
    plt.tight_layout()
    _save(fig, "diag_hold_vs_pnl.png")


# best/worst pair spread charts (multi-panel per pair)

def _plot_top_bottom_spreads(pair_stats, spread_data, trade_log, outdir,
                              n_pairs: int = 3, pad_days: int = 14):
    """Multi-panel spread diagnostics for top/bottom N pairs by PnL."""
    pair_stats = pair_stats.sort_values('total_pnl', ascending=False)
    top_pairs = pair_stats.head(n_pairs)["pair_id"].tolist()
    bottom_pairs = pair_stats.tail(n_pairs)["pair_id"].tolist()

    pad = pd.Timedelta(days=pad_days)

    for label, pair_ids in [("best", top_pairs), ("worst", bottom_pairs)]:
        for pid in pair_ids:
            if pid not in spread_data:
                continue
            sd = spread_data[pid]
            if len(sd.get("z_scores", [])) < 10:
                continue

            timestamps = np.array(sd["timestamps"])
            spreads = np.array(sd["spreads"])
            z_scores = np.array(sd["z_scores"])
            bar_indices = sd.get("bar_indices", [])
            betas = np.array(sd.get("betas", []))
            implied_rhos = np.array(sd.get("implied_rhos", []))
            prices_a = np.array(sd.get("prices_a", []))
            prices_b = np.array(sd.get("prices_b", []))
            sym_a = sd.get("sym_a", pid.split("__")[0] if "__" in pid else "A")
            sym_b = sd.get("sym_b", pid.split("__")[1] if "__" in pid else "B")

            has_prices = len(prices_a) == len(timestamps) and np.any(np.isfinite(prices_a))
            has_betas = len(betas) == len(timestamps) and np.any(np.isfinite(betas))
            has_rhos = len(implied_rhos) == len(timestamps) and np.any(np.isfinite(implied_rhos))

            bar_to_local = {}
            for local_idx, bidx in enumerate(bar_indices):
                bar_to_local[bidx] = local_idx

            pair_trades = None
            if trade_log is not None:
                pair_trades = trade_log[trade_log["pair_id"] == pid]

            if pair_trades is not None and len(pair_trades) > 0:
                trade_locals = []
                for _, tr in pair_trades.iterrows():
                    for bar_col in ["entry_bar", "exit_bar"]:
                        b = tr.get(bar_col)
                        loc = bar_to_local.get(b)
                        if loc is not None and loc < len(timestamps):
                            trade_locals.append(loc)

                if trade_locals:
                    t_min = timestamps[min(trade_locals)] - pad
                    t_max = timestamps[max(trade_locals)] + pad
                else:
                    t_min, t_max = timestamps[0], timestamps[-1]
            else:
                t_min, t_max = timestamps[0], timestamps[-1]

            mask = (timestamps >= t_min) & (timestamps <= t_max)
            ts_w = timestamps[mask]
            sp_w = spreads[mask]
            zs_w = z_scores[mask]
            if len(ts_w) < 10:
                continue

            n_panels = 3
            if has_betas:
                n_panels += 1
            if has_rhos:
                n_panels += 1

            ratios = [1.2]
            ratios.append(1.0)
            ratios.append(1.0)
            if has_betas:
                ratios.append(0.7)
            if has_rhos:
                ratios.append(0.7)

            fig, axes = plt.subplots(
                n_panels, 1, figsize=(16, 3.2 * n_panels),
                sharex=True,
                gridspec_kw={"height_ratios": ratios})
            if n_panels == 1:
                axes = [axes]

            ax_idx = 0

            # prices panel
            if has_prices:
                pa_w = prices_a[mask]
                pb_w = prices_b[mask]
                ax_pa = axes[ax_idx]
                ln1 = ax_pa.plot(ts_w, pa_w, color="#1f77b4",
                                  linewidth=0.9, alpha=0.85, label=sym_a)
                ax_pa.set_ylabel(sym_a, color="#1f77b4", fontsize=10)
                ax_pa.tick_params(axis="y", labelcolor="#1f77b4")

                ax_pb = ax_pa.twinx()
                ln2 = ax_pb.plot(ts_w, pb_w, color="#ff7f0e",
                                  linewidth=0.9, alpha=0.85, label=sym_b)
                ax_pb.set_ylabel(sym_b, color="#ff7f0e", fontsize=10)
                ax_pb.tick_params(axis="y", labelcolor="#ff7f0e")

                lines = ln1 + ln2
                labs = [l.get_label() for l in lines]
                ax_pa.legend(lines, labs, loc="upper left", fontsize=8,
                             framealpha=0.7)
                ax_pa.set_title(f"{pid}  ({label})", fontsize=13,
                                fontweight="bold")
                ax_pa.grid(True, alpha=0.15)
            else:
                axes[ax_idx].set_title(f"{pid}  ({label})",
                                       fontsize=13, fontweight="bold")
                axes[ax_idx].text(0.5, 0.5, "Price data not available",
                                   ha="center", va="center",
                                   transform=axes[ax_idx].transAxes)
            ax_idx += 1

            # spread panel
            ax_sp = axes[ax_idx]
            ax_sp.plot(ts_w, sp_w, color="#333333",
                        linewidth=0.8, alpha=0.7)
            ax_sp.axhline(np.nanmean(sp_w), color="#1f77b4",
                           linestyle="--", linewidth=0.7, alpha=0.4,
                           label="mean")
            ax_sp.set_ylabel("Spread", fontsize=10)
            ax_sp.legend(loc="upper left", fontsize=8, framealpha=0.7)
            ax_sp.grid(True, alpha=0.15)
            ax_idx += 1

            # z-score panel with trade markers
            ax_z = axes[ax_idx]
            ax_z.plot(ts_w, zs_w, color="#333333",
                       linewidth=0.8, alpha=0.7)
            ax_z.axhline(0, color="gray", linestyle="-", linewidth=0.5)
            for thresh in [2.0, -2.0]:
                ax_z.axhline(thresh, color="#ff7f0e", linestyle="--",
                              linewidth=0.7, alpha=0.5,
                              label="entry" if thresh > 0 else None)
            for thresh in [6, -6]:
                ax_z.axhline(thresh, color="#d62728", linestyle="--",
                              linewidth=0.7, alpha=0.5,
                              label="hard stop" if thresh > 0 else None)
            ax_z.fill_between(ts_w, -0.5, 0.5,
                               color="#2ca02c", alpha=0.05)
            ax_z.set_ylabel("Z-Score", fontsize=10)

            if pair_trades is not None:
                for _, tr in pair_trades.iterrows():
                    net = tr.get("net_pnl", 0)
                    color = "#2ca02c" if net > 0 else "#d62728"
                    reason = tr.get("exit_reason", "")

                    entry_local = bar_to_local.get(tr.get("entry_bar"))
                    if entry_local is not None and entry_local < len(timestamps):
                        t = timestamps[entry_local]
                        if t_min <= t <= t_max:
                            ax_z.axvline(t, color=color, alpha=0.3,
                                          linewidth=0.7)
                            exit_local = bar_to_local.get(tr.get("exit_bar"))
                            if exit_local is not None and exit_local < len(timestamps):
                                t_exit = timestamps[exit_local]
                                if t_min <= t_exit <= t_max:
                                    ax_sp.axvspan(t, t_exit, color=color,
                                                   alpha=0.06)

                    exit_local = bar_to_local.get(tr.get("exit_bar"))
                    if exit_local is not None and exit_local < len(timestamps):
                        t = timestamps[exit_local]
                        if t_min <= t <= t_max:
                            ax_z.axvline(t, color=color, alpha=0.3,
                                          linewidth=0.7, linestyle="--")
                            y_pos = zs_w[np.searchsorted(
                                ts_w, t, side="right") - 1] if len(zs_w) > 0 else 0
                            ax_z.annotate(
                                reason, xy=(t, y_pos),
                                xytext=(5, 8), textcoords="offset points",
                                fontsize=6, color="#555555", alpha=0.8,
                                rotation=30)

            ax_z.legend(loc="upper left", fontsize=8, framealpha=0.7)
            ax_z.grid(True, alpha=0.15)
            ax_idx += 1

            # beta panel
            if has_betas:
                beta_w = betas[mask]
                ax_beta = axes[ax_idx]
                ax_beta.plot(ts_w, beta_w, color="#9467bd",
                              linewidth=0.9, alpha=0.8)
                ax_beta.axhline(np.nanmean(beta_w), color="#9467bd",
                                 linestyle="--", linewidth=0.6, alpha=0.4)
                bm = np.nanmean(beta_w)
                ax_beta.axhspan(bm * 0.9, bm * 1.1,
                                 color="#9467bd", alpha=0.04)
                ax_beta.set_ylabel("Beta (hedge ratio)", fontsize=10)
                ax_beta.grid(True, alpha=0.15)
                ax_idx += 1

            # implied rho panel
            if has_rhos:
                rho_w = implied_rhos[mask]
                ax_rho = axes[ax_idx]
                ax_rho.plot(ts_w, rho_w, color="#e377c2",
                             linewidth=0.9, alpha=0.8)
                ax_rho.axhline(0.4, color="#ff7f0e", linestyle="--",
                                linewidth=0.6, alpha=0.4,
                                label="entry floor (0.4)")
                ax_rho.axhline(0, color="gray", linestyle="-",
                                linewidth=0.5, alpha=0.3)
                ax_rho.set_ylabel("Implied ρ", fontsize=10)
                ax_rho.set_ylim(-0.5, 1.1)
                ax_rho.legend(loc="upper left", fontsize=8, framealpha=0.7)
                ax_rho.grid(True, alpha=0.15)
                ax_idx += 1

            axes[-1].set_xlabel("Time")
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            fig.autofmt_xdate(rotation=30)

            plt.tight_layout()
            safe_pid = pid.replace("/", "_").replace("__", "_")
            fig.savefig(outdir / f"spread_{label}_{safe_pid}.png",
                        dpi=150, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            plt.close(fig)
            print(f"  [report] Spread chart ({label}): {safe_pid}")


# trade log printer (notebook use)

def print_trade_log(trade_log: pd.DataFrame, n: int = 20,
                    show_conditions: bool = True):
    """Pretty-print recent trades with entry/exit conditions."""
    if trade_log is None or len(trade_log) == 0:
        print("  No trades.")
        return

    df = trade_log.copy()
    if n > 0:
        df = df.tail(n)

    core_cols = [
        "pair_id", "direction", "exit_reason", "bars_held",
        "entry_z", "exit_z", "beta_entry",
        "notional", "net_pnl", "pnl_pct",
    ]
    cond_cols = [
        "entry_spread", "exit_spread",
        "beta_exit", "entry_beta_P11", "exit_beta_P11",
        "entry_confidence", "half_life",
        "entry_implied_rho", "entry_regime_conf",
    ]

    cols = [c for c in core_cols if c in df.columns]
    if show_conditions:
        cols += [c for c in cond_cols if c in df.columns]

    print(f"\n  Trade Log ({len(df)} trades shown):")
    print(f"  {'-'*100}")
    print(df[cols].to_string(index=False, max_colwidth=25))
    print()
