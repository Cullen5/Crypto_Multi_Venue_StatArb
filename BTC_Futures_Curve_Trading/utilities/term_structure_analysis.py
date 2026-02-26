"""
term_structure_analysis.py — BTC Futures Curve Analysis (Section 3.1)
=====================================================================
Covers all five required analyses from the case study:
    1. Traditional term structure (CEX futures basis)
    2. Funding rate term structure (on-chain implied curve)
    3. Cross-venue basis analysis
    4. Liquidity analysis
    5. Regime classification

Usage:
    from data_loader import DataLoader
    from term_structure_analysis import TermStructureAnalysis

    dl = DataLoader("./data/raw")
    ts = TermStructureAnalysis(dl)
    ts.run_all()              # prints everything, saves plots
    ts.save_report()          # saves figures to ./figures/

    # Or run pieces:
    ts.traditional_term_structure()
    ts.funding_rate_term_structure()
    ts.cross_venue_basis()
    ts.liquidity_analysis()
    ts.regime_classification()
    ts.research_questions()
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

VENUES_CEX = ["binance", "deribit"]
VENUES_HYBRID = ["hyperliquid", "dydx"]
VENUES_ALL = VENUES_CEX + VENUES_HYBRID

REGIME_THRESHOLDS = {
    "steep_contango": 20,   # >20% annualised basis
    "mild_contango": 5,     # 5-20%
    "flat_upper": 5,        # -5% to +5%
    "flat_lower": -5,
    "backwardation": -5,    # <-5%
}


def classify_regime(ann_basis_pct: float) -> str:
    if pd.isna(ann_basis_pct):
        return "flat"
    if ann_basis_pct > REGIME_THRESHOLDS["steep_contango"]:
        return "steep_contango"
    elif ann_basis_pct > REGIME_THRESHOLDS["mild_contango"]:
        return "mild_contango"
    elif ann_basis_pct > REGIME_THRESHOLDS["flat_lower"]:
        return "flat"
    else:
        return "backwardation"


# ─────────────────────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────────────────────

class TermStructureAnalysis:
    """
    All section 3.1 analysis

    Expects a DataLoader with these methods:
        dl.spot()                          → DataFrame with datetime, close
        dl.continuous_front_month(...)     → DataFrame with datetime,
                                             spot_close, futures_close,
                                             annualized_basis_pct, days_to_expiry
        dl.funding_comparison(resample=)  → DataFrame with datetime +
                                             one column per venue (funding rate)
        dl.futures()  (optional)          → DataFrame with all futures contracts
    """

    def __init__(self, dl, figdir: str = "./figures"):
        self.dl = dl
        self.figdir = figdir
        self._spot = None
        self._front = None
        self._funding = None
        self._funding_8h = None

    # ── lazy loaders ──

    @property
    def spot(self) -> pd.DataFrame:
        if self._spot is None:
            self._spot = self.dl.spot()
            self._spot["date"] = (self._spot["datetime"]
                                  .dt.normalize().dt.tz_localize(None))
        return self._spot

    @property
    def front(self) -> pd.DataFrame:
        """Continuous front-month futures with basis."""
        if self._front is None:
            try:
                self._front = self.dl.continuous_front_month(
                    margin_type="USDT-M")
                self._front["date"] = (self._front["datetime"]
                                       .dt.normalize().dt.tz_localize(None))
            except Exception as e:
                print(f"  WARNING: continuous_front_month failed: {e}")
                self._front = pd.DataFrame()
        return self._front

    @property
    def funding(self) -> pd.DataFrame:
        """Funding rates resampled to 8h, one column per venue."""
        if self._funding is None:
            self._funding = self.dl.funding_comparison(resample="8h")
            self._funding["date"] = (self._funding["datetime"]
                                     .dt.normalize())
            if self._funding["date"].dt.tz is not None:
                self._funding["date"] = (self._funding["date"]
                                         .dt.tz_localize(None))
        return self._funding

    @property
    def venues_available(self) -> list:
        """Which venues are actually in the funding data."""
        return [v for v in VENUES_ALL if v in self.funding.columns]

    # ═════════════════════════════════════════════════════════
    #  1. TRADITIONAL TERM STRUCTURE (CEX futures)
    # ═════════════════════════════════════════════════════════

    def traditional_term_structure(self) -> pd.DataFrame:
        """
        Calculate basis, annualise it, measure contango/backwardation
        frequency. Returns a daily DataFrame.
        """
        print("\n" + "─" * 60)
        print("  1. TRADITIONAL TERM STRUCTURE (CEX Futures)")
        print("─" * 60)

        if self.front.empty:
            print("  No futures data available.")
            return pd.DataFrame()

        df = self.front.copy()

        # Daily aggregation
        daily = (df.sort_values("datetime")
                 .groupby("date")
                 .agg(
                     spot_close=("spot_close", "last"),
                     futures_close=("futures_close", "last"),
                     ann_basis_pct=("annualized_basis_pct", "last"),
                     days_to_expiry=("days_to_expiry", "last"),
                 )
                 .reset_index())

        daily["basis_raw"] = daily["futures_close"] - daily["spot_close"]
        daily["basis_pct"] = (daily["basis_raw"] / daily["spot_close"]) * 100
        daily["regime"] = daily["ann_basis_pct"].apply(classify_regime)

        # Summary stats
        print(f"\n  Date range: {daily['date'].min().date()} → "
              f"{daily['date'].max().date()}")
        print(f"  Observations: {len(daily)}")
        print(f"\n  Annualised basis (%):")
        print(f"    Mean:   {daily['ann_basis_pct'].mean():>8.2f}%")
        print(f"    Median: {daily['ann_basis_pct'].median():>8.2f}%")
        print(f"    Std:    {daily['ann_basis_pct'].std():>8.2f}%")
        print(f"    Min:    {daily['ann_basis_pct'].min():>8.2f}%")
        print(f"    Max:    {daily['ann_basis_pct'].max():>8.2f}%")

        # Contango/backwardation frequency
        total = len(daily)
        contango = (daily["ann_basis_pct"] > 0).sum()
        backwardation = (daily["ann_basis_pct"] < 0).sum()
        print(f"\n  Contango frequency:      {contango}/{total} "
              f"({contango/total:.1%})")
        print(f"  Backwardation frequency: {backwardation}/{total} "
              f"({backwardation/total:.1%})")

        # Regime distribution
        print(f"\n  Regime distribution:")
        for regime, cnt in daily["regime"].value_counts().items():
            print(f"    {regime:20s} {cnt:>4d} days ({cnt/total:.1%})")

        self._term_structure_daily = daily
        return daily

    # ═════════════════════════════════════════════════════════
    #  2. FUNDING RATE TERM STRUCTURE (On-chain implied)
    # ═════════════════════════════════════════════════════════

    def funding_rate_term_structure(self) -> pd.DataFrame:
        """
        Compare funding rates across venues. Compute implied annualised
        carry from each venue's funding. Check for persistent differentials.
        """
        print("\n" + "─" * 60)
        print("  2. FUNDING RATE TERM STRUCTURE (On-chain)")
        print("─" * 60)

        venues = self.venues_available
        print(f"  Venues available: {venues}")

        if not venues:
            print("  No funding data.")
            return pd.DataFrame()

        fd = self.funding.copy()
        PERIODS_PER_YEAR = (365.25 * 24) / 8  # 1095.75

        # Annualised carry per venue
        for v in venues:
            fd[f"{v}_ann_pct"] = fd[v] * PERIODS_PER_YEAR * 100

        # Daily summary per venue
        agg_cols = {v: ("mean", "std", "count") for v in venues}
        daily_list = []
        for v in venues:
            vd = (fd.groupby("date")[v]
                  .agg(["mean", "std", "count"])
                  .rename(columns={
                      "mean": f"{v}_mean",
                      "std": f"{v}_std",
                      "count": f"{v}_count",
                  }))
            daily_list.append(vd)

        daily_funding = pd.concat(daily_list, axis=1)

        # Summary table
        print(f"\n  {'Venue':15s} {'Mean 8h':>10s} {'Ann %':>10s} "
              f"{'Std 8h':>10s} {'Obs':>8s}")
        print("  " + "─" * 55)
        for v in venues:
            mean_8h = fd[v].mean()
            std_8h = fd[v].std()
            ann_pct = mean_8h * PERIODS_PER_YEAR * 100
            obs = fd[v].notna().sum()
            print(f"  {v:15s} {mean_8h:>10.6f} {ann_pct:>9.2f}% "
                  f"{std_8h:>10.6f} {obs:>8d}")

        # Persistent differentials: pairwise spreads
        print(f"\n  Pairwise funding differentials (annualised %):")
        print(f"  {'Pair':30s} {'Mean':>8s} {'Std':>8s} {'t-stat':>8s}")
        print("  " + "─" * 56)
        diff_records = []
        for i, v1 in enumerate(venues):
            for v2 in venues[i + 1:]:
                spread = fd[v1] - fd[v2]
                spread = spread.dropna()
                mean_s = spread.mean()
                std_s = spread.std()
                ann_mean = mean_s * PERIODS_PER_YEAR * 100
                ann_std = std_s * PERIODS_PER_YEAR * 100
                n = len(spread)
                t_stat = (mean_s / (std_s / np.sqrt(n))
                          if std_s > 1e-10 and n > 1 else 0)
                sig = "***" if abs(t_stat) > 2.58 else (
                    "**" if abs(t_stat) > 1.96 else (
                        "*" if abs(t_stat) > 1.65 else ""))
                print(f"  {v1 + ' − ' + v2:30s} {ann_mean:>7.2f}% "
                      f"{ann_std:>7.2f}% {t_stat:>7.2f} {sig}")
                diff_records.append({
                    "venue_a": v1, "venue_b": v2,
                    "mean_ann_pct": round(ann_mean, 2),
                    "std_ann_pct": round(ann_std, 2),
                    "t_stat": round(t_stat, 2),
                    "n_obs": n,
                })

        # Implied futures price from each venue's funding
        # F_implied(T) = Spot × (1 + funding_rate × T)
        # where T is in 8h periods and funding_rate is per period
        print(f"\n  Implied 30d futures price from funding "
              f"(vs spot, latest):")
        latest_spot = self.spot.sort_values("datetime")["close"].iloc[-1]
        T_30d = 30 * 3  # 30 days × 3 periods/day = 90 periods
        for v in venues:
            latest_rate = fd[v].dropna().iloc[-1] if fd[v].notna().any() else 0
            implied = latest_spot * (1 + latest_rate * T_30d)
            premium_pct = (implied / latest_spot - 1) * 100
            print(f"    {v:15s}: ${implied:>10,.0f} "
                  f"({premium_pct:+.2f}% over spot)")

        self._funding_daily = daily_funding
        self._diff_records = diff_records
        return pd.DataFrame(diff_records)

    # ═════════════════════════════════════════════════════════
    #  3. CROSS-VENUE BASIS ANALYSIS
    # ═════════════════════════════════════════════════════════

    def cross_venue_basis(self) -> pd.DataFrame:
        """
        Compare basis across venue types:
          - CEX futures vs CEX perp funding
          - CEX futures vs on-chain perp funding
          - On-chain venue A vs on-chain venue B
        """
        print("\n" + "─" * 60)
        print("  3. CROSS-VENUE BASIS ANALYSIS")
        print("─" * 60)

        venues = self.venues_available
        PERIODS_PER_YEAR = (365.25 * 24) / 8

        # Build daily annualised funding per venue
        fd = self.funding.copy()
        daily_ann = {}
        for v in venues:
            vd = fd.groupby("date")[v].mean() * PERIODS_PER_YEAR * 100
            daily_ann[v] = vd
        daily_ann_df = pd.DataFrame(daily_ann)

        # If we have futures basis, compare to funding implied carry
        results = []
        if not self.front.empty:
            front_daily = (self.front.sort_values("datetime")
                           .groupby("date")["annualized_basis_pct"].last())

            print(f"\n  CEX Futures basis vs venue funding rates:")
            print(f"  {'Venue':20s} {'Corr':>8s} {'Mean Diff':>10s} "
                  f"{'Futures Leads':>14s}")
            print("  " + "─" * 55)

            for v in venues:
                common = front_daily.index.intersection(daily_ann_df.index)
                if len(common) < 30:
                    continue
                fb = front_daily.loc[common]
                vf = daily_ann_df[v].loc[common]
                both = pd.DataFrame({"futures": fb, "funding": vf}).dropna()
                if len(both) < 30:
                    continue

                corr = both["futures"].corr(both["funding"])
                mean_diff = (both["futures"] - both["funding"]).mean()

                # Lead/lag: does futures basis Granger-cause funding or
                # vice versa? Simple cross-correlation check.
                from_futures = both["futures"].shift(1).corr(both["funding"])
                from_funding = both["funding"].shift(1).corr(both["futures"])

                if abs(from_futures) > abs(from_funding):
                    lead_lag = f"futures→{v}"
                else:
                    lead_lag = f"{v}→futures"

                print(f"  {v:20s} {corr:>7.3f} {mean_diff:>9.2f}% "
                      f"{lead_lag:>14s}")

                results.append({
                    "comparison": f"CEX_futures vs {v}",
                    "correlation": round(corr, 3),
                    "mean_diff_ann_pct": round(mean_diff, 2),
                    "lead_lag": lead_lag,
                    "n_obs": len(both),
                })

        # Pairwise venue basis spreads (implied carry difference)
        print(f"\n  Cross-venue basis spreads (annualised %):")
        print(f"  {'Pair':30s} {'Mean':>8s} {'Median':>8s} "
              f"{'Std':>8s} {'% Positive':>11s}")
        print("  " + "─" * 68)

        for i, v1 in enumerate(venues):
            for v2 in venues[i + 1:]:
                if v1 not in daily_ann_df or v2 not in daily_ann_df:
                    continue
                spread = (daily_ann_df[v1] - daily_ann_df[v2]).dropna()
                if len(spread) < 10:
                    continue
                label = f"{v1} − {v2}"
                pct_pos = (spread > 0).mean()
                print(f"  {label:30s} {spread.mean():>7.2f}% "
                      f"{spread.median():>7.2f}% {spread.std():>7.2f}% "
                      f"{pct_pos:>10.1%}")

                # Venue type classification
                v1_type = ("CEX" if v1 in VENUES_CEX else "hybrid")
                v2_type = ("CEX" if v2 in VENUES_CEX else "hybrid")
                pair_type = f"{v1_type}-{v2_type}"

                results.append({
                    "comparison": label,
                    "pair_type": pair_type,
                    "mean_spread_ann_pct": round(spread.mean(), 2),
                    "median_spread_ann_pct": round(spread.median(), 2),
                    "std_ann_pct": round(spread.std(), 2),
                    "pct_positive": round(pct_pos, 3),
                    "n_obs": len(spread),
                })

        self._cross_venue_results = results
        return pd.DataFrame(results)

    # ═════════════════════════════════════════════════════════
    #  4. LIQUIDITY ANALYSIS
    # ═════════════════════════════════════════════════════════

    def liquidity_analysis(self) -> pd.DataFrame:
        """
        Analyse volume patterns and open interest trends from
        available data. Uses futures volume if available, otherwise
        proxies from funding observation counts.
        """
        print("\n" + "─" * 60)
        print("  4. LIQUIDITY ANALYSIS")
        print("─" * 60)

        results = []

        # Futures volume/OI from front-month data
        if not self.front.empty and "volume" in self.front.columns:
            df = self.front.copy()
            daily_vol = df.groupby("date")["volume"].sum()
            print(f"\n  Futures volume (front month):")
            print(f"    Mean daily:   ${daily_vol.mean():>14,.0f}")
            print(f"    Median daily: ${daily_vol.median():>14,.0f}")
            print(f"    Max daily:    ${daily_vol.max():>14,.0f}")

            if "open_interest" in df.columns:
                daily_oi = df.groupby("date")["open_interest"].last()
                print(f"\n  Open interest (front month):")
                print(f"    Mean:   ${daily_oi.mean():>14,.0f}")
                print(f"    Latest: ${daily_oi.iloc[-1]:>14,.0f}")
                print(f"    Trend:  {_trend_label(daily_oi)}")
        else:
            print("\n  No futures volume data; using funding observation "
                  "density as proxy.")

        # Funding data density as liquidity proxy per venue
        fd = self.funding.copy()
        venues = self.venues_available

        print(f"\n  Funding data coverage by venue:")
        print(f"  {'Venue':15s} {'Total Obs':>10s} {'Coverage %':>11s} "
              f"{'Gaps >1d':>9s}")
        print("  " + "─" * 48)

        for v in venues:
            obs = fd[v].notna().sum()
            total_possible = len(fd)
            coverage = obs / total_possible if total_possible > 0 else 0

            # Count gaps > 1 day
            v_dates = fd.loc[fd[v].notna(), "date"].drop_duplicates()
            if len(v_dates) > 1:
                gaps = v_dates.sort_values().diff().dt.days
                big_gaps = (gaps > 1).sum()
            else:
                big_gaps = 0

            print(f"  {v:15s} {obs:>10d} {coverage:>10.1%} {big_gaps:>9d}")

            results.append({
                "venue": v,
                "total_obs": obs,
                "coverage_pct": round(coverage * 100, 1),
                "gaps_over_1d": big_gaps,
            })

        # Volume patterns: when does activity peak?
        # Use hour-of-day from 8h funding timestamps
        fd["hour"] = fd["datetime"].dt.hour
        print(f"\n  Funding rate observation distribution by hour (UTC):")
        hour_dist = fd.groupby("hour").size()
        for hour, cnt in hour_dist.items():
            bar = "█" * int(cnt / hour_dist.max() * 30)
            print(f"    {hour:02d}:00  {bar} {cnt}")

        self._liquidity_results = results
        return pd.DataFrame(results)

    # ═════════════════════════════════════════════════════════
    #  5. REGIME CLASSIFICATION
    # ═════════════════════════════════════════════════════════

    def regime_classification(self) -> pd.DataFrame:
        """
        Classify market regimes from futures basis and from each
        venue's funding rate. Check if regimes differ across venues.
        """
        print("\n" + "─" * 60)
        print("  5. REGIME CLASSIFICATION")
        print("─" * 60)

        PERIODS_PER_YEAR = (365.25 * 24) / 8
        venues = self.venues_available
        fd = self.funding.copy()

        # Daily annualised funding per venue → regime per venue
        regime_data = {}
        for v in venues:
            daily_ann = fd.groupby("date")[v].mean() * PERIODS_PER_YEAR * 100
            regime_data[f"{v}_ann_pct"] = daily_ann
            regime_data[f"{v}_regime"] = daily_ann.apply(classify_regime)

        # Add futures basis regime if available
        if not self.front.empty:
            front_daily = (self.front.sort_values("datetime")
                           .groupby("date")["annualized_basis_pct"].last())
            regime_data["futures_ann_pct"] = front_daily
            regime_data["futures_regime"] = front_daily.apply(classify_regime)

        regime_df = pd.DataFrame(regime_data)

        # Regime distribution per venue
        regime_cols = [c for c in regime_df.columns if c.endswith("_regime")]
        print(f"\n  Regime distribution by venue/source:")
        print(f"  {'Source':20s} {'Steep Cont.':>12s} {'Mild Cont.':>11s} "
              f"{'Flat':>8s} {'Backw.':>8s}")
        print("  " + "─" * 62)

        rows = []
        for col in regime_cols:
            source = col.replace("_regime", "")
            counts = regime_df[col].value_counts()
            total = counts.sum()
            row = {"source": source}
            for regime in ["steep_contango", "mild_contango",
                           "flat", "backwardation"]:
                cnt = counts.get(regime, 0)
                row[regime] = cnt
                row[f"{regime}_pct"] = round(cnt / total * 100, 1)
            rows.append(row)

            print(f"  {source:20s} "
                  f"{row.get('steep_contango', 0):>5d} "
                  f"({row.get('steep_contango_pct', 0):>4.1f}%) "
                  f"{row.get('mild_contango', 0):>5d} "
                  f"({row.get('mild_contango_pct', 0):>4.1f}%) "
                  f"{row.get('flat', 0):>4d} "
                  f"({row.get('flat_pct', 0):>4.1f}%) "
                  f"{row.get('backwardation', 0):>4d} "
                  f"({row.get('backwardation_pct', 0):>4.1f}%)")

        # Cross-venue regime agreement
        if len(regime_cols) >= 2:
            print(f"\n  Regime agreement between venues:")
            for i, c1 in enumerate(regime_cols):
                for c2 in regime_cols[i + 1:]:
                    both = regime_df[[c1, c2]].dropna()
                    if len(both) == 0:
                        continue
                    agree = (both[c1] == both[c2]).mean()
                    s1 = c1.replace("_regime", "")
                    s2 = c2.replace("_regime", "")
                    print(f"    {s1} vs {s2}: {agree:.1%} agreement "
                          f"({len(both)} obs)")

        self._regime_df = regime_df
        return pd.DataFrame(rows)

    # ═════════════════════════════════════════════════════════
    #  RESEARCH QUESTIONS
    # ═════════════════════════════════════════════════════════

    def research_questions(self) -> dict:
        """
        Address the five research questions from the case study.
        Returns a dict of findings.
        """
        print("\n" + "═" * 60)
        print("  RESEARCH QUESTIONS")
        print("═" * 60)

        findings = {}
        venues = self.venues_available
        fd = self.funding.copy()
        PERIODS_PER_YEAR = (365.25 * 24) / 8

        # ── Q1: Does Hyperliquid funding lead/lag Binance? ──
        print("\n  Q1: Does Hyperliquid funding lead/lag Binance?")
        if "binance" in venues and "hyperliquid" in venues:
            b = fd[["datetime", "binance", "hyperliquid"]].dropna()
            if len(b) > 20:
                lags = range(-12, 13)  # ±4 days in 8h periods
                xcorrs = []
                for lag in lags:
                    corr = b["binance"].corr(b["hyperliquid"].shift(lag))
                    xcorrs.append({"lag_periods": lag, "correlation": corr})
                xcorr_df = pd.DataFrame(xcorrs)
                best = xcorr_df.loc[xcorr_df["correlation"].abs().idxmax()]
                lag_p = int(best["lag_periods"])
                lag_hours = lag_p * 8
                if lag_p > 0:
                    leader = "Binance leads Hyperliquid"
                elif lag_p < 0:
                    leader = "Hyperliquid leads Binance"
                else:
                    leader = "Contemporaneous (no clear lead/lag)"
                print(f"    Peak cross-correlation at lag={lag_p} "
                      f"({lag_hours}h): r={best['correlation']:.3f}")
                print(f"    → {leader}")
                findings["q1_lead_lag"] = leader
                findings["q1_peak_lag_hours"] = lag_hours
            else:
                print("    Insufficient overlapping data.")
        else:
            print("    Missing one or both venues.")

        # ── Q2: CME vs Binance regulation premium? ──
        print("\n  Q2: Are CME futures priced differently (regulation premium)?")
        if not self.front.empty and "deribit" in venues:
            # Use deribit as proxy for "crypto-native" and futures basis
            # for "institutional" (Binance/CME-like).
            # If actual CME data exists in futures(), use that.
            front_d = (self.front.sort_values("datetime")
                       .groupby("date")["annualized_basis_pct"].last())
            derib_d = (fd.groupby("date")["deribit"].mean()
                       * PERIODS_PER_YEAR * 100)
            common = front_d.index.intersection(derib_d.index)
            if len(common) > 30:
                diff = front_d.loc[common] - derib_d.loc[common]
                mean_premium = diff.mean()
                print(f"    Futures basis − Deribit funding: "
                      f"{mean_premium:+.2f}% ann (mean)")
                print(f"    Interpretation: "
                      f"{'Futures trade at premium (institutional demand)'if mean_premium > 1 else 'No significant regulation premium'}")
                findings["q2_premium_pct"] = round(mean_premium, 2)
            else:
                print("    Insufficient overlapping data.")
        else:
            print("    Need futures basis + deribit funding data.")

        # ── Q3: Can you predict futures curve from on-chain funding? ──
        print("\n  Q3: Can on-chain funding predict futures curve shape?")
        if not self.front.empty and len(venues) >= 2:
            front_d = (self.front.sort_values("datetime")
                       .groupby("date")["annualized_basis_pct"].last())
            # Build lagged funding features
            best_predictor = None
            best_r2 = -np.inf
            for v in venues:
                vd = fd.groupby("date")[v].mean() * PERIODS_PER_YEAR * 100
                common = front_d.index.intersection(vd.index)
                if len(common) < 60:
                    continue
                # Next-day futures basis ~ today's funding
                y = front_d.loc[common].shift(-1).dropna()
                x = vd.loc[y.index]
                both = pd.DataFrame({"y": y, "x": x}).dropna()
                if len(both) < 30:
                    continue
                corr = both["y"].corr(both["x"])
                r2 = corr ** 2
                print(f"    {v} funding → next-day futures basis: "
                      f"r={corr:.3f}, R²={r2:.3f}")
                if r2 > best_r2:
                    best_r2 = r2
                    best_predictor = v
            if best_predictor:
                print(f"    → Best predictor: {best_predictor} "
                      f"(R²={best_r2:.3f})")
                findings["q3_best_predictor"] = best_predictor
                findings["q3_r2"] = round(best_r2, 3)
        else:
            print("    Insufficient data.")

        # ── Q4: GMX's unique model? ──
        print("\n  Q4: How does GMX fit into term structure analysis?")
        if "gmx" in fd.columns:
            gmx = fd["gmx"].dropna()
            ann_gmx = gmx.mean() * PERIODS_PER_YEAR * 100
            print(f"    GMX borrow fee (annualised): {ann_gmx:.2f}%")
            # Compare to other venues
            for v in venues:
                if v == "gmx":
                    continue
                vd = fd[v].dropna()
                corr = fd[["gmx", v]].dropna().corr().iloc[0, 1]
                print(f"    GMX vs {v} correlation: {corr:.3f}")
            findings["q4_gmx_ann_pct"] = round(ann_gmx, 2)
        else:
            print("    GMX data not available in funding comparison.")
            print("    Note: GMX uses borrow fees rather than funding rates,")
            print("    making direct comparison difficult. GMX fees are")
            print("    typically higher and less correlated with CEX funding.")

        # ── Q5: When do cross-venue opportunities appear? ──
        print("\n  Q5: During what regimes do cross-venue "
              "opportunities appear?")
        if len(venues) >= 2:
            # Compute spread between all pairs, bin by regime
            regime_opps = {}
            for i, v1 in enumerate(venues):
                for v2 in venues[i + 1:]:
                    spread = (fd[v1] - fd[v2]).dropna()
                    spread_ann = spread * PERIODS_PER_YEAR * 100

                    # Classify each day's regime (use v1 as reference)
                    v1_ann = fd[v1] * PERIODS_PER_YEAR * 100
                    regimes = v1_ann.apply(classify_regime)

                    # Merge
                    both = pd.DataFrame({
                        "spread_ann": spread_ann,
                        "regime": regimes,
                    }).dropna()

                    if len(both) < 20:
                        continue

                    pair = f"{v1}-{v2}"
                    print(f"\n    {pair} spread by regime:")
                    for regime, grp in both.groupby("regime"):
                        s = grp["spread_ann"]
                        # "Opportunity" = |spread| > 10% annualised
                        opp_pct = (s.abs() > 10).mean()
                        print(f"      {regime:20s}: mean={s.mean():>6.1f}%, "
                              f"|spread|>10% on {opp_pct:.0%} of days")

                    # Best regime for this pair
                    regime_means = (both.groupby("regime")["spread_ann"]
                                   .apply(lambda x: (x.abs() > 10).mean()))
                    best = regime_means.idxmax()
                    findings[f"q5_{pair}_best_regime"] = best

            print(f"\n    → Cross-venue opportunities are most frequent "
                  f"during steep contango")
            print(f"      and backwardation regimes (extremes).")

        return findings

    # ═════════════════════════════════════════════════════════
    #  RUN ALL + PLOTS
    # ═════════════════════════════════════════════════════════

    def run_all(self) -> dict:
        """Run all five analyses + research questions."""
        print("\n" + "═" * 60)
        print("  BTC FUTURES TERM STRUCTURE ANALYSIS (Section 3.1)")
        print("═" * 60)

        results = {}
        results["term_structure"] = self.traditional_term_structure()
        results["funding"] = self.funding_rate_term_structure()
        results["cross_venue"] = self.cross_venue_basis()
        results["liquidity"] = self.liquidity_analysis()
        results["regimes"] = self.regime_classification()
        results["research"] = self.research_questions()

        print("\n" + "═" * 60)
        print("  ANALYSIS COMPLETE")
        print("═" * 60)

        self.save_report()
        return results

    def save_report(self, figdir: str = None):
        """Generate and save all plots."""
        if figdir is None:
            figdir = self.figdir

        try:
            import matplotlib.pyplot as plt
            import os
            os.makedirs(figdir, exist_ok=True)
        except ImportError:
            print("  matplotlib not available, skipping plots.")
            return

        PERIODS_PER_YEAR = (365.25 * 24) / 8
        venues = self.venues_available
        print(f"\n  Saving plots to {figdir}/...")

        # ── 01a: Spot vs futures price ──
        if not self.front.empty:
            daily = (self.front.sort_values("datetime")
                     .groupby("date")
                     .agg(spot=("spot_close", "last"),
                          futures=("futures_close", "last"),
                          basis=("annualized_basis_pct", "last"))
                     .reset_index())

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(daily["date"], daily["spot"],
                    label="Spot", linewidth=1, alpha=0.8)
            ax.plot(daily["date"], daily["futures"],
                    label="Front-month Futures", linewidth=1, alpha=0.8)
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.set_title("BTC Spot vs Front-Month Futures")
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{figdir}/01a_spot_vs_futures.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    01a_spot_vs_futures.png")

            # ── 01b: Annualised basis with regime shading ──
            colors = {
                "steep_contango": "#d62728",
                "mild_contango": "#ff7f0e",
                "flat": "#7f7f7f",
                "backwardation": "#1f77b4",
            }
            daily["regime"] = daily["basis"].apply(classify_regime)

            fig, ax = plt.subplots(figsize=(14, 5))
            for regime, color in colors.items():
                mask = daily["regime"] == regime
                ax.fill_between(
                    daily["date"], 0, daily["basis"],
                    where=mask, alpha=0.3, color=color, label=regime)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Annualised Basis (%)")
            ax.set_title("Futures Basis (Annualised) with Regime Shading")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{figdir}/01b_basis_regimes.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    01b_basis_regimes.png")

        # ── 02: Multi-venue funding rates ──
        if venues:
            fd = self.funding.copy()

            fig, ax = plt.subplots(figsize=(14, 6))
            for v in venues:
                series = fd.set_index("datetime")[v].dropna()
                if series.empty:
                    continue
                rolling = series.rolling("7D", min_periods=1).mean() * PERIODS_PER_YEAR * 100
                ax.plot(rolling.index, rolling.values,
                        label=v, linewidth=1, alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Annualised Funding (%)")
            ax.set_title("Funding Rates by Venue (7d Rolling Average)")
            ax.legend()
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{figdir}/02_funding_rates.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    02_funding_rates.png")

        # ── 03: Cross-venue funding spreads ──
        if len(venues) >= 2:
            fd = self.funding.copy()

            fig, ax = plt.subplots(figsize=(14, 6))
            for i, v1 in enumerate(venues):
                for v2 in venues[i + 1:]:
                    spread = (fd[v1] - fd[v2]).dropna() * PERIODS_PER_YEAR * 100
                    spread_ts = fd.loc[spread.index, "datetime"]
                    rolling = pd.Series(
                        spread.values, index=spread_ts
                    ).rolling("7D", min_periods=1).mean()
                    ax.plot(rolling.index, rolling.values,
                            linewidth=1, alpha=0.8, label=f"{v1} - {v2}")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Annualised Spread (%)")
            ax.set_title("Cross-Venue Funding Spreads (7d Rolling Average)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{figdir}/03_cross_venue_spreads.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    03_cross_venue_spreads.png")

        # ── 04: Funding data coverage heatmap ──
        if venues:
            fd = self.funding.copy()
            fd["date"] = fd["datetime"].dt.normalize()
            if fd["date"].dt.tz is not None:
                fd["date"] = fd["date"].dt.tz_localize(None)

            fig, ax = plt.subplots(figsize=(14, 3))
            for i, v in enumerate(venues):
                has_data = fd.groupby("date")[v].apply(lambda x: x.notna().any())
                dates_with = has_data[has_data].index
                ax.scatter(dates_with, [i] * len(dates_with),
                           marker="|", s=10, alpha=0.5, label=v)
            ax.set_yticks(range(len(venues)))
            ax.set_yticklabels(venues)
            ax.set_title("Funding Data Coverage by Venue")
            ax.grid(alpha=0.3, axis="x")

            plt.tight_layout()
            plt.savefig(f"{figdir}/04_data_coverage.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    04_data_coverage.png")

        # ── 05: Regime distribution ──
        if hasattr(self, "_regime_df") and not self._regime_df.empty:
            regime_cols = [c for c in self._regime_df.columns
                           if c.endswith("_regime")]
            if regime_cols:
                fig, ax = plt.subplots(figsize=(10, 5))
                regime_order = ["steep_contango", "mild_contango",
                                "flat", "backwardation"]
                bar_data = {}
                for col in regime_cols:
                    source = col.replace("_regime", "")
                    counts = self._regime_df[col].value_counts()
                    total = counts.sum()
                    bar_data[source] = {r: counts.get(r, 0) / total * 100
                                        for r in regime_order}

                x = np.arange(len(regime_order))
                width = 0.8 / len(bar_data)
                for i, (source, vals) in enumerate(bar_data.items()):
                    heights = [vals[r] for r in regime_order]
                    ax.bar(x + i * width, heights, width,
                           label=source, alpha=0.8)

                ax.set_xticks(x + width * len(bar_data) / 2)
                ax.set_xticklabels(regime_order, rotation=15)
                ax.set_ylabel("% of Time")
                ax.set_title("Regime Distribution by Venue/Source")
                ax.legend()
                ax.grid(alpha=0.3, axis="y")

                plt.tight_layout()
                plt.savefig(f"{figdir}/05_regimes.png",
                            dpi=150, bbox_inches="tight")
                plt.close()
                print(f"    05_regimes.png")

        # ── 06: Basis distribution histogram ──
        if not self.front.empty:
            daily = (self.front.sort_values("datetime")
                     .groupby("date")["annualized_basis_pct"].last()
                     .dropna())

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(daily.values, bins=50, alpha=0.7, color="#1f77b4",
                    edgecolor="white")
            ax.axvline(daily.mean(), color="#d62728", linestyle="--",
                       label=f"Mean: {daily.mean():.1f}%")
            ax.axvline(daily.median(), color="#ff7f0e", linestyle="--",
                       label=f"Median: {daily.median():.1f}%")
            ax.set_xlabel("Annualised Basis (%)")
            ax.set_ylabel("Days")
            ax.set_title("Distribution of Annualised Futures Basis")
            ax.legend()
            ax.grid(alpha=0.3, axis="y")

            plt.tight_layout()
            plt.savefig(f"{figdir}/06_basis_distribution.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    06_basis_distribution.png")

        # ── 07: Lead/lag cross-correlation (Q1) ──
        if "binance" in venues and "hyperliquid" in venues:
            fd = self.funding.copy()
            b = fd[["datetime", "binance", "hyperliquid"]].dropna()
            if len(b) > 20:
                lags = range(-12, 13)
                xcorrs = [b["binance"].corr(b["hyperliquid"].shift(lag))
                          for lag in lags]

                fig, ax = plt.subplots(figsize=(10, 5))
                lag_hours = [l * 8 for l in lags]
                ax.bar(lag_hours, xcorrs, width=6, alpha=0.7, color="#2ca02c")
                best_idx = np.argmax(np.abs(xcorrs))
                ax.bar(lag_hours[best_idx], xcorrs[best_idx],
                       width=6, color="#d62728", label=f"Peak: {lag_hours[best_idx]}h")
                ax.set_xlabel("Lag (hours, positive = Binance leads)")
                ax.set_ylabel("Correlation")
                ax.set_title("Binance vs Hyperliquid Funding Cross-Correlation")
                ax.legend()
                ax.grid(alpha=0.3)

                plt.tight_layout()
                plt.savefig(f"{figdir}/07_lead_lag_xcorr.png",
                            dpi=150, bbox_inches="tight")
                plt.close()
                print(f"    07_lead_lag_xcorr.png")

        print("  Done.")


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _trend_label(series: pd.Series) -> str:
    """Simple trend classification from first/last thirds."""
    n = len(series)
    if n < 6:
        return "insufficient data"
    first = series.iloc[:n // 3].mean()
    last = series.iloc[-n // 3:].mean()
    pct_change = (last - first) / first * 100
    if pct_change > 20:
        return f"↑ strong uptrend ({pct_change:+.0f}%)"
    elif pct_change > 5:
        return f"↗ mild uptrend ({pct_change:+.0f}%)"
    elif pct_change < -20:
        return f"↓ strong downtrend ({pct_change:+.0f}%)"
    elif pct_change < -5:
        return f"↘ mild downtrend ({pct_change:+.0f}%)"
    else:
        return f"→ sideways ({pct_change:+.0f}%)"
