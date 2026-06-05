"""
esim_ltv/src/pricing_bridge.py
--------------------------------
Connect LTV predictions to acquisition and retention strategy.

Three business questions answered:
1. CAC ceiling       — max acquisition spend per segment, given predicted LTV
2. Retention discount — max discount to reactivate a dormant high-LTV user
3. Corridor ranking  — destinations ranked by expected LTV net of wholesale cost

All calculations are margin-based (not revenue-based) to account for
Holafly's stochastic wholesale cost structure identified in Hito 1.

Design note: these are decision-support tools, not optimisers. They produce
human-interpretable tables and curves that a pricing or growth team can act
on directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. CAC ceiling
# ---------------------------------------------------------------------------

def compute_cac_ceiling(
    ltv_predictions: pd.DataFrame,
    target_ltv_margin_multiple: float = 3.0,
    margin_col: str = "label_margin",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute the maximum justifiable CAC per segment.

    CAC ceiling = predicted LTV margin / target_ltv_margin_multiple

    The LTV:CAC multiple is a standard SaaS/e-commerce heuristic. For a
    non-contractual travel product with uncertain repeat purchase timing,
    we use 3× as the baseline (conservative vs the 3–5× range typical in
    subscription businesses).

    Parameters
    ----------
    ltv_predictions   : output of build_feature_matrix, with predicted_ltv
                        and label_margin columns
    target_ltv_margin_multiple : LTV:CAC target ratio
    margin_col        : column to use as the LTV basis (margin, not revenue)
    group_cols        : columns to stratify by (default: archetype + corridor)

    Returns
    -------
    DataFrame with columns:
        group cols, n_users, median_predicted_ltv, median_margin,
        cac_ceiling_median, cac_ceiling_p25, cac_ceiling_p75
    """
    if group_cols is None:
        group_cols = ["archetype"]

    df = ltv_predictions.copy()

    # Use predicted margin: scale predicted_ltv by observed margin rate
    # (avoids using label data directly in the CAC calculation)
    df["margin_rate"] = np.where(
        df["label_revenue"] > 0,
        df["label_margin"] / df["label_revenue"],
        np.nan,
    )
    # Fill missing margin rate with corridor median
    corridor_margin_rate = (
        df.groupby("primary_corridor")["margin_rate"]
        .median()
        .rename("corridor_margin_rate")
    )
    df = df.merge(corridor_margin_rate, on="primary_corridor", how="left")
    df["margin_rate"] = df["margin_rate"].fillna(df["corridor_margin_rate"])

    df["predicted_margin"] = df["predicted_ltv"] * df["margin_rate"]
    df["cac_ceiling"] = df["predicted_margin"] / target_ltv_margin_multiple

    result = (
        df.groupby(group_cols)
        .agg(
            n_users=("user_id", "count"),
            median_predicted_ltv=("predicted_ltv", "median"),
            median_predicted_margin=("predicted_margin", "median"),
            cac_ceiling_median=("cac_ceiling", "median"),
            cac_ceiling_p25=("cac_ceiling", lambda x: x.quantile(0.25)),
            cac_ceiling_p75=("cac_ceiling", lambda x: x.quantile(0.75)),
        )
        .round(2)
        .reset_index()
    )

    result["ltv_cac_multiple_at_ceiling"] = target_ltv_margin_multiple

    return result


# ---------------------------------------------------------------------------
# 2. Retention discount strategy
# ---------------------------------------------------------------------------

def compute_retention_discount(
    ltv_predictions: pd.DataFrame,
    reactivation_probability: float = 0.25,
    avg_plan_revenue: float = 26.9,
    discount_increments: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Compute the maximum discount worth offering to dormant high-LTV users.

    Framework:
        Expected value of offering discount d to a dormant user =
            P(reactivate | discount) × (predicted_margin - discount_cost)

        Where:
        - P(reactivate | discount) scales with discount size (logistic)
        - discount_cost = discount_fraction × avg_plan_revenue
        - The offer is worth making if EV > 0 (i.e. margin > discount cost
          adjusted for reactivation probability)

        Max discount = predicted_margin × reactivation_probability
                       (break-even condition on expected margin)

    Parameters
    ----------
    ltv_predictions         : must contain predicted_ltv, recency_days,
                              archetype, primary_corridor
    reactivation_probability : baseline P(reactivate) at zero discount
    avg_plan_revenue         : reference plan price for discount calculation
    discount_increments      : discount amounts to evaluate (€)

    Returns
    -------
    Two DataFrames:
        - user_level: per-user max discount and reactivation EV
        - discount_curve: EV by discount level for each archetype
    """
    if discount_increments is None:
        discount_increments = np.arange(0, 16, 1)  # €0 to €15

    df = ltv_predictions.copy()

    # Margin rate (same logic as CAC ceiling)
    df["margin_rate"] = np.where(
        df["label_revenue"] > 0,
        df["label_margin"] / df["label_revenue"],
        np.nan,
    )
    corridor_rate = df.groupby("primary_corridor")["margin_rate"].median()
    df = df.merge(corridor_rate.rename("corridor_rate"), on="primary_corridor", how="left")
    df["margin_rate"] = df["margin_rate"].fillna(df["corridor_rate"])
    df["predicted_margin"] = df["predicted_ltv"] * df["margin_rate"]

    # Identify dormant users (recency > 180 days, at least 1 prior purchase)
    df["is_dormant"] = (df["recency_days"] > 180) & (df["frequency"] >= 1)

    # Max discount = break-even point
    # EV = P_reactivate × predicted_margin - discount_cost ≥ 0
    # → max_discount = P_reactivate × predicted_margin
    df["max_discount_eur"] = (reactivation_probability * df["predicted_margin"]).clip(
        lower=0, upper=avg_plan_revenue * 0.5  # cap at 50% off
    ).round(2)
    df["max_discount_pct"] = (df["max_discount_eur"] / avg_plan_revenue * 100).round(1)

    # Discount curve by archetype
    # P(reactivate | discount) using a simple logistic scaling
    curve_rows = []
    for arch in df["archetype"].unique():
        arch_df = df[(df["archetype"] == arch) & df["is_dormant"]]
        if arch_df.empty:
            continue
        median_margin = arch_df["predicted_margin"].median()
        for d in discount_increments:
            # Reactivation probability increases with discount (logistic)
            p_react = reactivation_probability + (1 - reactivation_probability) * (
                1 / (1 + np.exp(-0.3 * (d - 5)))
            ) * 0.5  # saturates at ~50% reactivation at high discounts
            ev = p_react * median_margin - d
            curve_rows.append({
                "archetype": arch,
                "discount_eur": d,
                "p_reactivate": round(p_react, 3),
                "expected_value_eur": round(ev, 2),
            })

    discount_curve = pd.DataFrame(curve_rows)

    user_level = df[[
        "user_id", "archetype", "primary_corridor",
        "recency_days", "frequency", "predicted_ltv", "predicted_margin",
        "is_dormant", "max_discount_eur", "max_discount_pct",
    ]].copy()

    return user_level, discount_curve


# ---------------------------------------------------------------------------
# 3. Corridor prioritisation
# ---------------------------------------------------------------------------

def rank_corridors(
    ltv_predictions: pd.DataFrame,
    corridor_margin_rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Rank destination corridors by expected LTV contribution net of wholesale cost.

    Incorporates:
    - Volume (number of users per corridor)
    - Predicted LTV per user
    - Margin rate per corridor (from Hito 1 empirical data)
    - Margin volatility (std of margin rate = wholesale cost risk)

    Parameters
    ----------
    ltv_predictions      : full prediction frame
    corridor_margin_rates: override median margin rates if known
                           (e.g. from actual Holafly data)

    Returns
    -------
    DataFrame ranked by expected_total_margin, with risk-adjusted score.
    """
    # Empirical margin rates from Hito 1 (median margin / median revenue)
    default_rates = {
        "thailand":       0.748,  # 74.8% — cheap wholesale
        "western_europe": 0.636,  # 63.6%
        "usa":            0.629,  # 62.9%
        "argentina":      0.458,  # 45.8% — expensive + volatile wholesale
    }
    if corridor_margin_rates:
        default_rates.update(corridor_margin_rates)

    df = ltv_predictions.copy()
    df["corridor_margin_rate"] = df["primary_corridor"].map(default_rates)
    df["predicted_margin"] = df["predicted_ltv"] * df["corridor_margin_rate"]

    ranking = (
        df.groupby("primary_corridor")
        .agg(
            n_users=("user_id", "count"),
            median_predicted_ltv=("predicted_ltv", "median"),
            mean_predicted_ltv=("predicted_ltv", "mean"),
            median_predicted_margin=("predicted_margin", "median"),
            mean_predicted_margin=("predicted_margin", "mean"),
            total_predicted_margin=("predicted_margin", "sum"),
            margin_std=("predicted_margin", "std"),
        )
        .round(2)
        .reset_index()
    )

    ranking["margin_rate"] = ranking["primary_corridor"].map(default_rates)

    # Sharpe-style score: expected margin / std (reward per unit of wholesale risk)
    ranking["risk_adjusted_score"] = (
        ranking["mean_predicted_margin"] / ranking["margin_std"]
    ).round(3)

    # Revenue concentration: what % of total predicted margin does this corridor represent
    ranking["margin_share_pct"] = (
        ranking["total_predicted_margin"] / ranking["total_predicted_margin"].sum() * 100
    ).round(1)

    ranking = ranking.sort_values("total_predicted_margin", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1  # 1-based rank
    ranking.index.name = "rank"

    return ranking
