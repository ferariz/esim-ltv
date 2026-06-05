"""
esim_ltv/src/features.py
-------------------------
Feature engineering for LTV modelling.

Design principles
-----------------
- All features are computed relative to a CUTOFF date. Everything after
  the cutoff is the label period — never used in features.
- Zero-purchase users are included with zero RFM values (no selection bias).
- The retention ping feature is computed using only pings BEFORE the cutoff
  to avoid the leakage identified in Hito 1 EDA (users who buy more also
  generate more pings — if we include post-cutoff pings we leak the label).

Public API
----------
build_rfm(purchases, users, cutoff, obs_start) -> DataFrame
    Classic RFM features per user, cutoff-safe.

build_survival_features(purchases, users, cutoff) -> DataFrame
    Survival-informed features: days_since_last, inter_trip_mean/std,
    estimated_next_trip_days.

build_ping_features(txns, cutoff) -> DataFrame
    Retention ping features, cutoff-safe.

build_feature_matrix(purchases, txns, users, cutoff, obs_start) -> DataFrame
    Full feature matrix joining all of the above. Includes all 2,000 users.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_rfm(
    purchases: pd.DataFrame,
    users: pd.DataFrame,
    cutoff: pd.Timestamp,
    obs_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute RFM features per user using only purchases before the cutoff.

    Returns a DataFrame with one row per user (including zero-purchase users).

    Columns
    -------
    recency_days        : days since most recent purchase before cutoff
                          (NaN for zero-purchase users → filled with max)
    frequency           : number of purchases before cutoff
    monetary_mean       : mean revenue per transaction before cutoff
    monetary_total      : total revenue before cutoff
    tenure_days         : days from acquisition to cutoff
    """
    pre = purchases[purchases["date"] < cutoff].copy()

    rfm = (
        pre.groupby("user_id")
        .agg(
            last_purchase_date=("date", "max"),
            frequency=("revenue_eur", "count"),
            monetary_mean=("revenue_eur", "mean"),
            monetary_total=("revenue_eur", "sum"),
        )
        .reset_index()
    )

    rfm["recency_days"] = (cutoff - rfm["last_purchase_date"]).dt.days

    # Merge with all users (includes zero-purchase users)
    base = users[["user_id", "acquisition_date", "archetype", "primary_corridor"]].copy()
    rfm = base.merge(rfm, on="user_id", how="left")

    # Fill zero-purchase users
    rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
    rfm["monetary_mean"] = rfm["monetary_mean"].fillna(0.0)
    rfm["monetary_total"] = rfm["monetary_total"].fillna(0.0)
    # Recency for zero-purchase users = time since acquisition (worst recency)
    rfm["recency_days"] = rfm["recency_days"].fillna(
        (cutoff - rfm["acquisition_date"]).dt.days
    )

    rfm["tenure_days"] = (cutoff - rfm["acquisition_date"]).dt.days

    return rfm[
        [
            "user_id",
            "archetype",
            "primary_corridor",
            "recency_days",
            "frequency",
            "monetary_mean",
            "monetary_total",
            "tenure_days",
        ]
    ]


def build_survival_features(
    purchases: pd.DataFrame,
    users: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """
    Survival-informed features using only pre-cutoff purchases.

    These features are the key differentiator from vanilla RFM — they encode
    the travel macro-cycle structure identified in Hito 1.

    Columns
    -------
    days_since_last         : recency in days (same as RFM but kept here for
                              model transparency)
    n_inter_trip_intervals  : number of observed inter-trip gaps
    inter_trip_mean_days    : mean inter-trip interval (NaN → 0 for new users)
    inter_trip_std_days     : std of inter-trip intervals (0 for single buyers)
    inter_trip_cv           : coefficient of variation (std/mean), travel
                              regularity proxy
    est_next_trip_days      : estimated days until next trip =
                              max(0, inter_trip_mean - days_since_last)
                              Positive → user is likely within their cycle
                              Zero     → user is overdue (possible churn signal)
    """
    pre = purchases[purchases["date"] < cutoff].copy()
    pre = pre.sort_values(["user_id", "date"])

    # Inter-trip intervals per user
    pre["prev_date"] = pre.groupby("user_id")["date"].shift(1)
    pre["interval_days"] = (pre["date"] - pre["prev_date"]).dt.days

    interval_stats = (
        pre.dropna(subset=["interval_days"])
        .groupby("user_id")["interval_days"]
        .agg(
            n_inter_trip_intervals="count",
            inter_trip_mean_days="mean",
            inter_trip_std_days="std",
        )
        .reset_index()
    )

    last_purchase = (
        pre.groupby("user_id")["date"]
        .max()
        .reset_index()
        .rename(columns={"date": "last_purchase_date"})
    )
    last_purchase["days_since_last"] = (cutoff - last_purchase["last_purchase_date"]).dt.days

    # Join to all users
    base = users[["user_id"]].copy()
    feats = (
        base
        .merge(last_purchase[["user_id", "days_since_last"]], on="user_id", how="left")
        .merge(interval_stats, on="user_id", how="left")
    )

    # Fill zero-purchase users
    feats["days_since_last"] = feats["days_since_last"].fillna(999)
    feats["n_inter_trip_intervals"] = feats["n_inter_trip_intervals"].fillna(0).astype(int)
    feats["inter_trip_mean_days"] = feats["inter_trip_mean_days"].fillna(0.0)
    feats["inter_trip_std_days"] = feats["inter_trip_std_days"].fillna(0.0)

    # CV: regularity of travel cadence
    feats["inter_trip_cv"] = np.where(
        feats["inter_trip_mean_days"] > 0,
        feats["inter_trip_std_days"] / feats["inter_trip_mean_days"],
        0.0,
    )

    # Estimated days to next trip
    feats["est_next_trip_days"] = (
        feats["inter_trip_mean_days"] - feats["days_since_last"]
    ).clip(lower=0)

    return feats[
        [
            "user_id",
            "days_since_last",
            "n_inter_trip_intervals",
            "inter_trip_mean_days",
            "inter_trip_std_days",
            "inter_trip_cv",
            "est_next_trip_days",
        ]
    ]


def build_ping_features(
    txns: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """
    Retention ping features — cutoff-safe.

    Only pings strictly before the cutoff are used. This avoids the leakage
    identified in Hito 1: post-cutoff pings correlate with the label because
    high-frequency buyers generate more pings AND more future revenue.

    Columns
    -------
    n_pings_pre_cutoff  : number of 1 GB/month pings before cutoff
    has_ping            : binary flag
    days_since_ping     : days since most recent ping (999 if no pings)
    """
    pings = txns[txns["is_retention_ping"] & (txns["date"] < cutoff)].copy()

    ping_stats = (
        pings.groupby("user_id")
        .agg(
            n_pings_pre_cutoff=("date", "count"),
            last_ping_date=("date", "max"),
        )
        .reset_index()
    )
    ping_stats["days_since_ping"] = (cutoff - ping_stats["last_ping_date"]).dt.days
    ping_stats["has_ping"] = 1

    # Use full user list from txns — but caller should pass users df for completeness.
    # We union txns user_ids with any user_id present in the index to catch
    # zero-purchase zero-ping users. Handled in build_feature_matrix via left join on rfm.
    all_users = txns[["user_id"]].drop_duplicates()
    feats = all_users.merge(ping_stats[["user_id", "n_pings_pre_cutoff",
                                        "days_since_ping", "has_ping"]],
                            on="user_id", how="left")
    feats["n_pings_pre_cutoff"] = feats["n_pings_pre_cutoff"].fillna(0).astype(int)
    feats["days_since_ping"] = feats["days_since_ping"].fillna(999)
    feats["has_ping"] = feats["has_ping"].fillna(0).astype(int)

    return feats


def build_feature_matrix(
    purchases: pd.DataFrame,
    txns: pd.DataFrame,
    users: pd.DataFrame,
    cutoff: pd.Timestamp,
    obs_start: pd.Timestamp,
    label_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Assemble the full feature matrix with label.

    Features: RFM + survival-informed + ping (all pre-cutoff).
    Label: total_revenue_post_cutoff (sum of purchases in [cutoff, label_end]).

    Includes all 2,000 users — zero-purchase users have label = 0.

    Parameters
    ----------
    cutoff     : feature/label split point (e.g. 2023-01-01)
    obs_start  : start of observation window (2021-01-01)
    label_end  : end of label window (2024-01-01)
    """
    rfm = build_rfm(purchases, users, cutoff, obs_start)
    surv = build_survival_features(purchases, users, cutoff)
    ping = build_ping_features(txns, cutoff)

    # Label: revenue in the post-cutoff period
    post = purchases[(purchases["date"] >= cutoff) & (purchases["date"] < label_end)]
    labels = (
        post.groupby("user_id")["revenue_eur"]
        .sum()
        .reset_index()
        .rename(columns={"revenue_eur": "label_revenue"})
    )

    # Also compute label margin for Hito 3
    labels_margin = (
        post.groupby("user_id")["margin_eur"]
        .sum()
        .reset_index()
        .rename(columns={"margin_eur": "label_margin"})
    )

    # Assemble
    fm = (
        rfm
        .merge(surv, on="user_id", how="left")
        .merge(ping, on="user_id", how="left")
        .merge(labels, on="user_id", how="left")
        .merge(labels_margin, on="user_id", how="left")
    )

    fm["label_revenue"] = fm["label_revenue"].fillna(0.0)
    fm["label_margin"] = fm["label_margin"].fillna(0.0)

    # Users not in txns at all (zero purchases + zero pings) arrive with null
    # ping columns after the merge — fill them explicitly
    fm["n_pings_pre_cutoff"] = fm["n_pings_pre_cutoff"].fillna(0).astype(int)
    fm["days_since_ping"] = fm["days_since_ping"].fillna(999)
    fm["has_ping"] = fm["has_ping"].fillna(0).astype(int)

    # Encode categoricals
    fm["archetype_enc"] = fm["archetype"].map(
        {"leisure_once": 0, "leisure_repeat": 1, "digital_nomad": 2}
    )
    fm["corridor_enc"] = fm["primary_corridor"].map(
        {"thailand": 0, "western_europe": 1, "usa": 2, "argentina": 3}
    )

    return fm
