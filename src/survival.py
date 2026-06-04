"""
esim_ltv/src/survival.py
-------------------------
Helpers for survival / inter-purchase analysis.

Key concepts
------------
- "Event" = a repeat purchase (the customer returned).
- "Censored" = user has not purchased again by the observation window end.
- For non-contractual travel eSIM, churn is NOT permanent — the Kaplan-Meier
  plateau at ~12 months is the empirical proof.

Public API
----------
build_survival_frame(txns, obs_end)
    → DataFrame with one row per (user, purchase) with days_to_next and
      event indicator. Ignores retention pings.

kaplan_meier(df, time_col, event_col, label_col=None)
    → Returns a tidy DataFrame of (time, survival, lower_ci, upper_ci, label)
      suitable for plotting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_survival_frame(
    txns: pd.DataFrame,
    obs_end: pd.Timestamp,
    include_retention_pings: bool = False,
) -> pd.DataFrame:
    """
    Build a survival frame from a transactions DataFrame.

    Parameters
    ----------
    txns : pd.DataFrame
        Must contain columns: user_id, date, is_retention_ping.
    obs_end : pd.Timestamp
        Observation window end (used to censor open intervals).
    include_retention_pings : bool
        If True, include the 1 GB/month pings in the inter-event calculation.

    Returns
    -------
    pd.DataFrame with columns:
        user_id, purchase_date, days_to_next, event (1=returned, 0=censored)
    """
    if not include_retention_pings:
        df = txns[~txns["is_retention_ping"]].copy()
    else:
        df = txns.copy()

    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    # Compute time to next purchase within each user
    df["next_date"] = df.groupby("user_id")["date"].shift(-1)
    df["is_last"] = df["next_date"].isna()

    # For the last purchase, censor at obs_end
    df["next_date"] = df["next_date"].fillna(obs_end)
    df["days_to_next"] = (df["next_date"] - df["date"]).dt.days.clip(lower=1)
    df["event"] = (~df["is_last"]).astype(int)

    return df[["user_id", "date", "days_to_next", "event"]].copy()


def kaplan_meier(
    df: pd.DataFrame,
    time_col: str = "days_to_next",
    event_col: str = "event",
    label: str = "all",
) -> pd.DataFrame:
    """
    Compute Kaplan-Meier survival function with Greenwood confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
        One row per interval, with time and event columns.
    time_col : str
    event_col : str
    label : str
        Name to attach to the output rows (for faceting in plots).

    Returns
    -------
    pd.DataFrame with columns: time, survival, lower_ci, upper_ci, label
    """
    times = np.sort(df[time_col].unique())
    n_total = len(df)

    rows = []
    S = 1.0
    greenwood_sum = 0.0  # for variance: Σ d_i / (n_i * (n_i - d_i))
    at_risk = n_total

    for t in times:
        mask_t = df[time_col] == t
        d_i = int(df.loc[mask_t, event_col].sum())   # events at time t
        n_i = int((df[time_col] >= t).sum())           # at risk at time t

        if n_i == 0:
            break

        if d_i > 0:
            S = S * (1 - d_i / n_i)
            if n_i > d_i:
                greenwood_sum += d_i / (n_i * (n_i - d_i))

        se = S * np.sqrt(greenwood_sum)
        z = 1.96  # 95% CI
        rows.append(
            {
                "time": t,
                "survival": S,
                "lower_ci": max(0.0, S - z * se),
                "upper_ci": min(1.0, S + z * se),
                "label": label,
            }
        )

    return pd.DataFrame(rows)


def build_km_by_group(
    txns: pd.DataFrame,
    users: pd.DataFrame,
    obs_end: pd.Timestamp,
    group_col: str = "archetype",
) -> pd.DataFrame:
    """
    Convenience wrapper: join txns with users, stratify by group_col,
    compute KM per stratum, return a single tidy DataFrame.
    """
    df = txns.merge(users[["user_id", group_col]], on="user_id", how="left")
    survival_df = build_survival_frame(df, obs_end)
    survival_df = survival_df.merge(users[["user_id", group_col]], on="user_id", how="left")

    results = []
    for group, grp in survival_df.groupby(group_col):
        km = kaplan_meier(grp, label=str(group))
        results.append(km)

    # Also compute overall
    km_all = kaplan_meier(survival_df, label="all")
    results.append(km_all)

    return pd.concat(results, ignore_index=True)
