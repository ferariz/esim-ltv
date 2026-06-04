"""
esim_ltv/src/generate_data.py
------------------------------
Synthetic transaction data generator for non-contractual travel eSIM LTV modelling.

Design principles:
- Three user archetypes with distinct purchase cadences (leisure-one-time,
  leisure-repeat, digital-nomad)
- Four destination corridors with realistic margin profiles
- Seasonal travel cycles (summer peak, Christmas/NYE, Easter)
- Background 1 GB/month retention pings after plan expiry
- Lognormal inter-arrival times (right-skewed, physically plausible)

Run directly:
    python src/generate_data.py          # writes data/raw/transactions.parquet
                                          #        data/raw/users.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
N_USERS = 2_000
START_DATE = pd.Timestamp("2021-01-01")
END_DATE = pd.Timestamp("2024-01-01")  # 3-year window

ARCHETYPE_SHARES = {
    "leisure_once": 0.80,
    "leisure_repeat": 0.15,
    "digital_nomad": 0.05,
}

# Destination corridors: (label, median_revenue_eur, median_cost_eur, peak_months)
CORRIDORS = {
    "thailand": dict(
        label="Thailand",
        revenue_median=14.0,   # 2 x 3-day plan ~ €11.37 each
        cost_median=3.5,       # cheap wholesale
        peak_months=[1, 2, 12],
        share=0.18,
    ),
    "western_europe": dict(
        label="Western Europe",
        revenue_median=22.0,
        cost_median=8.0,
        peak_months=[6, 7, 8],
        share=0.35,
    ),
    "usa": dict(
        label="USA",
        revenue_median=26.9,
        cost_median=10.0,
        peak_months=[6, 7, 8, 12],
        share=0.28,
    ),
    "argentina": dict(
        label="Argentina",
        revenue_median=33.9,   # expensive wholesale reflected in price
        cost_median=18.0,
        peak_months=[1, 2, 11, 12],
        share=0.19,
    ),
}

# Plan durations available (days)
PLAN_DURATIONS = [3, 7, 14, 30]
DURATION_WEIGHTS = [0.25, 0.45, 0.20, 0.10]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seasonal_weight(month: int, peak_months: list[int]) -> float:
    """Return a multiplier [0.6, 1.8] based on whether month is peak."""
    if month in peak_months:
        return 1.6 + 0.2 * np.random.rand()
    return 0.6 + 0.4 * np.random.rand()


def _draw_inter_trip_days(archetype: str, rng: np.random.Generator) -> float:
    """
    Draw inter-trip interval in days from a lognormal distribution.
    Raw Gaussian noise is physically invalid for travel gaps — trips cluster
    around annual macro-cycles, so we use lognormal with archetype-specific
    parameters.
    """
    params = {
        "leisure_once": dict(mu=np.log(365), sigma=0.8),   # ~1/year, high variance
        "leisure_repeat": dict(mu=np.log(180), sigma=0.6),  # ~2/year
        "digital_nomad": dict(mu=np.log(60), sigma=0.5),   # ~6/year
    }
    p = params[archetype]
    return float(rng.lognormal(mean=p["mu"], sigma=p["sigma"]))


def _draw_revenue(corridor: str, duration_days: int, rng: np.random.Generator) -> float:
    """
    Revenue: base median scaled by plan duration, with lognormal noise.
    Longer plans are discounted (not linearly scaled).
    """
    base = CORRIDORS[corridor]["revenue_median"]
    duration_factor = {3: 0.55, 7: 1.0, 14: 1.65, 30: 3.0}[duration_days]
    median_rev = base * duration_factor
    # lognormal noise: std ~ 15% of median
    sigma = 0.15
    return float(rng.lognormal(mean=np.log(median_rev), sigma=sigma))


def _draw_cost(corridor: str, duration_days: int, rng: np.random.Generator) -> float:
    """
    Wholesale cost: stochastic (models per-GB cost uncertainty).
    Correlated with revenue but with higher dispersion.
    """
    base = CORRIDORS[corridor]["cost_median"]
    duration_factor = {3: 0.5, 7: 1.0, 14: 1.8, 30: 3.2}[duration_days]
    median_cost = base * duration_factor
    sigma = 0.25  # higher spread than revenue
    return float(rng.lognormal(mean=np.log(median_cost), sigma=sigma))


# ---------------------------------------------------------------------------
# Core generators
# ---------------------------------------------------------------------------


def generate_users(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate user-level attributes."""
    archetype_choices = rng.choice(
        list(ARCHETYPE_SHARES.keys()),
        size=n,
        p=list(ARCHETYPE_SHARES.values()),
    )
    corridor_keys = list(CORRIDORS.keys())
    corridor_shares = [CORRIDORS[k]["share"] for k in corridor_keys]
    primary_corridor = rng.choice(corridor_keys, size=n, p=corridor_shares)

    acquisition_dates = START_DATE + pd.to_timedelta(
        rng.integers(0, 180, size=n), unit="D"
    )

    users = pd.DataFrame(
        {
            "user_id": [f"U{i:05d}" for i in range(n)],
            "archetype": archetype_choices,
            "primary_corridor": primary_corridor,
            "acquisition_date": acquisition_dates,
        }
    )
    return users


def generate_transactions(
    users: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate the full 3-year transaction history.

    For each user:
    1. Simulate purchase events using lognormal inter-arrival times.
    2. Add a background retention ping (1 GB/month) after each plan expires.
    3. Apply seasonal weighting to acceptance of next trip.
    """
    records = []

    for _, user in users.iterrows():
        uid = user["user_id"]
        archetype = user["archetype"]
        corridor = user["primary_corridor"]
        t = user["acquisition_date"]

        # Determine number of purchases this user will make over 3 years
        # (cap at sensible maximum per archetype)
        max_purchases = {
            "leisure_once": 3,
            "leisure_repeat": 8,
            "digital_nomad": 20,
        }[archetype]

        # First purchase at acquisition
        gap = pd.Timedelta(days=float(rng.integers(0, 14)))
        purchase_date = t + gap

        purchase_count = 0
        while purchase_date < END_DATE and purchase_count < max_purchases:
            # Seasonality: sub-sample based on month
            c_info = CORRIDORS[corridor]
            s_weight = _seasonal_weight(purchase_date.month, c_info["peak_months"])
            if rng.random() > min(s_weight / 1.8, 1.0):
                # This trip is skipped — travel didn't materialize
                inter = _draw_inter_trip_days(archetype, rng)
                purchase_date += pd.Timedelta(days=inter)
                continue

            duration = int(rng.choice(PLAN_DURATIONS, p=DURATION_WEIGHTS))
            revenue = _draw_revenue(corridor, duration, rng)
            cost = _draw_cost(corridor, duration, rng)

            records.append(
                {
                    "user_id": uid,
                    "date": purchase_date.normalize(),
                    "corridor": corridor,
                    "duration_days": duration,
                    "revenue_eur": round(revenue, 2),
                    "cost_eur": round(cost, 2),
                    "is_retention_ping": False,
                }
            )
            purchase_count += 1

            # After plan expires, add 1 GB/month retention pings
            plan_end = purchase_date + pd.Timedelta(days=duration)
            ping_date = plan_end + pd.Timedelta(days=30)
            ping_count = 0
            max_pings = {
                "leisure_once": 2,
                "leisure_repeat": 3,
                "digital_nomad": 6,
            }[archetype]
            while ping_date < END_DATE and ping_count < max_pings:
                records.append(
                    {
                        "user_id": uid,
                        "date": ping_date.normalize(),
                        "corridor": corridor,
                        "duration_days": 0,
                        "revenue_eur": 0.0,
                        "cost_eur": 0.0,
                        "is_retention_ping": True,
                    }
                )
                ping_date += pd.Timedelta(days=30)
                ping_count += 1

            # Next purchase
            inter = _draw_inter_trip_days(archetype, rng)
            purchase_date += pd.Timedelta(days=max(inter, float(duration) + 1))

    txns = pd.DataFrame(records)
    txns["date"] = pd.to_datetime(txns["date"])
    txns["margin_eur"] = txns["revenue_eur"] - txns["cost_eur"]
    txns = txns.sort_values(["user_id", "date"]).reset_index(drop=True)
    return txns


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(output_dir: str = "data/raw") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    log.info("Generating %d users …", N_USERS)
    users = generate_users(N_USERS, rng)

    log.info("Generating transactions …")
    txns = generate_transactions(users, rng)

    purchase_txns = txns[~txns["is_retention_ping"]]
    log.info(
        "  %d purchase transactions | %d retention pings | %d total rows",
        len(purchase_txns),
        txns["is_retention_ping"].sum(),
        len(txns),
    )
    log.info(
        "  Revenue range: €%.2f – €%.2f",
        purchase_txns["revenue_eur"].min(),
        purchase_txns["revenue_eur"].max(),
    )

    users_path = out / "users.parquet"
    txns_path = out / "transactions.parquet"
    users.to_parquet(users_path, index=False)
    txns.to_parquet(txns_path, index=False)
    log.info("Saved → %s", users_path)
    log.info("Saved → %s", txns_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    main(args.output_dir)
