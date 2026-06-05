"""
esim_ltv/src/ltv_models.py
---------------------------
Two LTV modelling approaches for non-contractual travel eSIM.

BG/NBD + Gamma-Gamma
--------------------
Probabilistic baseline from Fader, Hardie & Lee (2005). Models:
  - BG/NBD: probability of being alive + expected number of future purchases
  - Gamma-Gamma: expected spend per transaction (conditional on activity)

Key limitation for travel eSIM (documented explicitly):
  BG/NBD assumes a CONSTANT individual dropout rate (geometric distribution
  over "alive" periods). Travel users do not drop out randomly — they follow
  annual macro-cycles. A user 9 months post-purchase is NOT more likely to
  have permanently churned than at 3 months; they are simply waiting for
  their next trip. The KM plateau in Hito 1 is the empirical proof.

  Consequence: BG/NBD will systematically UNDERESTIMATE LTV for dormant
  users who are within a travel macro-cycle, and will OVERESTIMATE churn
  probability for the leisure-once segment.

LightGBM
---------
Gradient-boosted trees on the RFM + survival-informed feature matrix.
Handles the zero-inflated, heavy-tailed label distribution without
distributional assumptions. Survival features (inter_trip_mean_days,
est_next_trip_days) allow the model to implicitly encode macro-cycle
structure that BG/NBD cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# ---------------------------------------------------------------------------
# BG/NBD helpers (wraps the lifetimes library)
# ---------------------------------------------------------------------------

def fit_bgnbd(
    purchases: pd.DataFrame,
    cutoff: pd.Timestamp,
    obs_start: pd.Timestamp,
) -> tuple:
    """
    Fit BG/NBD + Gamma-Gamma models using pre-cutoff purchase history.

    Returns (bgnbd_model, gg_model, summary_df)
    where summary_df is the lifetimes RFM summary frame.

    Requires: pip install lifetimes
    """
    try:
        from lifetimes import BetaGeoFitter, GammaGammaFitter
        from lifetimes.utils import summary_data_from_transaction_data
    except ImportError:
        raise ImportError(
            "lifetimes is required for BG/NBD. Install with: pip install lifetimes"
        )

    pre = purchases[purchases["date"] < cutoff].copy()

    # lifetimes expects: customer_id, datetime, monetary_value
    summary = summary_data_from_transaction_data(
        pre,
        customer_id_col="user_id",
        datetime_col="date",
        monetary_value_col="revenue_eur",
        observation_period_end=cutoff,
    )

    # BG/NBD: frequency of repeat purchases + recency + T (tenure).
    # Synthetic lognormal inter-arrivals require a higher penalizer than
    # real e-commerce data. Retry with increasing values.
    fitted = False
    for penalizer in [0.01, 0.1, 0.5, 1.0]:
        try:
            bgf = BetaGeoFitter(penalizer_coef=penalizer)
            bgf.fit(summary["frequency"], summary["recency"], summary["T"])
            fitted = True
            break
        except Exception:
            continue
    if not fitted:
        raise RuntimeError(
            "BG/NBD did not converge even with penalizer=1.0. "
            "This confirms the synthetic data's macro-cycle structure "
            "violates BG/NBD's geometric dropout assumption — see Hito 1 KM curves."
        )

    # Gamma-Gamma: spend model (only customers with repeat purchases)
    gg_data = summary[summary["frequency"] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(gg_data["frequency"], gg_data["monetary_value"])

    return bgf, ggf, summary


def predict_bgnbd_ltv(
    bgf,
    ggf,
    summary: pd.DataFrame,
    horizon_days: int,
    discount_rate: float = 0.01,
) -> pd.DataFrame:
    """
    Predict LTV for each user using BG/NBD + Gamma-Gamma.

    Parameters
    ----------
    horizon_days  : prediction horizon in days
    discount_rate : monthly discount rate for CLV calculation

    Returns DataFrame with user_id and predicted_ltv_bgnbd.
    """
    from lifetimes import GammaGammaFitter

    horizon_months = horizon_days / 30.0

    clv = ggf.customer_lifetime_value(
        bgf,
        summary["frequency"],
        summary["recency"],
        summary["T"],
        summary["monetary_value"],
        time=horizon_months,
        discount_rate=discount_rate,
        freq="D",
    )

    result = clv.reset_index()
    result.columns = ["user_id", "predicted_ltv_bgnbd"]
    return result


# ---------------------------------------------------------------------------
# LightGBM LTV model
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # RFM
    "recency_days",
    "frequency",
    "monetary_mean",
    "monetary_total",
    "tenure_days",
    # Survival-informed
    "days_since_last",
    "n_inter_trip_intervals",
    "inter_trip_mean_days",
    "inter_trip_std_days",
    "inter_trip_cv",
    "est_next_trip_days",
    # Retention ping
    "n_pings_pre_cutoff",
    "days_since_ping",
    "has_ping",
    # Categorical (encoded)
    "archetype_enc",
    "corridor_enc",
]


def train_lgbm(
    feature_matrix: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    lgbm_params: dict | None = None,
) -> tuple:
    """
    Train a LightGBM regressor for LTV prediction.

    Uses Tweedie loss (power=1.5) — appropriate for zero-inflated,
    right-skewed distributions like travel spend.

    Returns (model, X_train, X_test, y_train, y_test, feature_importance_df)
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm is required. Install with: pip install lightgbm")

    fm = feature_matrix.copy()
    X = fm[FEATURE_COLS]
    y = fm["label_revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    default_params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "metric": "tweedie",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": random_state,
        "verbose": -1,
    }
    if lgbm_params:
        default_params.update(lgbm_params)

    model = lgb.LGBMRegressor(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return model, X_train, X_test, y_train, y_test, importance


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "model",
) -> dict:
    """
    Compute MAE, MAPE (on non-zero actuals), and a calibration summary.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).clip(min=0)

    mae = mean_absolute_error(y_true, y_pred)

    # MAPE only on non-zero actuals (zero-LTV users trivially have infinite % error)
    nonzero = y_true > 0
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        if nonzero.sum() > 0 else np.nan
    )

    # Calibration: predicted vs actual by decile
    df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    df["decile"] = pd.qcut(y_pred, q=10, labels=False, duplicates="drop")
    calibration = (
        df.groupby("decile")[["actual", "predicted"]]
        .mean()
        .reset_index()
    )

    return {
        "model": model_name,
        "mae": round(mae, 2),
        "mape": round(mape, 1) if not np.isnan(mape) else None,
        "n_test": len(y_true),
        "calibration": calibration,
    }
