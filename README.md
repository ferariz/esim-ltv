# esim-ltv — Non-Contractual LTV Modelling for Travel eSIM

> **Core thesis:** Standard LTV models assume churn is permanent. In travel eSIM it is not — a user who has not bought in 9 months is likely waiting for their next trip, not lost forever. This project implements a non-contractual LTV framework built around that insight.

---

## Why this problem is different

A travel eSIM provider sells unlimited plans priced by duration (days), not gigabytes. Their customers follow **travel macro-cycles** — annual summer holidays, Christmas trips, shoulder-season breaks — not subscription renewal rhythms.

A BG/NBD model treats every dormant user as having some probability of permanent dropout. Applied naively to travel eSIM, this mislabels a large fraction of waiting customers as churned, understates LTV, and misallocates retention budget.

The Kaplan-Meier survival curves in Hito 1 make this concrete: S(t) **plateaus at ~12 months**, meaning a substantial fraction of users remain in a pending-trip state well past the point where standard models would write them off.

---

## Project structure

```
esim-ltv/
├── src/
│   ├── generate_data.py    # Synthetic data generator (2,000 users, 3-year history)
│   └── survival.py         # KM estimator and survival frame builder
├── notebooks/
│   ├── 01_eda_survival.ipynb    # EDA, KM curves, margin profiles
│   ├── 02_ltv_models.ipynb      # BG/NBD + Gamma-Gamma vs LightGBM
│   └── 03_pricing_bridge.ipynb  # CAC optimisation, retention discounts, corridor ranking
├── data/
│   ├── raw/                # Generated parquet files (gitignored)
│   └── processed/          # Figures and model outputs
├── requirements.txt
└── Makefile
```

---

## Milestones

| Hito | Notebook | Status | Key deliverable |
|------|----------|--------|-----------------|
| 1 | `01_eda_survival.ipynb` | ✅ | KM survival curves proving dormant ≠ churned |
| 2 | `02_ltv_models.ipynb` | ✅ | BG/NBD + Gamma-Gamma vs LightGBM comparison |
| 3 | `03_pricing_bridge.ipynb` | ✅ | CAC ceiling, retention discount strategy, corridor LTV ranking |

---

## Slide deck

An interactive summary of the project is available at [`slides.html`](./slides.html)
— open locally in a browser or via GitHub Pages.

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
make data

# Launch notebooks
jupyter lab notebooks/
```

---

## Synthetic data design

**2,000 users, 3-year history (2021–2024).** Three customer archetypes with distinct purchase cadences:

| Archetype | Share | Avg trips/year | Description |
|-----------|-------|----------------|-------------|
| Leisure once | 80% | ~1 | Buys once, dormant 6–12 months, not truly churned |
| Leisure repeat | 15% | ~2 | 1–2 trips per year |
| Digital nomad | 5% | ~6 | 3–6 trips per year, high LTV |

**Four destination corridors** with realistic margin profiles:
- **Thailand** — cheap wholesale, high volume, lower margin
- **Western Europe** — mid-tier, summer-peaked
- **USA** — mid-tier, summer + Christmas peaked
- **Argentina** — expensive wholesale, higher price, margin-volatile

Inter-arrival times use **lognormal distributions** (not Gaussian) — physically motivated by the right-skewed nature of travel gaps, which cluster around annual macro-cycles.

---

## Key findings (Hito 1)

- **Dormant ≠ churned:** KM plateau at ~12 months is empirically confirmed in synthetic data generated with physically-motivated lognormal inter-arrival times
- **Revenue is Pareto-distributed:** Top ~20% of users generate ~80% of revenue — high-frequency travellers dominate LTV
- **Margin is stochastic:** Argentina corridor shows meaningful negative-margin tail due to wholesale cost volatility
- **Retention pings are predictive:** Users who generate 1 GB/month background pings have higher repeat purchase rates

---

## Related work

Companion project: [`esim-pricing-engine`](https://github.com/ferariz/esim-pricing-engine) — price elasticity modelling, demand forecasting, and A/B testing framework for the same product context.

---

## Author

Fernando Arizmendi — [github.com/ferariz](https://github.com/ferariz)  
PhD in Climate Dynamics & Complex Systems (Marie Curie Fellow)  
Senior Data Scientist
