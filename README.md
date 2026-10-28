# Nassau Intelligence — Factory Reallocation & Shipping Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit 1.57](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn 1.3](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E.svg)](https://scikit-learn.org/)
[![Tests 23/23 Passing](https://img.shields.io/badge/tests-23%2F23%20passed-22c55e.svg)](tests/)
[![Design: Swiss Modernist](https://img.shields.io/badge/UI%2FUX-Swiss%20Modernist%20%7C%20Glassmorphism-0F2B1D.svg)](styles/)

> **Executive Analytical Disclaimer:**  
> **“Alternative-factory benefits are simulated estimates, not verified operational outcomes. Historical product assignments do not establish what would happen if a product were manufactured at a different factory. Validate promising scenarios using factory capacity, production compatibility, actual transportation costs, and a controlled operational pilot.”**

---

## 1. Project Overview & Architecture

**Nassau Intelligence** is an enterprise supply-chain decision-support system built for Nassau Candy Distributor. It pairs rigorous econometric evaluation with counterfactual factory reallocation simulations, packaged in a Swiss Modernist and glassmorphism-inspired Streamlit interface.

### The Business Challenge
In the historical dataset of 10,194 shipments across North America, products are strictly partitioned: **each candy SKU was manufactured at exactly one factory**. For example:
- *Wonka Bar - Milk Chocolate* was produced exclusively at **Lot's of Nuts** (Chicago, IL).
- *Almond Joy* was produced exclusively at **Wicked Choc Factory** (Denver, CO).
- *Twix Miniatures* was produced exclusively at **Secret Factory** (Dallas, TX).

No SKU in historical data was ever dual-sourced. Consequently:
1. Any estimate of shipping lead time if an SKU were shifted to an alternative plant is a **counterfactual scenario simulation**, not an observed fact.
2. Machine learning models cannot observe within-SKU factory variance in historical data.
3. Claims of guaranteed savings from factory reallocation are methodologically false without operational validation.

This platform bridges this gap by establishing strict **data provenance**, enforcing transparent **evidence classification**, and evaluating whether candidate reallocations warrant operational pilot trials.

---

## 2. Standardized Transparency & Provenance Taxonomy

All data elements and visualizations across Nassau Intelligence adhere to a strict 5-tier taxonomy:

| Taxonomy Level | Definition | UI Treatment / Badge |
| :--- | :--- | :--- |
| **Historical Observation** | Direct empirical facts extracted from the shipment manifest (e.g., recorded dispatch dates, historical order quantities, factory locations). | `Observed Fact` (Forest Green badge) |
| **Model Prediction** | Order-to-dispatch lead time estimated by a trained, leakage-free linear response surface ($R^2 \approx 0.6052$, $RMSE \approx 1.1527$ days). | `Model Prediction` (Teal badge) |
| **Scenario Simulation** | Counterfactual outcome computed under explicit spatial and dispatch assumptions. | `Simulated Estimate — not yet validated in production` (Amber badge) |
| **Unvalidated Assumption** | Heuristic financial or transit parameters not verified by carrier invoices (e.g., $\$0.20$ transit cost per 1,000 km, $\$0.08$ inventory holding/day). | `Unvalidated Assumption` (Subtle italic notation) |
| **Insufficient Evidence** | SKUs with low transaction volume ($<20$ total orders or $<5$ regional orders) where counterfactual modeling is statistically unreliable. | `Insufficient Evidence` (Crimson pill & warning alert) |

---

## 3. Machine Learning Evaluation & Parsimony Benchmark

### Target Metric Definition
- **Measurement Semantic:** Order-to-Dispatch Lead Time ($T_{\text{dispatch}} = \text{Ship Date} - \text{Order Date}$).
- **Scope:** Measures factory fulfillment latency (internal plant warehouse staging to carrier handover), **not final customer doorstep arrival**.
- **Data Range & Cleaning:** Historical values span 0 to 7 days ($Mean = 3.96$ days, $Median = 4.0$ days, $IQR = 2.0$ days). Extreme calendar artifacts are cleanly normalized using modular cycle reduction.

### Chronological Holdout Evaluation (Leakage-Free)
All models were trained strictly on the first 80% chronologically ordered records ($N_{\text{train}} = 8,155$) and evaluated on the subsequent 20% held-out test set ($N_{\text{test}} = 2,039$). Preprocessing scalers and encoders were fit strictly on the training partition.

| Model / Estimator | Test MAE (days) | Test RMSE (days) | Test $R^2$ | Time-Series CV RMSE (5-Fold) | Parsimony Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Carrier Mean Baseline (4 Parameters)** | **0.9275** | **1.1504** | **0.6068** | **1.1711 ± 0.038** | **Best Predictive Accuracy** |
| **Ridge Regression (Continuous Surface)** | 0.9279 | 1.1527 | 0.6052 | 1.1738 ± 0.040 | **Selected for Simulation** |
| **Gradient Boosting Regressor** | 0.9297 | 1.1610 | 0.5995 | 1.1824 ± 0.042 | Overfitting on small non-linearities |
| **Random Forest Regressor** | 0.9437 | 1.1883 | 0.5804 | 1.2051 ± 0.045 | Overfitting on training noise |

### Statistical Parsimony Finding & Model Selection Rationale
1. **The 4-parameter Carrier Mean Baseline** (predicting the mean historical dispatch lead time of each `Ship Mode`: First Class $\approx 2.19$d, Second Class $\approx 4.04$d, Standard Class $\approx 5.01$d, Same Day $\approx 0.04$d) outperforms Ridge Regression by **0.0023 days (~3.3 minutes)**.
2. A paired two-sample t-test on holdout residuals confirms this difference is **statistically insignificant** ($t = -1.8788$, $p = 0.0604$).
3. **Crucial Scientific Transparency:** **No claim of ML predictive superiority is made.** Carrier mode accounts for ~98% of all explainable lead-time variance. Correlation between geographic distance and dispatch lead time is near-zero ($r = +0.006$).
4. **Why Ridge Regression is retained:** The Carrier Mean Baseline has a zero spatial derivative ($\partial \hat{y} / \partial \text{Factory} = 0$), rendering it mathematically incapable of evaluating geographic relocations. Ridge Regression provides a regularized, monotonic response surface for spatial simulation without overfitting.

---

## 4. Counterfactual Reallocation Engine & Categories

Every evaluated SKU-factory scenario is classified into one of four transparent operational categories:

1. **Potential Opportunity:** Favorable predicted dispatch reduction, significant freight distance reduction, and robust historical sample size ($\ge 20$ orders).
2. **Needs Validation:** Scenario appears favorable, but cost savings depend heavily on unverified transport heuristics ($\$0.20/\text{ton-km}$, $\$0.08/\text{unit-day}$) or unconfirmed factory tooling compatibility.
3. **Insufficient Evidence:** Low historical volume ($<20$ total orders or $<5$ regional orders). Counterfactual simulation is suppressed or explicitly flagged with a warning banner to prevent misleading management decisions.
4. **Operationally Infeasible:** Haulage distance exceeds 3,800 km, requires cross-border customs clearance without cold chain, or violates known tooling constraints.

---

## 5. UI/UX Design System (Swiss Modernist + Glassmorphism)

The Streamlit interface delivers an enterprise boardroom visual experience:
- **Typography:** Swiss Modernist typographic scale (Outfit, Inter, Space Grotesk) with crisp contrast.
- **Color Palette:** Deep Forest Green (`#0A1D13`, `#112F20`), Warm Ivory background (`#F9F8F5`), Frosted Mint highlights (`#5EEAD4`), and subtle border translucency (`rgba(255, 255, 255, 0.45)`).
- **Navigation:** 7 dedicated operational views:
  1. *Executive Overview:* KPI summary, 3-tier data provenance strip, lead time distribution, top opportunities.
  2. *Product Analysis:* Deep-dive single SKU evaluation, current vs alternative factory comparison cards, order volume distribution.
  3. *Factory Footprint:* Plant workload utilization, regional fulfillment maps, geographical centroid analysis.
  4. *Recommendations:* Multi-criteria scenario filtering, category tabs, impact quadrant chart, full scenario export.
  5. *Route Optimization:* Haulage distance reduction, freight corridor clustering, carbon footprint estimates.
  6. *Model Performance:* Benchmark comparison table, parsimony explanation, true holdout parity scatter plot.
  7. *Executive Reporting:* One-click Markdown/HTML audit export, sensitivity analysis slider, PDF-ready synthesis.

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.10 or higher
- Git & Virtualenv

### Local Setup
```bash
# 1. Clone repository
git clone https://github.com/veereshmv88/nassau-intelligence.git
cd nassau-intelligence

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Retraining & Evaluating Models
To re-run the leakage-free training pipeline, cross-validation, and diagnostic benchmarking:
```bash
python src/train.py
```
This updates `models/best_lead_time_pipeline.joblib`, `models/model_metrics.json`, and `models/feature_importance.json`.

### Executing Automated Test Suite
To verify statistical integrity, leakage prevention, bounding, and UI logic:
```bash
python -m pytest tests/ -v
```
Current suite status: **23 passed in ~14s (100% pass rate)**.

---

## 7. Directory Structure

```
nassau-intelligence/
├── .streamlit/
│   └── config.toml                    # Streamlit server & theme configuration
├── assets/                            # Brand assets and factory visual imagery
│   ├── candy_sidebar.jpg
│   ├── factory_choccy.jpg
│   ├── factory_secret.jpg
│   ├── factory_sugar.jpg
│   └── hero_factory.jpg
├── data/
│   └── raw/
│       └── nassau_candy_distributor.csv # Historical shipment manifest (10,194 records, ~1.98 MB)
├── models/
│   ├── best_lead_time_pipeline.joblib # Serialized champion Ridge Regression pipeline
│   ├── feature_importance.json        # Calibrated feature coefficients and importance
│   └── model_metrics.json             # Benchmark metrics, CV scores, and paired t-test results
├── reports/
│   ├── factory_reallocation_optimization_report.md # Final engineering & validation delivery report
│   └── ml_and_calculation_audit.md    # Deep-dive 5-section calculation & ML audit report
├── src/
│   ├── __init__.py                    # Source package marker
│   ├── chart_builder.py               # Visual analytics, network maps, and distribution plots
│   ├── config.py                      # Master paths, coordinates, themes, and baseline mappings
│   ├── data_loader.py                 # Leakage-free ingestion, coordinate lookup, cleaning
│   ├── optimization_engine.py         # Counterfactual simulation engine, evidence categorizer
│   ├── prediction_service.py          # Serialized model inference with fallback and clipping
│   ├── preprocessing.py               # Haversine distance, scaling, one-hot encoding pipelines
│   ├── reporting.py                   # Executive summary and downloadable audit generator
│   ├── route_analytics.py             # Haulage distance analytics and freight corridor clustering
│   └── train.py                       # Leakage-free training, baseline comparison, time-series CV
├── styles/
│   └── main.css                       # Swiss Modernist glassmorphism design system tokens
├── tests/
│   ├── test_all.py                    # End-to-end integration tests (8 tests)
│   └── test_audit_ml_and_logic.py     # Rigorous ML audit, leakage, bounds, and baseline tests (15 tests)
├── .gitignore                         # Strict exclusion rules for environments, caches & local data
├── app.py                             # Main Streamlit Swiss Modernist application (7 analytical views)
├── README.md                          # Production documentation and methodology
└── requirements.txt                   # Pinned runtime dependencies
```

---

## 8. Deployment Options

### Streamlit Community Cloud
1. Fork or push to GitHub (`veereshmv88/nassau-intelligence`).
2. Navigate to [share.streamlit.io](https://share.streamlit.io).
3. Select repository `veereshmv88/nassau-intelligence`, branch `main`, and main file path `app.py`.
4. Deploy with one click. Dependencies will automatically resolve from `requirements.txt`.

### Docker Container Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 9. Verified Core Assumptions & Operational Limitations

1. **Dispatch vs Delivery Semantic:** `Ship Date - Order Date` represents factory processing and carrier dispatch handover. It does **not** capture road freight transit days to customer doors.
2. **Tooling & Asset Reconfigurability:** Counterfactual scenarios assume candy molds, enrobing lines, and nut-allergen certifications can be supported at alternative facilities.
3. **No Dual-Sourcing Precedent:** Because no SKU historically was produced at multiple plants, individual factory fixed-effects cannot be disentangled from SKU recipe complexities.
4. **Cost Model Heuristics:** Transportation savings ($\$0.20/\text{ton-1000km}$) and carrying cost ($\$0.08/\text{unit-day}$) are operational heuristics that must be replaced with contracted freight tariffs before financial commitments.
5. **Mandatory Pilot Deployment:** Any candidate SKU reallocation (e.g. *Wonka Bar* to *Secret Factory*) must undergo a controlled 90-day shadow pilot with capacity and quality auditing.

---

## 10. License & Governance
Proprietary & Confidential — Nassau Candy Distributor Supply Chain Solutions. All rights reserved. Analytical models governed by Nassau Operational Validation Protocols.
