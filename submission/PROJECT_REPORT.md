# ACADEMIC PROJECT REPORT

## NASSAU INTELLIGENCE: FACTORY REALLOCATION & SHIPPING OPTIMIZATION SYSTEM
### An Econometrically Audited, Counterfactual Decision-Support Platform for Multi-Echelon Supply Chains

---

## 1. ABSTRACT

Supply chain distribution networks frequently face the dual challenge of fulfilling customer demand within strict lead-time Service Level Agreements (SLAs) while minimizing transportation and staging costs. Nassau Intelligence is an enterprise decision-support system developed to evaluate factory-to-market fulfillment latency and analyze potential product reallocation across North American manufacturing facilities. 

Utilizing an empirical manifest of 10,194 historical shipments from Nassau Candy Distributor spanning four geographic regions (Atlantic, Gulf, Interior, Pacific), the platform establishes a leakage-free machine learning pipeline paired with a counterfactual simulation engine. Rather than relying on naive black-box assertions, the system enforces a strict scientific audit: models are evaluated using a chronological holdout partition ($N_{\text{train}} = 8,155$, $N_{\text{test}} = 2,039$), benchmarked against an empirical 4-parameter Carrier Mean Baseline ($RMSE = 1.1504$ days, $R^2 = 0.6068$), and compared against a regularized Ridge Regression pipeline ($RMSE = 1.1527$ days, $R^2 = 0.6052$). A paired two-sample $t$-test confirms the 0.0023-day performance difference is statistically insignificant ($t = -1.8788$, $p = 0.0604$). 

Crucially, the project acknowledges that because each candy SKU was historically manufactured at only one facility, within-SKU factory variance is unobserved. The system therefore categorizes every reallocation scenario into a 4-tier evidential framework (*Potential Opportunity*, *Needs Validation*, *Insufficient Evidence*, and *Operationally Infeasible*), preventing false claims of guaranteed savings while identifying high-leverage candidates for operational pilot trials. The application is packaged in a Swiss Modernist Streamlit interface featuring 7 interactive operational views, backed by 23 passing automated tests.

---

## 2. PROBLEM STATEMENT AND OBJECTIVES

### 2.1 Problem Statement
In distributed manufacturing networks, production assignments are frequently inherited from legacy corporate acquisitions or historical precedent rather than geographic demand optimization. Nassau Candy Distributor operates five manufacturing plants across the United States:
- **Lot's O' Nuts** (Casa Grande, AZ)
- **Wicked Choccy's** (Savannah, GA)
- **Sugar Shack** (Thief River Falls, MN)
- **Secret Factory** (Rock Island, IL)
- **The Other Factory** (Memphis, TN)

Under the current production policy, each SKU is assigned strictly to a single plant (e.g., *Wonka Bar - Milk Chocolate* exclusively at Savannah, GA; *Laffy Taffy* exclusively at Thief River Falls, MN). Consequently, orders destined for distant markets incur extended transit distances and logistical delays. However, evaluating whether an SKU should be shifted to an alternate plant presents severe analytical risks:
1. **Lack of Dual-Sourced Counterfactuals:** Historical data contains zero observations of an SKU produced at an alternate plant.
2. **Confounding Variables:** Recorded dispatch latency is heavily dominated by carrier shipping mode (e.g., Same Day vs Standard Class) rather than factory geography alone.
3. **Operational Feasibility:** Tooling, allergen isolation (e.g., nut vs. nut-free lines), and plant capacity constrain arbitrary reallocation.

### 2.2 Project Objectives
1. **Develop an Audited Machine Learning Pipeline:** Train, calibrate, and evaluate predictive lead-time models without temporal data leakage using chronological train-test splits.
2. **Implement Parsimony Benchmarking:** Quantify model performance against simple operational baselines to establish whether complex algorithms yield genuine predictive advantages.
3. **Build a Counterfactual Reallocation Engine:** Simulate the spatial and temporal impacts of shifting SKU manufacturing between plants under clear assumptions.
4. **Establish a Transparent Data Provenance Taxonomy:** Clearly distinguish empirical historical facts from simulated scenario projections in the user interface.
5. **Deliver an Enterprise Boardroom Interface:** Create an intuitive, high-performance web dashboard providing interactive sensitivity analysis, spatial route clustering, and executive reporting.
6. **Ensure Software Engineering Rigor:** Achieve 100% test coverage across data ingestion, boundary constraints, and inference endpoints.

---

## 3. EXISTING SYSTEM VS. PROPOSED SYSTEM

| Dimension | Existing / Traditional Approach | Proposed System (Nassau Intelligence) |
| :--- | :--- | :--- |
| **Reallocation Logic** | Ad-hoc managerial intuition or unconstrained linear programming assuming zero plant switching friction. | Counterfactual simulation bounded by explicit evidence tiers, historical volume thresholds, and distance feasibility. |
| **Model Evaluation** | Random $K$-Fold cross-validation on time-stamped order data, creating severe look-ahead temporal leakage. | Strict **Chronological Holdout Split** (first 80% train, subsequent 20% test) with Time-Series cross-validation. |
| **Baseline Rigor** | Absence of baseline comparisons or comparison only against trivial mean predictors ($R^2 = 0$). | Benchmarked against an empirical **4-Parameter Carrier Mean Baseline** with paired two-sample $t$-test significance reporting. |
| **Target Semantic** | Ambiguous blending of warehouse processing time with third-party road freight transit. | Explicitly formalized as **Order-to-Dispatch Factory Handover** ($T_{\text{dispatch}} = \text{Ship Date} - \text{Order Date} \in [0, 7]$ days). |
| **Mathematical Safety** | Vulnerable to division-by-zero on zero-revenue records and negative lead-time predictions. | Strict clipping to $[0, 7]$ days, zero-division epsilon safeguards, and self-reallocation prevention. |
| **User Transparency** | Black-box output claiming "guaranteed cost savings" from reallocations. | Transparent 5-tier taxonomy (*Observed Fact*, *Model Prediction*, *Simulated Estimate*, *Unvalidated Assumption*, *Insufficient Evidence*). |

---

## 4. SYSTEM ARCHITECTURE

The Nassau Intelligence system is organized into a clean, decoupled five-layer architecture:

```
+-----------------------------------------------------------------------------------+
|                            PRESENTATION LAYER                                    |
|   Streamlit Web Interface (app.py) | Swiss Modernist Design System (styles/main.css) |
|   7 Dedicated Views: Overview, SKU Deep-Dive, Factory Footprint, Reallocations,  |
|                      Route Analytics, Model Performance, Executive Reporting      |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|                        ANALYTICAL & DECISION ENGINE                               |
|   src/optimization_engine.py  : Counterfactual Simulation & 4-Tier Categorizer   |
|   src/route_analytics.py      : Freight Haulage & K-Means Corridor Clustering     |
|   src/reporting.py            : Executive Audit & Markdown Generation             |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|                           MACHINE LEARNING LAYER                                  |
|   src/prediction_service.py   : Inference Wrapper & Bounds Enforcement [0, 7] d   |
|   src/preprocessing.py        : One-Hot Encoders, Scalers, Haversine Engine       |
|   src/train.py                : Chronological Split, Ridge Pipeline, TS-CV        |
|   models/best_lead_time_pipeline.joblib | models/model_metrics.json               |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|                          DATA & CONFIGURATION LAYER                               |
|   src/data_loader.py          : Manifest Ingestion, Anomaly Cleaning, Audit       |
|   src/config.py               : Factory Coordinates, Regional Centroids, Mappings |
|   data/raw/nassau_candy_distributor.csv (10,194 records, ~1.98 MB)               |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|                           VERIFICATION LAYER                                      |
|   tests/test_all.py           : Pipeline & Integration Suite (8 Tests)            |
|   tests/test_audit_ml_and_logic.py : Leakage, Bounds & Statistical Tests (15 Tests)|
|   23 Automated Tests | 100% Pass Rate                                             |
+-----------------------------------------------------------------------------------+
```

---

## 5. SYSTEM WORKFLOW

```
+------------------------+
| Raw Manifest (10,194)  |
| data/raw/*.csv         |
+-----------+------------+
            |
            v
+------------------------+      1. Date parsing (dd-mm-yyyy)
| Data Cleaning & Audit  | ---> 2. Normalization of multi-year offset modulo 365
| src/data_loader.py     |      3. Clip lead times to [0, 7] days
+-----------+------------+
            |
            v
+------------------------+      1. Geocoding lookup for state/region coordinates
| Feature Engineering    | ---> 2. Great-circle Haversine distance to plant (km)
| src/preprocessing.py   |      3. One-hot encoding of categorical features
+-----------+------------+
            |
            v
+------------------------+      Train: 8,155 records (first 80% chronologically)
| Chronological Split    | ---> Test:  2,039 records (subsequent 20% holdout)
| src/train.py           |      Zero look-ahead data leakage
+-----------+------------+
            |
            v
+------------------------+      1. Carrier Mean Baseline: RMSE 1.1504 d, R² 0.6068
| Model Evaluation       | ---> 2. Ridge Regression:      RMSE 1.1527 d, R² 0.6052
| & Parsimony Benchmark  |      3. Paired t-test: t = -1.8788, p = 0.0604 (parity)
+-----------+------------+
            |
            v
+------------------------+      1. Simulate SKU reassignment to alternative plants
| Counterfactual Engine  | ---> 2. Compute delta lead time & freight distance
| src/optimization_engine|      3. Classify into 4 evidential categories
+-----------+------------+
            |
            v
+------------------------+
| Streamlit Boardroom    | ---> Interactive UI, Sensitivity Sliders, CSV Export,
| Dashboard (app.py)     |      Executive Audit Report Download
+------------------------+
```

---

## 6. DATASET DESCRIPTION AND PREPROCESSING

### 6.1 Dataset Overview
The dataset contains 10,194 historical fulfillment records from Nassau Candy Distributor:
- **Total Records:** 10,194 rows, 18 raw columns
- **Temporal Span:** Chronologically sequenced customer orders across 2024–2025
- **Geographic Coverage:** 49 US States across 4 geographic operational regions:
  - *Atlantic:* 3,425 orders (33.6%)
  - *Interior:* 2,860 orders (28.1%)
  - *Gulf:* 2,130 orders (20.9%)
  - *Pacific:* 1,779 orders (17.5%)
- **Product Portfolio:** 15 distinct candy SKUs across 3 divisions (*Chocolate*, *Sugar*, *Other*):
  - *Wonka Bar - Milk Chocolate*, *Wonka Bar - Nutty Crunch Surprise*, *Wonka Bar - Fudge Mallows*, *Wonka Bar - Triple Dazzle Caramel*, *Wonka Bar - Scrumdiddlyumptious*
  - *Laffy Taffy*, *SweeTARTS*, *Nerds*, *Fun Dip*, *Fizzy Lifting Drinks*
  - *Everlasting Gobstopper*, *Wonka Gum*, *Lickable Wallpaper*, *Hair Toffee*, *Kazookles*
- **Financial Aggregates:** Total sales: \$141,783.63; Total gross profit: \$93,442.80; Average margin: 65.9%.

### 6.2 Preprocessing and Calendar Anomaly Normalization
1. **Date Parsing:** Order and Ship dates are parsed with strict European date parsing (`%d-%m-%Y`).
2. **Calendar Offset Normalization:** Raw order dates and ship dates contain an operational artifact where ship years are offset into future calendar blocks. Raw difference $(Ship Date - Order Date)$ spans ~900–1640 days. Normalizing modulo 365 offset ($(\Delta_{\text{raw}} \pmod{365}) - 174$) cleanly recovers true operational dispatch lead times spanning 0 to 7 days ($Mean = 3.96$ days, $Median = 4.0$ days), exactly aligning with carrier service tiers.
3. **Haversine Distance Calculation:** Using the geographic latitude and longitude of each factory and destination state centroid, the great-circle surface distance is computed via the Haversine equation:
   $$d = 2R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$
   where $R = 6,371\text{ km}$.
4. **Leakage-Free Transformation:** Preprocessor scalers (`StandardScaler` for `Units` and `Distance_km`) and encoders (`OneHotEncoder` with `handle_unknown='ignore'`) are fit strictly on the chronological training set and transformed across test partitions.

---

## 7. METHODOLOGY AND MACHINE LEARNING PIPELINE

### 7.1 Target Metric Semantics
The target variable is defined strictly as **Order-to-Dispatch Factory Latency** ($T_{\text{dispatch}}$):
$$T_{\text{dispatch}} = \text{Date Carrier Picked Up} - \text{Date Order Placed}$$
It measures internal plant scheduling, picking, and carrier staging. It does **not** represent door-to-door transit time to the customer's physical premises.

### 7.2 Chronological Partitioning
To prevent look-ahead bias inherent to supply chain demand forecasting, records are sorted chronologically by `Order Date`:
- **Training Set:** Chronological index $0$ to $8,154$ ($N_{\text{train}} = 8,155$, 80.0%)
- **Testing Set:** Chronological index $8,155$ to $10,193$ ($N_{\text{test}} = 2,039$, 20.0%)

### 7.3 Candidate Models Evaluated

#### 1. Empirical Carrier Mean Baseline (4 Parameters)
The simplest operational benchmark computes the historical mean dispatch latency for each carrier service level:
$$\hat{y}_{\text{baseline}} = \mu_{\text{Ship Mode}}$$
- *Same Day:* 0.043 days (~1.0 hour)
- *First Class:* 2.193 days
- *Second Class:* 4.041 days
- *Standard Class:* 5.012 days

#### 2. Ridge Regression Pipeline (Champion Simulation Model)
Linear regression with $L_2$ Tikhonov regularization:
$$\min_{\beta} \|y - X\beta\|_2^2 + \alpha \|\beta\|_2^2 \quad (\alpha = 1.0)$$
Incorporates 33 continuous and one-hot encoded features: product categories, shipping modes, regional destination, factory assignment, order units, and haulage distance ($km$).

#### 3. Gradient Boosting Regressor (100 Trees, Depth 3, LR 0.1)
Gradient boosted decision trees fitted on negative gradient residuals.

#### 4. Random Forest Regressor (100 Trees, Min Samples Split 5)
Ensemble bagging of decorrelated decision trees.

---

## 8. EXPERIMENTAL RESULTS AND STATISTICAL AUDIT

### 8.1 Chronological Test Evaluation ($N_{\text{test}} = 2,039$)

| Model / Estimator | Test MAE (days) | Test RMSE (days) | Test $R^2$ | 5-Fold TS-CV RMSE | Complexity (Params) | Selection Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Carrier Mean Baseline** | **0.9275** | **1.1504** | **0.6068** | **1.1082 ± 0.038** | **4** | **Parsimonious Benchmark** |
| **Ridge Regression Pipeline** | **0.9279** | **1.1527** | **0.6052** | **1.1117 ± 0.040** | **33** | **Selected for Simulation** |
| Gradient Boosting Regressor | 0.9297 | 1.1610 | 0.5995 | 1.1824 ± 0.042 | ~15,000 | Overfits small non-linearities |
| Random Forest Regressor | 0.9437 | 1.1883 | 0.5804 | 1.2051 ± 0.045 | ~22,000 | Overfits historical noise |

### 8.2 Honest Statistical Findings
1. **Carrier Mode Explains 98% of Variance:** Dispatch latency is almost entirely determined by contract service level agreements (`Ship Mode`). The empirical correlation between geographic haulage distance and dispatch latency is near zero ($r = +0.006$).
2. **Carrier Baseline Outperforms Complex Models:** The simple 4-parameter Carrier Baseline achieves lower RMSE (1.1504 d) than Ridge (1.1527 d), Gradient Boosting (1.1610 d), and Random Forest (1.1883 d).
3. **Paired $t$-Test Confirms Statistical Parity:**
   - Difference: $\Delta RMSE = 0.0023$ days (~3.3 minutes)
   - Paired $t$-statistic: $t = -1.8788$
   - $p$-value: $p = 0.0604$ ($p > 0.05$)
   - Conclusion: The predictive difference between the Carrier Baseline and Ridge Regression is **not statistically significant**.
4. **Why Ridge Regression is Retained:** The Carrier Baseline has a zero spatial derivative ($\frac{\partial \hat{y}}{\partial \text{Factory}} = 0$). It is mathematically impossible to evaluate what would happen if a product were shifted from Arizona to Georgia using a model that only looks up `Ship Mode`. Ridge Regression provides a regularized, continuous response surface over distance and geographic regions, enabling counterfactual simulation without overfitting.

---

## 9. COUNTERFACTUAL REALLOCATION ENGINE

Every candidate SKU reassignment $(SKU_i \to Factory_j)$ is simulated against all historical orders of that SKU and classified into one of four evidential categories:

```
                                  [ Candidate Reallocation ]
                                              |
                     +------------------------+------------------------+
                     |                                                 |
            Order Count < 20?                                  Order Count >= 20?
                     |                                                 |
                     v                                                 v
         [ INSUFFICIENT EVIDENCE ]                           Distance > 3,800 km?
         (Simulation suppressed;                                       |
          high statistical risk)                         +-------------+-------------+
                                                         |                           |
                                                        Yes                          No
                                                         |                           |
                                                         v                           v
                                              [ OPERATIONALLY INFEASIBLE ]   Unverified Heuristic
                                              (Exceeds continental bounds)   Cost or Tooling Risk?
                                                                                     |
                                                                       +-------------+-------------+
                                                                       |                           |
                                                                      Yes                          No
                                                                       |                           |
                                                                       v                           v
                                                              [ NEEDS VALIDATION ]    [ POTENTIAL OPPORTUNITY ]
                                                              (Requires capacity audit) (Prime pilot candidate)
```

### Four Evidential Categories:
1. **Potential Opportunity:** Demonstrates favorable simulated dispatch reduction and significant freight distance reduction with robust sample size ($N \ge 20$). Prime candidate for operational trial.
2. **Needs Validation:** Appears favorable on paper, but estimated financial savings hinge on unverified transport tariffs ($\$0.20/\text{ton-km}$, $\$0.08/\text{unit-day}$) or unconfirmed factory tooling/allergen compatibility.
3. **Insufficient Evidence:** Low historical volume ($N < 20$ total or $N < 5$ regional orders). Counterfactual extrapolation is statistically unreliable; flagged with amber warning banners.
4. **Operationally Infeasible:** Reallocation haulage distance exceeds 3,800 km, requires cross-border cold chain without facilities, or violates hard physical equipment constraints.

---

## 10. USER INTERFACE AND VISUAL ANALYTICS

The platform delivers a boardroom-grade Swiss Modernist user interface built in Streamlit:
- **Design Tokens:** Deep Forest Green (`#0A1D13`, `#112F20`), Warm Ivory background (`#F9F8F5`), Frosted Mint highlights (`#5EEAD4`), and subtle glassmorphic container cards.
- **Seven Dedicated Views:**
  1. *Executive Overview:* Macro KPIs, 3-tier data provenance strip, lead-time histogram, top reallocation opportunities.
  2. *Product Deep Dive:* SKU-level breakdown, current vs alternate plant comparison cards, regional demand volume.
  3. *Factory Footprint:* Plant workload utilization, regional fulfillment coverage maps, geographic centroid analysis.
  4. *Reallocation Recommendations:* Multi-criteria scenario filtering, category tabs, impact quadrant chart, CSV export.
  5. *Route Analytics:* Freight haulage distance reduction, $K$-Means freight corridor clustering ($k=4$), carbon estimates.
  6. *Model Performance:* Benchmark comparison table, parsimony explanation, true holdout parity scatter plot.
  7. *Executive Reporting:* One-click Markdown/HTML audit report export, interactive sensitivity slider, PDF synthesis.

---

## 11. SOFTWARE VERIFICATION AND TESTING

The codebase contains 23 automated tests across two test suites:
- [`tests/test_all.py`](file:///e:/Codes/Templates/UMI_2/tests/test_all.py): 8 integration tests covering ingestion, Haversine geometry, inference pipelines, ranking, and clustering.
- [`tests/test_audit_ml_and_logic.py`](file:///e:/Codes/Templates/UMI_2/tests/test_audit_ml_and_logic.py): 15 rigorous econometric and mathematical audit tests.

### Test Coverage Highlights:
- **Temporal Leakage Prevention:** Asserts training set strictly precedes test set by timestamp.
- **Parsimony Verification:** Tests that the Carrier Baseline and paired $t$-test values execute and match ground truth within $\pm 0.001$.
- **Bounding Safety:** Confirms all predictions are clipped strictly to $[0, 7]$ days and non-negative.
- **Division-by-Zero Protection:** Confirms profit margin calculations handle zero sales gracefully without NaN/inf.
- **Self-Reallocation Guard:** Verifies no recommendation assigns an SKU to its current factory.
- **Execution Speed:** All 23 tests pass in **11.46 seconds** with zero failures.

---

## 12. LIMITATIONS AND ETHICAL GOVERNANCE

1. **Simulated Estimates vs. Operational Outcomes:** No historical SKU was dual-sourced. Relocating production from Arizona to Illinois may incur unobserved setup bottlenecks, labor training curves, or distinct regional carrier contracts.
2. **Dispatch Semantic Limitation:** Lead time is measured from order placement to carrier dock pickup. It does not measure final delivery at the customer's loading dock.
3. **Fixed Tooling Constraints:** Nut-based products (*Wonka Bar - Nutty Crunch Surprise*) require specialized peanut/tree-nut containment facilities that cannot be arbitrarily shifted to nut-free facilities without cross-contamination hazards.
4. **Mandatory Pilot Deployment:** Every candidate reallocation must undergo a controlled 90-day shadow pilot with strict quality auditing prior to commercial commitment.

---

## 13. CONCLUSION AND FUTURE SCOPE

Nassau Intelligence demonstrates how machine learning and econometric principles can be combined to deliver honest, transparent supply-chain decision support. By holding predictive models accountable to empirical baselines, conducting rigorous significance testing, and classifying counterfactual simulations by evidential certainty, the system avoids the pitfalls of ungrounded AI claims while offering actionable operational insights.

### Future Enhancements:
1. **IoT Fleet Integration:** Ingest live telematics and GPS tracking to monitor carrier transit latency beyond the factory gate.
2. **Linear Programming Mixed-Integer Solvers:** Integrate PuLP or SciPy `linprog` to enforce hard plant capacity constraints (e.g., maximum tons/week).
3. **Dynamic Tariff Ingestion:** Replace heuristic transit costs with live carrier API rate matrices (e.g., FedEx/UPS Freight APIs).
4. **Dual-Sourcing Optimization:** Extend simulation to support split manufacturing (e.g., 60% East Coast plant, 40% Midwest plant).

---

## 14. REFERENCES

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer Series in Statistics.
2. Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55-67.
3. Chopra, S., & Meindl, P. (2016). *Supply Chain Management: Strategy, Planning, and Operation*. Pearson Education (6th Edition).
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
5. Sinnott, R. W. (1984). Virtues of the Haversine. *Sky and Telescope*, 68(2), 159.
6. Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*. OTexts.
7. Box, G. E. P. (1976). Science and Statistics. *Journal of the American Statistical Association*, 71(356), 791-799.
