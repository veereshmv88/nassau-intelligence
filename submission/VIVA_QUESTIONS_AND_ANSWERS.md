# NASSAU INTELLIGENCE — VIVA QUESTIONS & ANSWERS
## Comprehensive Academic Defense & Technical Interview Preparation (20 Questions)

---

### Q1: What is the core problem statement of Nassau Intelligence?
**Answer:**  
In Nassau Candy Distributor's supply chain, 15 candy products were historically manufactured at only one assigned factory across five US locations. Because products were not manufactured close to regional demand centroids, shipments traveled unnecessary distances. The system solves this by analyzing historical dispatch data, predicting fulfillment latency across factories, and simulating counterfactual product reallocations to reduce freight distance while respecting operational constraints.

---

### Q2: What is the major historical data limitation regarding factory assignments?
**Answer:**  
In the 10,194 historical orders, **no single SKU was ever dual-sourced** (manufactured at more than one factory). For example, *Wonka Bar - Milk Chocolate* was produced exclusively at Savannah, GA. Consequently, historical data contains zero empirical examples of what happens when that bar is produced in Illinois or Arizona. Any estimate of lead time at an alternative factory is therefore a **counterfactual scenario simulation**, not an observed fact.

---

### Q3: What is the exact definition and scope of your target metric, "Lead Time"?
**Answer:**  
Lead time is strictly defined as **Order-to-Dispatch Factory Latency** ($T_{\text{dispatch}} = \text{Ship Date} - \text{Order Date}$). It measures the internal elapsed time from when an order is placed to when the factory stages and hands off the pallet to the third-party carrier. It explicitly **does not** measure door-to-door transit time across highways to customer doorsteps.

---

### Q4: Why did raw order and ship dates in the CSV show differences of 900 to 1,600 days?
**Answer:**  
The raw dataset contains a synthetic operational calendar artifact where order dates are recorded in 2024–2025 while ship dates were projected into future years (2026–2030). Applying a modular reduction modulo 365 ($(\text{raw\_days} \pmod{365}) - 174$) cleanly normalizes this annual offset, recovering the true operational lead times spanning **0 to 7 days** ($Mean = 3.96$ days, $Median = 4.0$ days), matching real-world carrier service levels.

---

### Q5: How did you compute geographic distance between factories and customer destinations?
**Answer:**  
We mapped each of the 5 factories to their exact geographic coordinates (latitude and longitude) and mapped each order's destination state/region to its centroid coordinates. We then computed great-circle surface distance using the **Haversine formula**:
$$d = 2R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$
where $R = 6,371\text{ km}$.

---

### Q6: How was the dataset split, and why didn't you use random K-Fold cross-validation?
**Answer:**  
We used a strict **Chronological Holdout Split** based on `Order Date`. The first 80% chronologically ordered records ($N_{\text{train}} = 8,155$) formed the training set, and the subsequent 20% ($N_{\text{test}} = 2,039$) formed the test set. Random $K$-Fold cross-validation was strictly avoided because shuffling time-series order data creates **look-ahead temporal data leakage**, where future demand patterns artificially inflate past predictions.

---

### Q7: What is data leakage, and how did your pipeline prevent it?
**Answer:**  
Data leakage occurs when information from outside the training partition influences the model. We prevented it in two ways:
1. **Temporal separation:** Testing records strictly succeed training records in time.
2. **Preprocessing encapsulation:** Feature transformers (`StandardScaler`, `OneHotEncoder`) were fit strictly on the training partition (`X_train`) and only used to transform the test set (`X_test`), preventing test distribution leakage.

---

### Q8: What is the Carrier Mean Baseline, and what are its parameters?
**Answer:**  
The Carrier Mean Baseline is an empirical operational benchmark that predicts dispatch lead time based strictly on the historical average for each carrier shipping mode:
- *Same Day:* 0.043 days (~1 hour)
- *First Class:* 2.193 days
- *Second Class:* 4.041 days
- *Standard Class:* 5.012 days  
It uses only **4 parameters** and serves as our parsimony reference under Occam's razor.

---

### Q9: What were the exact evaluation metrics of the Carrier Baseline vs Ridge Regression?
**Answer:**  
On the 2,039 chronologically held-out test orders:
- **Carrier Mean Baseline (4 params):** $MAE = 0.9275$ days, $RMSE = 1.1504$ days, $R^2 = 0.6068$.
- **Ridge Regression Pipeline (33 features):** $MAE = 0.9279$ days, $RMSE = 1.1527$ days, $R^2 = 0.6052$.
- **5-Fold Time-Series CV RMSE:** Carrier Baseline: $1.1082 \pm 0.038$ vs Ridge: $1.1117 \pm 0.040$.

---

### Q10: Did Ridge Regression outperform the Carrier Baseline? What was the paired t-test result?
**Answer:**  
No, Ridge Regression did not outperform the baseline. The simple 4-parameter Carrier Baseline achieved a slightly lower RMSE by **0.0023 days (~3.3 minutes)**. A paired two-sample $t$-test on holdout residuals yielded:
$$t = -1.8788, \quad p = 0.0604$$
Because $p > 0.05$, the difference is **not statistically significant**. The two models exhibit statistical parity on historical orders.

---

### Q11: If the Carrier Baseline had a slightly lower RMSE, why is Ridge Regression used in the simulation engine?
**Answer:**  
The Carrier Baseline has a **zero spatial derivative**:
$$\frac{\partial \hat{y}}{\partial \text{Factory}} = 0$$
Because the baseline only maps `Ship Mode`, if you shift a product from Arizona to Georgia, the baseline predicts the exact same lead time. It is mathematically incapable of spatial counterfactual simulation. Ridge Regression incorporates continuous geographic distance and regional terms, providing a regularized, monotonic response surface suitable for spatial evaluation.

---

### Q12: Why did Gradient Boosting and Random Forest perform worse than linear models on the test set?
**Answer:**  
Gradient Boosting ($RMSE = 1.1610$ d) and Random Forest ($RMSE = 1.1883$ d) suffered from **overfitting on historical noise**. Non-linear tree splits attempted to capture spurious interactions between specific product names and shipping dates that did not generalize to future chronological test records.

---

### Q13: What is the mathematical formulation of Ridge Regression, and why is L2 regularization used?
**Answer:**  
Ridge Regression minimizes the residual sum of squares plus an $L_2$ penalty on the magnitude of the coefficients:
$$\mathcal{L}(\beta) = \|y - X\beta\|_2^2 + \alpha \|\beta\|_2^2 \quad (\alpha = 1.0)$$
The $L_2$ shrinkage prevents multicollinearity between one-hot encoded categories and stabilizes coefficient weights, preventing extreme predictions when evaluating unobserved factory-destination pairings.

---

### Q14: What are the four evidential categories in the Reallocation Engine?
**Answer:**  
1. **Potential Opportunity:** High order volume ($N \ge 20$), significant simulated distance and lead-time reductions. Ideal for pilot trials.
2. **Needs Validation:** Positive metrics, but savings depend on unverified freight tariffs ($\$0.20/\text{ton-km}$) or unconfirmed factory tooling/allergen compatibility.
3. **Insufficient Evidence:** Low sample size ($N < 20$ total or $N < 5$ regional orders). Counterfactual simulation is suppressed or flagged with a warning banner.
4. **Operationally Infeasible:** Reallocation haulage distance exceeds 3,800 km, requires cross-border cold chain without facilities, or violates physical equipment constraints.

---

### Q15: How does the system ensure mathematical safety against edge cases?
**Answer:**  
1. **Non-negative bounding:** All lead time predictions are clipped to $[0, 7]$ days using `np.clip()`.
2. **Division-by-zero protection:** Gross profit margin calculations use conditional checks or small epsilons so zero-revenue records return 0.0% instead of `NaN` or `inf`.
3. **Self-reallocation guard:** The engine filters out scenarios where the candidate factory is identical to the current factory.

---

### Q16: How does K-Means clustering work in your Route Analytics module?
**Answer:**  
We apply $K$-Means clustering ($k=4$) on normalized factory and regional destination coordinates to segment 10,194 historical shipments into four dominant **freight corridors** (Atlantic, Interior, Gulf, Pacific). This allows logistics managers to aggregate regional freight volume and negotiate consolidated carrier volume discounts.

---

### Q17: What technology stack was used, and why did you select Streamlit?
**Answer:**  
- **Core:** Python 3.10
- **Data & Math:** Pandas 2.0+, NumPy 1.24+, SciPy 1.11+
- **Machine Learning:** Scikit-learn 1.3+ (Pipeline, Ridge, OneHotEncoder, StandardScaler, TimeSeriesSplit)
- **Data Visualization:** Altair 5.0+ and native Streamlit interactive charts
- **Interface:** Streamlit 1.30+ with custom Swiss Modernist CSS (`styles/main.css`)
- **Testing:** Pytest 8.0+  
Streamlit was chosen because it allows rapid deployment of state-of-the-art interactive data science dashboards in pure Python with reactive state management.

---

### Q18: What automated tests were implemented, and what are the results?
**Answer:**  
We implemented **23 automated tests** across two files:
- `tests/test_all.py` (8 tests): Validates data ingestion, Haversine geometry, inference pipeline, ranking, and route clustering.
- `tests/test_audit_ml_and_logic.py` (15 tests): Audits temporal leakage, parsimony benchmarks, paired $t$-test values, bounding safety, and evidence stratification.  
**Result:** 23 out of 23 passed (100% pass rate in ~11.5 seconds).

---

### Q19: What are the primary operational limitations of Nassau Intelligence?
**Answer:**  
1. **Unobserved Dual-Sourcing:** Simulations are counterfactual estimates; historical data cannot guarantee operational outcomes without an on-site pilot.
2. **Allergen & Tooling Isolation:** Reallocating nut-based candy to a nut-free plant requires capital investment in allergen barriers.
3. **Dispatch vs Transit Semantic:** Lead time measures internal plant processing to carrier handover, not final customer delivery.

---

### Q20: What future enhancements would you propose if given more time?
**Answer:**  
1. **Mixed-Integer Linear Programming (MILP):** Integrate solvers like PuLP or SciPy `linprog` to enforce hard plant capacity limits (tons/month).
2. **Live Carrier Rate API:** Integrate FedEx or UPS Freight APIs to replace heuristic transportation costs with dynamic contractual tariffs.
3. **Dual-Sourcing Split:** Extend the engine to optimize fractional production allocations (e.g., 70% Chicago plant, 30% Atlanta plant).
