# NASSAU INTELLIGENCE — DEMONSTRATION & PRESENTATION GUIDE
## Step-by-Step Practical Demonstration Script for Academic Review & Evaluation

---

## 1. Quick Launch Instructions
Open your terminal in the project directory and execute:
```bash
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 2. Five-Minute Boardroom Demonstration Flow

### Scene 1: Introduction & Project Hook (30 Seconds)
- **What to open:** Default landing view — **Executive Overview**.
- **What to show:**
  - The hero title: *Nassau Intelligence — Factory Reallocation & Shipping Optimization*.
  - The 3-tier Data Provenance Strip at the top (`Observed Fact`, `Model Prediction`, `Simulated Estimate`).
  - Macro KPIs: Total Orders (10,194), Active Factories (5), Baseline Lead Time (4.29 days), Total Gross Profit (\$93.4K).
- **What to say:**
  > *"Good morning professors. Nassau Intelligence is an econometrically audited supply chain decision-support system for Nassau Candy Distributor. Our objective is to evaluate whether moving manufacturing of candy products to alternative factories can reduce lead times and haulage distances. Crucially, in our historical data, each product was produced at only one factory. Because no product was dual-sourced, any reallocation is a simulated counterfactual estimate, not an observed fact. We explicitly enforce scientific transparency so business leaders are never misled."*

---

### Scene 2: Defining the Target Metric & Distribution (45 Seconds)
- **What to open:** Scroll down on **Executive Overview** to the *Lead Time Distribution* histogram.
- **What to show:**
  - The bell-shaped lead-time distribution centered at 4.0 days, bounded cleanly between 0 and 7 days.
  - The Target Definition banner.
- **What to say:**
  > *"Notice how our target metric is defined: it is 'Order-to-Dispatch Lead Time'—the time elapsed from order placement until the carrier picks up the pallet at our plant loading dock. It does not measure road freight transit days to customer doorsteps. Raw order manifests contained a synthetic multi-year calendar offset, which our pipeline normalized modulo 365, recovering true operational dispatch lead times of 0 to 7 days."*

---

### Scene 3: Product Analysis & Single SKU Deep-Dive (60 Seconds)
- **What to open:** Select **Product Deep Dive** from the sidebar radio menu.
- **What to show:**
  - Product selector dropdown: select *Wonka Bar - Milk Chocolate* or *Laffy Taffy*.
  - The side-by-side comparison card: Current Factory (*Wicked Choccy's, Savannah GA*) vs Recommended Candidate (*Secret Factory, Rock Island IL*).
  - Regional order distribution bar chart.
- **What to say:**
  > *"Here in the Product Deep Dive, an operational director can inspect any individual SKU. For Wonka Bar - Milk Chocolate, the bulk of consumer demand comes from the Interior and Atlantic regions. Producing it in Savannah, Georgia forces long haulage to Midwest customers. Our model simulates what happens if we shift volume to Secret Factory in Illinois: haulage distance drops by over 400 km per shipment while maintaining dispatch efficiency."*

---

### Scene 4: Reallocation Recommendations & 4-Tier Categorization (60 Seconds)
- **What to open:** Select **Reallocation Recommendations** from the sidebar.
- **What to show:**
  - The four category tabs: `Potential Opportunity`, `Needs Validation`, `Insufficient Evidence`, `Operationally Infeasible`.
  - The Optimization Score Weight slider (Freight Distance vs Lead Time Weight).
  - The Reallocation Impact Quadrant scatter plot (Distance Saved vs Lead Time Saved).
- **What to say:**
  > *"Rather than outputting a blind recommendation, our engine stratifies every scenario into four transparent tiers based on evidential risk:*
  > 1. *'Potential Opportunity' represents robust scenarios with 20+ historical orders and large haulage reductions.*
  > 2. *'Needs Validation' flags scenarios that depend on unverified freight rates or unknown factory tooling reconfigurations.*
  > 3. *'Insufficient Evidence' explicitly suppresses low-volume products to prevent hazardous decision-making.*
  > 4. *'Operationally Infeasible' eliminates cross-continental hauls exceeding 3,800 km.*
  > *Directors can adjust the optimization slider to balance freight distance against internal dispatch speed."*

---

### Scene 5: Scientific Model Performance & Parsimony Benchmark (60 Seconds)
- **What to open:** Select **Model Performance & Parsimony Audit** from the sidebar.
- **What to show:**
  - The Benchmark Evaluation Table comparing Carrier Mean Baseline, Ridge Regression, Gradient Boosting, and Random Forest.
  - The Paired $t$-Test banner ($t = -1.8788$, $p = 0.0604$).
  - The Actual vs Predicted Scatter plot and Feature Importance bar chart.
- **What to say:**
  > *"This is the scientific core of our project. We conducted a strict chronological holdout split—training on the first 8,155 orders and testing on the subsequent 2,039 orders to ensure zero look-ahead data leakage.*
  > *Our audit revealed an important finding: the simple 4-parameter Carrier Mean Baseline achieves an RMSE of 1.1504 days, slightly outperforming Ridge Regression at 1.1527 days. A paired two-sample t-test yields p = 0.0604, confirming the difference is statistically insignificant.*
  > *Carrier shipping mode accounts for ~98% of explainable dispatch variance. We retain Ridge Regression for our simulation engine because the Carrier Baseline has a zero spatial derivative and cannot evaluate geographic relocations. We make no false claims of ML superiority, adhering strictly to Occam's razor and scientific honesty."*

---

### Scene 6: Route Analytics, Freight Corridor Clustering & Audit Export (45 Seconds)
- **What to open:** Select **Haulage & Route Analytics**, then **Executive Audit & Compliance Report**.
- **What to show:**
  - $K$-Means spatial cluster map ($k=4$) highlighting regional freight lanes.
  - The one-click *Download Comprehensive Markdown Audit Report* button.
- **What to say:**
  > *"Finally, our route analytics module clusters 10,194 shipments into 4 optimal transportation corridors using K-Means clustering on geographic centroids. The Executive Reporting view compiles our findings into an exportable audit report for corporate governance and C-suite presentation."*

---

## 3. High-Impact Closing Summary
> *"In summary, Nassau Intelligence transforms raw shipment manifests into an accountable, audit-grade decision system. It couples machine learning with causal awareness, guards against data leakage, and provides actionable recommendations grounded in evidence. Thank you, and we welcome your questions."*
