# 🎤 ChurnIQ — Presentation Kit

Use this guide to present the **Telecom Customer Churn Prediction & Analytics
Dashboard**. It includes a slide outline, a live-demo script, and Q&A notes.

> Always open with the disclaimer: **the dataset is synthetic and the metrics are
> for demonstration only.**

---

## 1. Elevator pitch (30 seconds)

> “ChurnIQ is an end-to-end telecom churn analytics demo. It generates a
> realistic synthetic customer base, trains and compares four machine-learning
> models with leakage-safe pipelines and 5-fold cross-validation, and puts the
> results in a Streamlit dashboard. A business user can explore churn drivers,
> review model performance, score an individual customer, and get transparent
> retention recommendations — with ML predictions and business rules clearly
> separated.”

---

## 2. Slide outline

### Slide 1 — Title

**ChurnIQ: Telecom Customer Churn Prediction & Analytics Dashboard**

- Python · Streamlit · scikit-learn · XGBoost · Plotly · Joblib
- Synthetic-data demonstration
- Your name / course / date

---

### Slide 2 — Problem

- Telecom operators lose revenue when customers leave.
- Retention teams need to know **who is at risk** and **what to do next**.
- Raw customer data is messy and churn is imbalanced.
- A black-box score alone is not enough for business action.

**Goal:** build a complete, reproducible churn workflow from data generation to
an interactive dashboard.

---

### Slide 3 — What we built

Five dashboard sections:

1. **Dashboard Overview** — KPIs and portfolio trends
2. **Customer Analytics** — filters, churn drivers, heatmaps
3. **Model Performance** — CV comparison, test metrics, ROC, confusion matrix
4. **Churn Prediction** — individual customer scoring
5. **Retention Insights** — ML risk + rule-based recommendations

---

### Slide 4 — Synthetic dataset

- **7,000 customers**, generated with a fixed random seed.
- 16 features + `CustomerID`; target `Churn` (Yes/No).
- Demographics: gender, senior citizen, tenure.
- Services: internet, phone, multiple lines, security, tech support.
- Billing: contract type, monthly/total charges, payment method, paperless.
- Behaviour: monthly usage, support calls.
- **Churn rate ≈ 29%**, built from realistic risk relationships.
- Small amounts of missing data are injected to exercise imputation.
- **Every record is synthetic.**

**Talking point:** the generator encodes higher churn for month-to-month
contracts, short tenure, fiber optic, no security/support, electronic checks and
more support calls.

---

### Slide 5 — ML workflow

1. Generate synthetic data.
2. Clean: numeric coercion, de-duplication, valid target only.
3. Stratified **80/20** train/test split.
4. Preprocess inside a scikit-learn `Pipeline`:
   - median imputation + scaling for numeric features
   - most-frequent imputation + one-hot encoding for categorical features
5. Compare four models with **5-fold stratified cross-validation**.
6. Select by **mean validation F1**.
7. Refit the winner on the full training set.
8. Evaluate **once** on the untouched test set.
9. Save the fitted pipeline with Joblib.

**Key line:** “Preprocessing is inside the pipeline, so there is no data
leakage.”

---

### Slide 6 — Models compared

| Model | Why include it? |
|---|---|
| Logistic Regression | Strong, interpretable baseline |
| Decision Tree | Non-linear, easy to explain |
| Random Forest | Robust ensemble, handles interactions |
| XGBoost | Gradient boosting, strong tabular performance |

**Selection rule:** highest mean cross-validation F1-score, not test accuracy.

---

### Slide 7 — Results (synthetic, seed 42)

| Model | CV F1 | CV ROC-AUC |
|---|---|---|
| Logistic Regression | 0.602 | 0.838 |
| XGBoost | 0.595 | 0.825 |
| Random Forest | 0.585 | 0.829 |
| Decision Tree | 0.563 | 0.801 |

**Selected model:** Logistic Regression

**Held-out test metrics:**

- Accuracy: **0.798**
- Precision: **0.685**
- Recall: **0.547**
- F1-score: **0.609**
- ROC-AUC: **0.842**

**Talking point:** the synthetic data was generated from a logistic process, so
a linear model performing well is expected. On real data, the ranking could
change — which is exactly why we compare models.

---

### Slide 8 — Dashboard overview

Show:

- Total customers: **7,000**
- Active: **4,990**
- Churned: **2,010**
- Churn rate: **28.7%**
- High-risk customers: **1,110** (model probability ≥ 60%)

Charts:

- churn distribution
- churn by contract type
- churn by tenure
- churn by payment method
- monthly charges distribution
- priority customer table

**Talking point:** month-to-month customers churn at ~42%, versus ~7% for
two-year contracts. Customers in their first year churn at ~58%.

---

### Slide 9 — Model performance page

Show:

- model comparison table
- cross-validation F1 bar chart with error bars
- confusion matrix
- ROC curve
- feature importance

Explain the confusion matrix in business terms:

- **TN / TP** — correctly identified
- **FP** — customer flagged but stays (unnecessary retention cost)
- **FN** — customer leaves without being flagged (lost revenue)

**Talking point:** precision vs recall is a business trade-off. Lowering the
threshold catches more churners but creates more false alarms.

---

### Slide 10 — Churn prediction demo

Live demo steps:

1. Open **Churn Prediction**.
2. Keep or change the profile:
   - Month-to-month contract
   - Fiber optic internet
   - Electronic check
   - No online security / tech support
   - 12-month tenure
3. Click **Predict churn risk**.
4. Point out:
   - churn probability gauge
   - Yes/No status
   - High/Medium/Low risk band
   - rule-based next steps

**Talking point:** the score is an estimate, not a fact. The 50% threshold is a
business choice.

---

### Slide 11 — Retention insights

Show a high-risk customer and explain the two separate signals:

**ML assessment**

- probability learned from historical patterns
- ranks risk
- does not explain cause

**Rule-based recommendations**

- month-to-month → offer a better contract plan
- 4+ support calls → provide customer support
- tenure ≤ 12 months → strengthen onboarding
- monthly charges ≥ $85 → review pricing
- no security/support → bundle services
- electronic check → incentivise autopay
- tenure ≥ 48 months → loyalty discount

**Talking point:** use ML to prioritise, and rules to decide the next action.

---

### Slide 12 — Limitations & next steps

**Limitations**

- synthetic data; not real business performance
- no production calibration, drift monitoring or fairness audit
- rules are illustrative heuristics, not proven interventions
- no causal uplift modelling

**Next steps**

- connect a real, governed customer dataset
- calibrate probabilities and tune the decision threshold
- add SHAP explanations
- run A/B tests on retention offers
- monitor drift and retrain on a schedule
- add role-based access and privacy controls

---

### Slide 13 — Key takeaways

- Complete workflow: data → pipeline → CV → evaluation → dashboard.
- Leakage-safe preprocessing and a single untouched test evaluation.
- Business-friendly UI with KPIs, charts and individual scoring.
- Clear separation between **ML prediction** and **rule-based action**.
- Reproducible: fixed seeds, saved artifacts, one-command run.

**Close:** “ChurnIQ shows how a model becomes useful only when it is connected
to a clear retention decision.”

---

## 3. Live demo script (5–7 minutes)

1. **Start** — “This is ChurnIQ, built on synthetic telecom data. The sidebar
   shows the best model and lets us retrain.”
2. **Overview** — point to the five KPI cards, then the churn-by-contract and
   tenure charts. Say the headline: month-to-month and early-tenure customers
   are the riskiest.
3. **Analytics** — change a filter (e.g. remove two-year contracts) and show the
   KPIs and charts update.
4. **Model Performance** — show the comparison table and CV F1 chart. Explain
   why the test set is untouched. Open the confusion matrix and ROC curve.
5. **Prediction** — submit a high-risk profile. Point out the gauge, risk band
   and rule-based next steps.
6. **Retention** — select a top high-risk customer. Show the ML panel and the
   rule-based recommendations side by side. Show the downloadable high-risk
   list.
7. **Close** — repeat the synthetic-data disclaimer and the limitations.

---

## 4. Likely questions and answers

**Q: Is this real data?**
No. Every record is synthetic and generated for demonstration. The metrics do
not represent real telecom performance.

**Q: Why did Logistic Regression win?**
The synthetic churn process is logistic, so a linear model fits it well. The
point of the project is the comparison workflow, not a claim that one algorithm
is universally best.

**Q: How do you prevent data leakage?**
All imputation, scaling and encoding are inside the scikit-learn pipeline.
Cross-validation refits the pipeline on each training fold, and the test set is
used only once after model selection.

**Q: Why use F1 instead of accuracy?**
Churn is the minority class. Accuracy can look good while missing churners. F1
balances precision and recall.

**Q: What is the difference between the ML score and the recommendations?**
The ML score is a learned probability. The recommendations are deterministic
business rules. The dashboard labels them separately.

**Q: How would you productionise this?**
Connect governed real data, calibrate probabilities, monitor drift, add
explainability, run retention-offer experiments, and add access controls.

**Q: What would you improve next?**
Real data, SHAP explanations, threshold optimisation based on retention cost,
uplift modelling, and automated monitoring.

---

## 5. One-line disclaimer to repeat

> “This is a synthetic-data demonstration. The model metrics and predictions are
> illustrative and must not be presented as real telecom business performance.”
