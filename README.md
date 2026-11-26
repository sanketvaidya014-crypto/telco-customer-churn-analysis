# Customer Churn Prediction & Analysis (Telco)

**Business question:** *Which customers are likely to cancel, and what behaviour do they share?*

An end-to-end project: **SQL** for churn-rate analysis and customer segmentation, **Python** for EDA, correlation analysis and a **scikit-learn** model, producing a **churn-risk customer list**, a **feature-importance chart** and **retention recommendations**.

## Dataset
[Telco Customer Churn (Kaggle, IBM sample data)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers, 21 columns (demographics, services, contract, billing, `Churn`).
Save it as `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Project structure
```
data/raw/             source CSV (git-ignored)
sql/                  8 queries (SQLite): churn by contract/payment/tenure/charges/services, segments, hotspots
src/features.py       shared paths + feature engineering
src/load_db.py        cleans the CSV and loads SQLite (data/telco.db)
src/eda.py            runs the SQL, EDA charts, correlation / association analysis
src/model.py          trains + evaluates models, risk list, importance, recommendations
src/make_sample_data.py   synthetic data for testing ONLY
outputs/              figures/, tables/ (incl. churn_risk_customers.csv), recommendations.md
```

## How to run
```bash
pip install -r requirements.txt
python src/load_db.py
python src/eda.py
python src/model.py
```
No dataset yet? `python src/make_sample_data.py` makes a synthetic file to test the pipeline. **Never report numbers from it as findings.**

## SQL analysis
| File | Question | Techniques |
|---|---|---|
| 01_overall_churn | Churn rate and monthly revenue lost | aggregation |
| 02_churn_by_dimension | Churn by contract, payment, internet, billing, seniority | `UNION ALL`, CTE, gap vs overall |
| 03_churn_by_tenure | Churn by lifecycle stage | `CASE` bands |
| 04_churn_by_monthly_charges | Do higher bills churn more? | `NTILE(4)` |
| 05_churn_by_service | Add-on services vs churn | `UNION ALL` pivoting |
| 06_churn_by_addon_count | Do protective bundles reduce churn? | derived counts |
| 07_customer_segments | Lifecycle x value segments, share of revenue lost | `CASE`, median via `OFFSET`, window `SUM() OVER ()` |
| 08_churn_hotspots | Riskiest contract x service x tenure groups | `RANK() OVER`, `HAVING` |

## Python analysis
- **Cleaning:** blank `TotalCharges` (tenure 0, never billed) set to 0, whitespace stripped, duplicates checked, `churn_flag` created.
- **Features:** `num_addons`, `num_protective`, `autopay`, `avg_monthly_spend`, `charge_vs_history`. Only pre-churn information is used (no target leakage).
- **Correlation analysis:** Pearson r for numeric features and Cramer's V for categoricals, vs churn; correlation heatmap.
- **Modelling:** stratified 80/20 split, 5-fold CV, Logistic Regression vs Random Forest vs a baseline; model chosen on **CV score from the training set only**; decision threshold tuned for F1 on out-of-fold predictions (not on the test set).
- **Metrics:** ROC-AUC, PR-AUC, precision, recall, F1, Brier score (calibration), confusion matrix.
- **Explainability:** permutation importance on held-out data (original column names) and logistic-regression odds ratios.

## Outputs
| Deliverable | File |
|---|---|
| Churn-risk customer list (active customers, tiers, revenue at risk, suggested action) | `outputs/tables/churn_risk_customers.csv` |
| Feature-importance chart | `outputs/figures/feature_importance.png` |
| EDA dashboard | `outputs/figures/eda_dashboard.png` |
| Model evaluation (ROC, PR, confusion matrix) | `outputs/figures/model_evaluation.png` |
| Retention recommendations (numbers auto-generated) | `outputs/recommendations.md` |

**Risk list method:** each customer is scored out-of-fold (by a model that did not train on them). Tiers: High = top 15% by risk, Medium = next 25%. `expected_monthly_revenue_at_risk` = churn probability x monthly charge. `recommended_action` is a rule-based playbook derived from the churn drivers.

## Results
Run the pipeline, then copy the key numbers from `outputs/recommendations.md` and `outputs/tables/model_metrics.csv` here (model, ROC-AUC, recall/precision, top drivers), and add:
```markdown
![EDA dashboard](outputs/figures/eda_dashboard.png)
![Feature importance](outputs/figures/feature_importance.png)
```

## Limitations
- The dataset is a **snapshot**, so the model predicts churn *propensity*, not a dated "next month" event. A production version needs monthly snapshots and a defined prediction window.
- Feature importance and correlations show **association, not causation**. Retention offers must be validated with A/B tests.
- Probabilities are only approximately calibrated; check the Brier score before using revenue-at-risk figures for budgeting.
- Single company, limited features (no usage, support-ticket or competitor data).

## Interview talking points
- "I used SQL segments (lifecycle x value) to show where revenue loss concentrates, then built a model to rank individual customers."
- "I picked the model on cross-validation, tuned the threshold on out-of-fold predictions, and kept the test set untouched until the end."
- "I scored the risk list out-of-fold so no customer was scored by a model trained on their own label."
- "I treated recommendations as hypotheses to A/B test, because importance is not causation."
