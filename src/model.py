"""Train, evaluate and explain churn models; produce the churn-risk customer list and recommendations.
Usage: python src/model.py   (after python src/load_db.py)"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, average_precision_score,
                             brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import DROP, FIG, MOD, OUT, TAB, TARGET, engineer, load_customers

SEED = 42
sns.set_theme(style="whitegrid", context="notebook")
BLUE, RED = "#2F5D8A", "#C0504D"


def make_pipeline(model, X):
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat)])
    return Pipeline([("pre", pre), ("model", model)])


def best_threshold(y, p):
    grid = np.arange(.05, .95, .01)
    return float(grid[np.argmax([f1_score(y, p >= t) for t in grid])])


def action(r):
    """Rule-based retention play, derived from the churn drivers found in the SQL/EDA. Hypotheses to A/B test."""
    if r.Contract == "Month-to-month" and r.tenure <= 12:
        return "Discounted 1-year contract offer"
    if r.InternetService == "Fiber optic" and r.TechSupport == "No":
        return "Free tech-support trial"
    if r.PaymentMethod == "Electronic check":
        return "Autopay switch incentive"
    if r.Contract == "Month-to-month":
        return "Loyalty reward for longer contract"
    return "Proactive account-manager call"


def rate(df, mask):
    return 100 * df.loc[mask, TARGET].mean()


def main():
    raw = load_customers()
    df = engineer(raw)
    X, y = df.drop(columns=DROP), df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    models = {
        "Baseline (prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
    }
    fitted, rows = {}, []
    for name, m in models.items():
        pipe = make_pipeline(m, X)
        cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
        oof = cross_val_predict(pipe, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        thr = best_threshold(ytr, oof) if name != "Baseline (prior)" else .5
        pipe.fit(Xtr, ytr); p = pipe.predict_proba(Xte)[:, 1]; yhat = p >= thr
        fitted[name] = (pipe, thr, p)
        rows.append({"model": name, "cv_roc_auc": cv_auc, "test_roc_auc": roc_auc_score(yte, p),
                     "test_pr_auc": average_precision_score(yte, p), "threshold": thr,
                     "precision": precision_score(yte, yhat, zero_division=0), "recall": recall_score(yte, yhat),
                     "f1": f1_score(yte, yhat), "brier": brier_score_loss(yte, p)})
    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(TAB / "model_metrics.csv", index=False)
    print(metrics.to_string(index=False))

    cand = metrics[metrics.model != "Baseline (prior)"]
    best_name = cand.loc[cand.cv_roc_auc.idxmax(), "model"]   # selected on CV (train only), not on the test set
    best, thr, pbest = fitted[best_name]
    print(f"\nSelected model: {best_name} (threshold {thr:.2f})")

    # ---- curves + confusion matrix
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for n, (pipe, _, p) in fitted.items():
        if n.startswith("Baseline"): continue
        RocCurveDisplay.from_predictions(yte, p, name=n, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(yte, p, name=n, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", lw=.8); axes[0].set_title("ROC curve (test set)")
    axes[1].axhline(yte.mean(), color="k", ls="--", lw=.8); axes[1].set_title("Precision-recall curve (test set)")
    ConfusionMatrixDisplay.from_predictions(yte, pbest >= thr, display_labels=["Stay", "Churn"], cmap="Blues", ax=axes[2], colorbar=False)
    axes[2].set_title(f"Confusion matrix - {best_name}\n(threshold {thr:.2f})"); axes[2].grid(False)
    fig.tight_layout(); fig.savefig(FIG / "model_evaluation.png", dpi=150); plt.close(fig)

    # ---- feature importance (permutation, on held-out data, in terms of the original columns)
    pi = permutation_importance(best, Xte, yte, scoring="roc_auc", n_repeats=15, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": Xte.columns, "importance": pi.importances_mean, "std": pi.importances_std})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(TAB / "feature_importance.csv", index=False)
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.feature, top.importance, xerr=top["std"], color=BLUE)
    ax.set(title=f"Feature importance - {best_name}", xlabel="Drop in ROC-AUC when feature is shuffled (permutation importance)")
    fig.tight_layout(); fig.savefig(FIG / "feature_importance.png", dpi=150); plt.close(fig)

    # ---- interpretable view: logistic regression odds ratios
    lr = fitted["Logistic Regression"][0]
    names = lr.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(lr.named_steps["model"].coef_[0], index=[n.split("__", 1)[1] for n in names])
    odds = np.exp(coefs).rename("odds_ratio").sort_values()
    odds.to_csv(TAB / "logistic_odds_ratios.csv")
    sel = pd.concat([odds.head(8), odds.tail(8)])
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(sel.index, sel.values - 1, left=1, color=[BLUE if v < 1 else RED for v in sel.values])
    ax.axvline(1, color="k", lw=.8)
    ax.set(title="Logistic regression odds ratios (>1 raises churn risk)", xlabel="Odds ratio per 1 SD (numeric) / vs baseline category")
    fig.tight_layout(); fig.savefig(FIG / "odds_ratios.png", dpi=150); plt.close(fig)

    # ---- churn-risk list: out-of-fold scores so no customer is scored by a model that saw their own label
    oof_all = cross_val_predict(make_pipeline(models[best_name].__class__(**models[best_name].get_params()), X), X, y, cv=cv, method="predict_proba")[:, 1]
    scored = raw.assign(churn_probability=oof_all)
    scored = engineer(scored)
    act = scored[scored.Churn == "No"].copy()
    act["risk_rank_pct"] = act.churn_probability.rank(pct=True, ascending=False)
    act["risk_tier"] = np.select([act.risk_rank_pct <= .15, act.risk_rank_pct <= .40], ["High", "Medium"], "Low")
    act["expected_monthly_revenue_at_risk"] = (act.churn_probability * act.MonthlyCharges).round(2)
    act["recommended_action"] = act.apply(action, axis=1)
    cols = ["customerID", "churn_probability", "risk_tier", "MonthlyCharges", "expected_monthly_revenue_at_risk", "tenure",
            "Contract", "InternetService", "PaymentMethod", "TechSupport", "recommended_action"]
    risk = act.sort_values("churn_probability", ascending=False)[cols]
    risk["churn_probability"] = risk.churn_probability.round(4)
    risk.to_csv(TAB / "churn_risk_customers.csv", index=False)

    # final model on ALL data for deployment/scoring new customers
    final = make_pipeline(models[best_name].__class__(**models[best_name].get_params()), X).fit(X, y)
    joblib.dump({"pipeline": final, "threshold": thr, "features": X.columns.tolist()}, MOD / "churn_model.joblib")

    # ---- recommendations (numbers computed from the data)
    inet = df[df.InternetService != "No"]
    m2m, two = rate(df, df.Contract == "Month-to-month"), rate(df, df.Contract == "Two year")
    new, old = rate(df, df.tenure <= 12), rate(df, df.tenure > 24)
    echeck, auto = rate(df, df.PaymentMethod == "Electronic check"), rate(df, df.autopay == 1)
    fiber, dsl = rate(df, df.InternetService == "Fiber optic"), rate(df, df.InternetService == "DSL")
    no_ts, ts = rate(inet, inet.TechSupport == "No"), rate(inet, inet.TechSupport == "Yes")
    hi = risk[risk.risk_tier == "High"]
    topf = ", ".join(imp.head(5).feature)
    L = ["# Retention recommendations (auto-generated from the data currently loaded)\n",
         f"**Model:** {best_name}, test ROC-AUC {metrics.set_index('model').loc[best_name, 'test_roc_auc']:.3f}, "
         f"recall {metrics.set_index('model').loc[best_name, 'recall']:.2f} / precision {metrics.set_index('model').loc[best_name, 'precision']:.2f} "
         f"at threshold {thr:.2f}. Top drivers (permutation importance): {topf}.\n",
         "## What drives churn\n",
         f"- **Contract:** month-to-month customers churn at {m2m:.1f}% vs {two:.1f}% on two-year contracts.",
         f"- **Lifecycle:** customers in their first 12 months churn at {new:.1f}% vs {old:.1f}% after 24 months.",
         f"- **Payment:** electronic-check payers churn at {echeck:.1f}% vs {auto:.1f}% on automatic payment.",
         f"- **Internet service:** fiber customers churn at {fiber:.1f}% vs {dsl:.1f}% for DSL.",
         f"- **Support:** internet customers without tech support churn at {no_ts:.1f}% vs {ts:.1f}% with it.\n",
         "## Recommended actions (test each with an A/B experiment; these are associations, not proven causes)\n",
         "1. **Convert new month-to-month customers to 1-year contracts** with a first-year discount or bill credit.",
         "2. **Push autopay** with a small recurring discount to move customers off electronic check.",
         "3. **Bundle tech support / online security** into fiber plans (free trial for 2-3 months) and track churn of the treated group.",
         "4. **Run an early-life onboarding programme** (first-90-days check-ins), since early tenure is the riskiest window.",
         "5. **Use the churn-risk list** to prioritise outreach: contact the High tier first, with the `recommended_action` column.\n",
         "## Churn-risk list\n",
         f"- Active customers scored: {len(risk):,}. High-risk tier (top 15%): {len(hi):,} customers, expected monthly revenue at risk "
         f"${hi.expected_monthly_revenue_at_risk.sum():,.0f} (probability x monthly charge).",
         "- Scores are out-of-fold (each customer is scored by a model that did not train on them).",
         "\n## Caveats\n",
         "- The dataset is a single snapshot, so the model estimates churn *propensity*, not a dated 'next month' prediction.",
         "- Feature importance shows what the model uses, not what causes churn. Validate with experiments."]
    (OUT / "recommendations.md").write_text("\n".join(L), encoding="utf-8")
    (TAB / "run_summary.json").write_text(json.dumps({"best_model": best_name, "threshold": thr, "n_train": len(Xtr), "n_test": len(Xte)}, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
