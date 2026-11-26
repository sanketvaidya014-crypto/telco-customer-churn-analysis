"""Run the SQL analysis, EDA charts and association (correlation) analysis.
Usage: python src/eda.py   (after python src/load_db.py)"""
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from features import DB, DROP, FIG, SQL_DIR, TAB, TARGET, engineer, load_customers

sns.set_theme(style="whitegrid", context="notebook")
BLUE, RED = "#2F5D8A", "#C0504D"


def run_sql():
    out = {}
    with sqlite3.connect(DB) as con:
        for f in sorted(SQL_DIR.glob("*.sql")):
            out[f.stem] = pd.read_sql_query(f.read_text(), con)
            out[f.stem].to_csv(TAB / f"{f.stem}.csv", index=False)
    return out


def cramers_v(x, y):
    t = pd.crosstab(x, y)
    chi2 = chi2_contingency(t, correction=False)[0]
    return np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))


def associations(df):
    """Correlation of every feature with churn: point-biserial (numeric) / Cramer's V (categorical)."""
    rows = []
    for c in df.columns.difference(DROP):
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 6:
            rows.append((c, "numeric (Pearson r)", abs(df[c].corr(df[TARGET])), df[c].corr(df[TARGET])))
        else:
            v = cramers_v(df[c], df[TARGET])
            rows.append((c, "categorical (Cramer's V)", v, np.nan))
    return pd.DataFrame(rows, columns=["feature", "method", "strength", "signed_r"]).sort_values("strength", ascending=False)


def bar_rate(ax, data, x, title, horizontal=False, overall=None):
    data = data.sort_values("churn_rate_pct", ascending=horizontal)
    if horizontal:
        bars = ax.barh(data[x], data.churn_rate_pct, color=BLUE)
        for b, v in zip(bars, data.churn_rate_pct):
            ax.text(v + .5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
        ax.set(xlabel="Churn rate (%)")
        if overall: ax.axvline(overall, color=RED, ls="--", lw=1)
    else:
        bars = ax.bar(data[x], data.churn_rate_pct, color=BLUE)
        for b, v in zip(bars, data.churn_rate_pct):
            ax.text(b.get_x() + b.get_width() / 2, v + .5, f"{v:.1f}%", ha="center", fontsize=9)
        ax.set(ylabel="Churn rate (%)")
        if overall: ax.axhline(overall, color=RED, ls="--", lw=1, label=f"overall {overall:.1f}%"); ax.legend(fontsize=8)
    ax.set_title(title)


def main():
    sq = run_sql()
    df = engineer(load_customers())
    overall = float(sq["01_overall_churn"].churn_rate_pct.iloc[0])
    dim = sq["02_churn_by_dimension"]
    assoc = associations(df)
    assoc.to_csv(TAB / "association_with_churn.csv", index=False)

    def contract(ax): bar_rate(ax, dim[dim.dimension == "Contract"], "category", "Churn by contract type", overall=overall)
    def tenure(ax):
        t = sq["03_churn_by_tenure"].copy(); t["band"] = t.tenure_band.str[3:]
        bar_rate(ax, t.sort_values("tenure_band"), "band", "Churn by tenure band", overall=overall)
        ax.set_xticks(range(len(t))); ax.set_xticklabels(t.band)
    def payment(ax): bar_rate(ax, dim[dim.dimension == "PaymentMethod"], "category", "Churn by payment method", horizontal=True, overall=overall)
    def internet(ax): bar_rate(ax, dim[dim.dimension == "InternetService"], "category", "Churn by internet service", overall=overall)
    def charges(ax):
        sns.histplot(data=df, x="MonthlyCharges", hue="Churn", bins=30, stat="density", common_norm=False,
                     palette={"No": BLUE, "Yes": RED}, element="step", ax=ax)
        ax.set_title("Monthly charges: churned vs retained")
    def assoc_chart(ax):
        a = assoc.head(10).iloc[::-1]
        ax.barh(a.feature, a.strength, color=BLUE)
        ax.set(title="Association with churn (|r| or Cramer's V)", xlabel="Strength")

    charts = [("churn_by_contract", contract), ("churn_by_tenure", tenure), ("churn_by_payment_method", payment),
              ("churn_by_internet_service", internet), ("monthly_charges_distribution", charges),
              ("association_with_churn", assoc_chart)]
    for name, fn in charts:
        fig, ax = plt.subplots(figsize=(7.5, 4.5)); fn(ax); fig.tight_layout()
        fig.savefig(FIG / f"{name}.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(19, 9.5))
    for ax, (_, fn) in zip(axes.ravel(), charts): fn(ax)
    fig.suptitle(f"Customer Churn - EDA Dashboard (overall churn {overall:.1f}%)", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(FIG / "eda_dashboard.png", dpi=130); plt.close(fig)

    num = df[["tenure", "MonthlyCharges", "TotalCharges", "num_addons", "autopay", "SeniorCitizen", TARGET]]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation matrix (numeric features)"); fig.tight_layout()
    fig.savefig(FIG / "correlation_matrix.png", dpi=150); plt.close(fig)
    print(sq["01_overall_churn"].T.to_string(header=False))
    print("\nTop associations with churn:\n", assoc.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
