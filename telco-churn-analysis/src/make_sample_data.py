"""Generate a SYNTHETIC file with the Telco schema to test the pipeline end to end.
Numbers produced from it are NOT real findings. Use the real Kaggle dataset for the analysis."""
import numpy as np
import pandas as pd

from features import RAW

rng = np.random.default_rng(7)
n = 7043
pick = lambda opts, p: rng.choice(opts, n, p=p)

contract = pick(["Month-to-month", "One year", "Two year"], [.55, .21, .24])
tenure = np.where(contract == "Month-to-month", np.clip(rng.exponential(18, n), 0, 72), rng.uniform(12, 72, n)).round().astype(int)
internet = pick(["DSL", "Fiber optic", "No"], [.34, .44, .22])
phone = pick(["Yes", "No"], [.9, .1])
pay = pick(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], [.34, .23, .22, .21])
senior = (rng.random(n) < .16).astype(int)

df = pd.DataFrame({
    "customerID": [f"{i:04d}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for i in range(n)],
    "gender": pick(["Male", "Female"], [.5, .5]), "SeniorCitizen": senior,
    "Partner": pick(["Yes", "No"], [.48, .52]), "Dependents": pick(["Yes", "No"], [.3, .7]),
    "tenure": tenure, "PhoneService": phone,
})
df["MultipleLines"] = np.where(phone == "No", "No phone service", pick(["Yes", "No"], [.42, .58]))
df["InternetService"] = internet
for col, p in [("OnlineSecurity", .3), ("OnlineBackup", .35), ("DeviceProtection", .35), ("TechSupport", .3),
               ("StreamingTV", .45), ("StreamingMovies", .45)]:
    df[col] = np.where(internet == "No", "No internet service", np.where(rng.random(n) < p, "Yes", "No"))
df["Contract"] = contract
df["PaperlessBilling"] = pick(["Yes", "No"], [.6, .4])
df["PaymentMethod"] = pay
addons = (df[["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]] == "Yes").sum(axis=1)
df["MonthlyCharges"] = (np.where(phone == "Yes", 20, 0) + np.select([internet == "DSL", internet == "Fiber optic"], [25, 55], 0)
                        + 5 * addons + rng.normal(0, 4, n)).clip(18.25, 118.75).round(2)
df["TotalCharges"] = (df.MonthlyCharges * df.tenure * rng.uniform(.95, 1.05, n)).round(2).astype(str)
df.loc[df.tenure == 0, "TotalCharges"] = " "

logit = (-1.0 + 1.5 * (contract == "Month-to-month") - 1.2 * (contract == "Two year") - 0.035 * tenure
         + 0.8 * (internet == "Fiber optic") + 0.5 * (pay == "Electronic check") - 0.7 * (df.TechSupport == "Yes")
         - 0.5 * (df.OnlineSecurity == "Yes") + 0.012 * (df.MonthlyCharges - 65) + 0.25 * senior + rng.normal(0, .5, n))
df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

RAW.mkdir(parents=True, exist_ok=True)
df.to_csv(RAW / "WA_Fn-UseC_-Telco-Customer-Churn.csv", index=False)
print(f"Synthetic sample written ({n:,} rows, churn {np.mean(df.Churn == 'Yes'):.1%}). NOT real data.")
