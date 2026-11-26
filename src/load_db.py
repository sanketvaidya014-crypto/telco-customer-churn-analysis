"""Clean the Telco Customer Churn CSV and load it into SQLite (data/telco.db, table `customers`)."""
import sqlite3

import pandas as pd

from features import DB, RAW

CANDIDATES = ["WA_Fn-UseC_-Telco-Customer-Churn.csv", "Telco-Customer-Churn.csv"]


def main() -> None:
    src = next((RAW / f for f in CANDIDATES if (RAW / f).exists()), None)
    if src is None:
        raise SystemExit(
            f"No dataset found in {RAW}.\nDownload 'Telco Customer Churn' from Kaggle (blastchar/telco-customer-churn) "
            f"and save it as {CANDIDATES[0]}, or run `python src/make_sample_data.py` for a synthetic demo file."
        )
    df = pd.read_csv(src)
    n0 = len(df)
    df.columns = df.columns.str.strip()
    for c in df.columns[df.dtypes.map(lambda t: t == object or str(t) in ("str", "string"))]:
        df[c] = df[c].str.strip()

    # TotalCharges contains blanks for customers with tenure 0 (never billed yet) -> numeric, 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blanks = df.TotalCharges.isna().sum()
    df["TotalCharges"] = df.TotalCharges.fillna(0.0)
    df = df.drop_duplicates("customerID")
    df["churn_flag"] = (df.Churn == "Yes").astype(int)

    if DB.exists():
        DB.unlink()
    with sqlite3.connect(DB) as con:
        df.to_sql("customers", con, index=False)
        con.execute("CREATE INDEX ix_cust_contract ON customers(Contract)")
    print(f"Source: {src.name}\nRows loaded: {len(df):,} (dropped {n0 - len(df)} duplicates); "
          f"blank TotalCharges set to 0: {blanks}\nChurn rate: {df.churn_flag.mean():.2%}\nDatabase ready: {DB}")


if __name__ == "__main__":
    main()
