-- Churn rate by contract type, payment method, internet service, billing and seniority
-- (long format: one row per dimension/category; `vs_overall_pp` = gap vs overall churn in percentage points)
WITH dims AS (
    SELECT 'Contract'        AS dimension, Contract        AS category, churn_flag, MonthlyCharges FROM customers
    UNION ALL SELECT 'PaymentMethod',    PaymentMethod,    churn_flag, MonthlyCharges FROM customers
    UNION ALL SELECT 'InternetService',  InternetService,  churn_flag, MonthlyCharges FROM customers
    UNION ALL SELECT 'PaperlessBilling', PaperlessBilling, churn_flag, MonthlyCharges FROM customers
    UNION ALL SELECT 'SeniorCitizen', CASE WHEN SeniorCitizen = 1 THEN 'Yes' ELSE 'No' END, churn_flag, MonthlyCharges FROM customers
),
overall AS (SELECT 100.0 * AVG(churn_flag) AS rate FROM customers)
SELECT d.dimension, d.category,
       COUNT(*)                                        AS customers,
       SUM(d.churn_flag)                               AS churned,
       ROUND(100.0 * AVG(d.churn_flag), 2)             AS churn_rate_pct,
       ROUND(100.0 * AVG(d.churn_flag) - o.rate, 2)    AS vs_overall_pp,
       ROUND(AVG(d.MonthlyCharges), 2)                 AS avg_monthly_charges
FROM dims d CROSS JOIN overall o
GROUP BY d.dimension, d.category
ORDER BY d.dimension, churn_rate_pct DESC;
