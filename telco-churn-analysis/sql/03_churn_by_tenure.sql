-- Churn by tenure band (customer lifecycle stage)
SELECT CASE WHEN tenure <= 12 THEN '1) 0-12 months'
            WHEN tenure <= 24 THEN '2) 13-24 months'
            WHEN tenure <= 48 THEN '3) 25-48 months'
            ELSE                   '4) 49+ months' END     AS tenure_band,
       COUNT(*)                                            AS customers,
       SUM(churn_flag)                                     AS churned,
       ROUND(100.0 * AVG(churn_flag), 2)                   AS churn_rate_pct,
       ROUND(AVG(MonthlyCharges), 2)                       AS avg_monthly_charges
FROM customers
GROUP BY 1
ORDER BY 1;
