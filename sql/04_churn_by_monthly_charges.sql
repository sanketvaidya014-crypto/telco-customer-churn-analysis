-- Churn by monthly-charge quartile (NTILE); Q4 = highest bills
WITH q AS (
    SELECT *, NTILE(4) OVER (ORDER BY MonthlyCharges) AS charge_quartile FROM customers
)
SELECT charge_quartile,
       ROUND(MIN(MonthlyCharges), 2)           AS min_charge,
       ROUND(MAX(MonthlyCharges), 2)           AS max_charge,
       COUNT(*)                                AS customers,
       ROUND(100.0 * AVG(churn_flag), 2)       AS churn_rate_pct
FROM q
GROUP BY charge_quartile
ORDER BY charge_quartile;
