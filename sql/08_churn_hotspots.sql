-- Hotspots: contract x internet service x tenure band, ranked by churn rate (min 50 customers)
WITH grp AS (
    SELECT Contract, InternetService,
           CASE WHEN tenure <= 12 THEN '0-12m' WHEN tenure <= 36 THEN '13-36m' ELSE '37m+' END AS tenure_band,
           COUNT(*)                          AS customers,
           ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct,
           ROUND(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges END), 2) AS monthly_revenue_lost
    FROM customers
    GROUP BY 1, 2, 3
    HAVING COUNT(*) >= 50
)
SELECT RANK() OVER (ORDER BY churn_rate_pct DESC) AS rnk, *
FROM grp
ORDER BY rnk
LIMIT 10;
