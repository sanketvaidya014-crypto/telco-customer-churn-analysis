-- Business segments = lifecycle stage x value tier (value = above/below median monthly charge)
WITH med AS (
    SELECT MonthlyCharges AS median_charge
    FROM customers
    ORDER BY MonthlyCharges
    LIMIT 1 OFFSET (SELECT COUNT(*) / 2 FROM customers)
),
seg AS (
    SELECT c.*,
           CASE WHEN c.Contract = 'Month-to-month' AND c.tenure <= 12 THEN '1) New, no commitment'
                WHEN c.Contract = 'Month-to-month'                    THEN '2) Established, no commitment'
                WHEN c.tenure <= 24                                   THEN '3) New, contracted'
                ELSE                                                       '4) Loyal, contracted' END AS lifecycle,
           CASE WHEN c.MonthlyCharges >= m.median_charge THEN 'High value' ELSE 'Low value' END AS value_tier
    FROM customers c CROSS JOIN med m
)
SELECT lifecycle, value_tier,
       COUNT(*)                                   AS customers,
       ROUND(100.0 * AVG(churn_flag), 2)          AS churn_rate_pct,
       ROUND(AVG(MonthlyCharges), 2)              AS avg_monthly_charges,
       ROUND(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges END), 2) AS monthly_revenue_lost,
       ROUND(100.0 * SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges END)
             / SUM(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges END)) OVER (), 2) AS share_of_revenue_lost_pct
FROM seg
GROUP BY lifecycle, value_tier
ORDER BY churn_rate_pct DESC;
