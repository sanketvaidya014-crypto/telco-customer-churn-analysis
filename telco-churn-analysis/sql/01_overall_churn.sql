-- Headline KPIs: churn rate and the recurring revenue lost to churn
SELECT COUNT(*)                                             AS customers,
       SUM(churn_flag)                                      AS churned,
       ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2)         AS churn_rate_pct,
       ROUND(SUM(MonthlyCharges), 2)                        AS monthly_revenue_total,
       ROUND(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges END), 2) AS monthly_revenue_lost,
       ROUND(AVG(CASE WHEN churn_flag = 1 THEN MonthlyCharges END), 2) AS avg_charge_churned,
       ROUND(AVG(CASE WHEN churn_flag = 0 THEN MonthlyCharges END), 2) AS avg_charge_retained
FROM customers;
