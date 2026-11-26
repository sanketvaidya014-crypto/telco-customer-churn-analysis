-- Service usage: churn for internet customers with vs without each add-on
WITH inet AS (SELECT * FROM customers WHERE InternetService <> 'No'),
svc AS (
    SELECT 'OnlineSecurity' AS service, OnlineSecurity AS has_service, churn_flag FROM inet
    UNION ALL SELECT 'OnlineBackup',     OnlineBackup,     churn_flag FROM inet
    UNION ALL SELECT 'DeviceProtection', DeviceProtection, churn_flag FROM inet
    UNION ALL SELECT 'TechSupport',      TechSupport,      churn_flag FROM inet
    UNION ALL SELECT 'StreamingTV',      StreamingTV,      churn_flag FROM inet
    UNION ALL SELECT 'StreamingMovies',  StreamingMovies,  churn_flag FROM inet
)
SELECT service, has_service,
       COUNT(*)                          AS customers,
       ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
FROM svc
GROUP BY service, has_service
ORDER BY service, has_service DESC;
