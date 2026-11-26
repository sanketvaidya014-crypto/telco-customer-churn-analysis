-- Does bundling protective add-ons (security, backup, protection, tech support) reduce churn?
WITH inet AS (
    SELECT churn_flag,
           (OnlineSecurity = 'Yes') + (OnlineBackup = 'Yes')
         + (DeviceProtection = 'Yes') + (TechSupport = 'Yes') AS protective_addons
    FROM customers
    WHERE InternetService <> 'No'
)
SELECT protective_addons,
       COUNT(*)                          AS customers,
       ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
FROM inet
GROUP BY protective_addons
ORDER BY protective_addons;
