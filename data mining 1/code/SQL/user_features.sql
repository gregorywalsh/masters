#USER FEATURES - PU
CREATE TABLE instacart_intermediate.U_features_1_PU1 AS
	SELECT
		user_id,
		COUNT(DISTINCT opa.order_ID) orders_TOT_COUNT,
		COUNT(DISTINCT opa.product_id) AS prods_ordered_DIST_CNT,
		COUNT(*) AS prods_ordered_TOT_CNT,
		COUNT(DISTINCT p.department_id) AS departments_DIST_CNT,
		COUNT(DISTINCT p.aisle_id) AS aisle_DIST_CNT
		
	FROM
		order_products__all AS opa
		
		INNER JOIN products AS p ON
			p.product_id = opa.product_id
		
	WHERE
		opa.source = 'prior'
        AND opa.user_id < 100
		
	GROUP BY
		user_id
	
    ORDER BY
		user_id ASC
;

#USER FEATURES DISTRIBUTION STATISTICS - PU
CREATE TABLE instacart_intermediate.U_features_1_PU2 AS
	SELECT
		user_id,
		AVG(LOG(prods_per_order)) AS prods_per_order_LOG_AVG,
		STDDEV(LOG(prods_per_order)) AS prods_per_order_LOG_STD
		
	FROM
		(SELECT
			user_id,
			order_id,
			COUNT(*) AS prods_per_order
			
		FROM
			order_products__all AS opa
			
		WHERE
			source = 'prior'
			
		GROUP BY
			user_id,
			order_id
		
        ORDER BY
			user_id
            
		) log_stats
		
	GROUP BY
		user_id
	
    ORDER BY
		user_id ASC
;

#USER FEATURES - OU
CREATE TABLE instacart_intermediate.U_features_1_OU1 AS
	SELECT
		user_id,
		AVG(LOG(days_since_prior_order + 1)) AS days_between_orders_LOG_AVG,
		STDDEV(LOG(days_since_prior_order + 1)) AS days_between_orders_LOG_STD
		
	FROM
		orders AS o
		
	WHERE
		eval_set = 'prior'
		
	GROUP BY
		user_id
	
    ORDER BY
		user_id ASC
;

#USER FEATURES - OU
CREATE TABLE instacart_intermediate.U_features_1_OU2 AS
	SELECT
		user_id,
		AVG(prev_ord_as_pred_F1) AS prev_ord_as_pred_F1_AVG,
		AVG(prev_ord_as_pred_PRECISION) AS prev_ord_as_pred_PRECISION_AVG,
		AVG(prev_ord_as_pred_RECALL) AS prev_ord_as_pred_RECALL_AVG,
		STDDEV(prev_ord_as_pred_F1) AS prev_ord_as_pred_F1_STD,
		STDDEV(prev_ord_as_pred_PRECISION) AS prev_ord_as_pred_PRECISION_STD,
		STDDEV(prev_ord_as_pred_RECALL) AS prev_ord_as_pred_RECALL_STD
		
	FROM
		instacart_features.OU_features
		
	GROUP BY
		user_id
	
    ORDER BY
		user_id ASC
;

CREATE TABLE instacart_features.U_features AS
	SELECT
		PU1.user_id,
		PU1.orders_TOT_COUNT,
		PU1.prods_ordered_DIST_CNT,
		PU1.prods_ordered_TOT_CNT,
		PU1.departments_DIST_CNT,
		PU1.aisle_DIST_CNT,
		PU2.prods_per_order_LOG_AVG,
		PU2.prods_per_order_LOG_STD,
		OU1.days_between_orders_LOG_AVG,
		OU1.days_between_orders_LOG_STD,
		OU2.prev_ord_as_pred_F1_AVG,
		OU2.prev_ord_as_pred_PRECISION_AVG,
		OU2.prev_ord_as_pred_RECALL_AVG,
		OU2.prev_ord_as_pred_F1_STD,
		OU2.prev_ord_as_pred_PRECISION_STD,
		OU2.prev_ord_as_pred_RECALL_STD
	FROM
		instacart_intermediate.U_features_1_PU1 AS PU1
        
        INNER JOIN instacart_intermediate.U_features_1_PU2 AS PU2 ON
			PU1.user_id = PU2.user_id
        
        INNER JOIN instacart_intermediate.U_features_1_OU1 AS OU1 ON
			PU1.user_id = OU1.user_id
        
        INNER JOIN instacart_intermediate.U_features_1_OU2 AS OU2 ON
			PU1.user_id = OU2.user_id
		
    ORDER BY
		user_id ASC