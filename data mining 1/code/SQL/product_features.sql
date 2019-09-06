-- Global Rate of reorder of particular product
-- Proportion of all orders which contain a particular product
-- How many times an item have been reordered
-- How many times an item have been ordered
-- Average delay between reorders of a product across all users

# PRODUCT FEATURES - P1
SELECT @total_orders := COUNT(*) FROM orders WHERE eval_set = 'prior'
;

INSERT INTO instacart_intermediate.P_features_1_OP1

SELECT
	opa.product_id,
    COUNT(DISTINCT(opa.user_id)) AS orders_CNT,
    COUNT(DISTINCT IF(opa.reordered = 1, opa.user_id, NULL)) AS reorders_CNT,
	COUNT(DISTINCT IF(opa.reordered = 1, opa.user_id, NULL)) / COUNT(DISTINCT(opa.user_id)) AS proportion_reordered_PROP,
    COUNT(*) / @total_orders AS occurances_per_order_PROP
    
FROM
	order_products__all AS opa

WHERE
	opa.source = 'prior'
    
GROUP BY
	opa.product_id
;

# PRODUCT FEATURES - P2
INSERT INTO instacart_intermediate.P_features_1_OP2

SELECT
	product_id,
	IF(
		total_number_of_reorders = 0,
        NULL,
		LOG( (1 + total_days_between_all_orders) / total_number_of_reorders) 
	) AS delay_between_reorders_LOG_AVG

FROM
	(SELECT
		purchase_hist.product_id,

		SUM(
			IF(
				purchase_hist.num_times_purchased = 1,
				NULL,
				last_ordered.days_since_first_order - first_ordered.days_since_first_order
			)
		) AS total_days_between_all_orders,
			
		SUM(
			IF(
				purchase_hist.num_times_purchased = 1,
				NULL,
				purchase_hist.num_times_purchased - 1
			)
		) AS total_number_of_reorders
		
		
	FROM

		(SELECT
			opa.user_id,
			opa.product_id,
			MAX(opa.order_number) AS last_order_number,
			MIN(opa.order_number) AS first_order_number,
			COUNT(*) AS num_times_purchased

		FROM
			order_products__all AS opa
			
		WHERE
			opa.source = 'prior'
		
		GROUP BY
			opa.user_id,
			opa.product_id
		) AS purchase_hist
		
		INNER JOIN orders AS last_ordered ON
			purchase_hist.user_id = last_ordered.user_id
			AND purchase_hist.last_order_number = last_ordered.order_number
		
		INNER JOIN orders AS first_ordered ON
			purchase_hist.user_id = first_ordered.user_id
			AND purchase_hist.first_order_number = first_ordered.order_number
			
	GROUP BY
		purchase_hist.product_id
	) AS history_agg
;

# PRODUCT FEATURES AGG
INSERT INTO instacart_features.P_features

SELECT
	p1.product_id,
    p1.orders_CNT,
    p1.reorders_CNT,
	p1.proportion_reordered_PROP,
    p1.occurances_per_order_PROP,
    p2.delay_between_reorders_LOG_AVG
FROM
	instacart_intermediate.P_features_1_OP1 AS p1
    
    INNER JOIN instacart_intermediate.P_features_1_OP2 AS p2 ON
		p1.product_id = p2.product_id

