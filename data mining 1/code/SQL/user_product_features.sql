#PRODUCT USER FEATURES - PU1
CREATE TABLE instacart_intermediate.PU_features_1_PU1 AS

SELECT
	opa.user_id,
    opa.product_id,
	COUNT(DISTINCT opa.order_id) AS purchased_CNT,
    COUNT(DISTINCT opa.order_id) / MIN(uf.orders_TOT_COUNT) AS proportion_of_orders_containing_PROP,
    AVG(LOG(opa.add_to_cart_order)) AS position_in_cart_LOG_AVG,
    STDDEV(LOG(opa.add_to_cart_order)) AS position_in_cart_LOG_STD
    
FROM
	order_products__all AS opa
        
	INNER JOIN instacart_features.U_features AS uf ON
		opa.user_id = uf.user_id
    
WHERE
	opa.source = 'prior'
    
GROUP BY
	opa.user_id,
    opa.product_id
;

#PRODUCT USER FEATURES - PU2
CREATE TABLE instacart_intermediate.PU_features_1_PU2 AS

SELECT
	purchase_hist.user_id,
    purchase_hist.product_id,
--     purchase_hist.last_order_number,
--     purchase_hist.first_order_number,
--     purchase_hist.num_times_purchased,
--     last_ordered.days_since_first_order AS last_purchase_date,
--     first_ordered.days_since_first_order AS first_purchase_date,
--     current_order.days_since_first_order AS current_order_date,
    current_order.days_since_first_order - last_ordered.days_since_first_order AS days_since_last_purchased_CNT,
    IF(
		purchase_hist.num_times_purchased = 1,
		NULL, 
        LOG(1 + last_ordered.days_since_first_order - first_ordered.days_since_first_order) / (purchase_hist.num_times_purchased - 1)
	)AS delay_between_purchases_LOG_AVG
    
    
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
        
	INNER JOIN instacart_features.U_features AS uf ON
		purchase_hist.user_id = uf.user_id
	
    INNER JOIN orders AS current_order ON
		purchase_hist.user_id = current_order.user_id
        AND uf.orders_TOT_COUNT + 1 = current_order.order_number
        
;

CREATE TABLE instacart_features.PU_features AS

SELECT
	pu1.user_id,
    pu1.product_id,
	pu1.purchased_CNT,
    pu1.proportion_of_orders_containing_PROP,
    pu1.position_in_cart_LOG_AVG,
    pu1.position_in_cart_LOG_STD,
    pu2.days_since_last_purchased_CNT,
    pu2.delay_between_purchases_LOG_AVG
    
FROM
	instacart_intermediate.PU_features_1_PU1 AS pu1
    
    INNER JOIN instacart_intermediate.PU_features_1_PU2 AS pu2 ON
		pu1.product_id = pu2.product_id
        AND pu1.user_id = pu2.user_id
	