CREATE TABLE instacart_intermediate.test_train_orders AS 
SELECT
	user_id,
    order_id,
    eval_set

FROM
	instacart.orders
    
WHERE
	eval_set IN ('test', 'train')
;
    
INSERT INTO instacart_features.ALL_features

SELECT
	`PU_features`.`user_id`,
    `PU_features`.`product_id`,
    `PU_features`.`purchased_CNT`,
    `PU_features`.`proportion_of_orders_containing_PROP`,
    `PU_features`.`position_in_cart_LOG_AVG`,
    `PU_features`.`position_in_cart_LOG_STD`,
    `PU_features`.`days_since_last_purchased_CNT`,
    `PU_features`.`delay_between_purchases_LOG_AVG`,
    `P_features`.`orders_CNT`,
    `P_features`.`reorders_CNT`,
    `P_features`.`proportion_reordered_PROP`,
    `P_features`.`occurances_per_order_PROP`,
    `P_features`.`delay_between_reorders_LOG_AVG`,
    `U_features`.`orders_TOT_COUNT`,
    `U_features`.`prods_ordered_DIST_CNT`,
    `U_features`.`prods_ordered_TOT_CNT`,
    `U_features`.`departments_DIST_CNT`,
    `U_features`.`aisle_DIST_CNT`,
    `U_features`.`prods_per_order_LOG_AVG`,
    `U_features`.`prods_per_order_LOG_STD`,
    `U_features`.`days_between_orders_LOG_AVG`,
    `U_features`.`days_between_orders_LOG_STD`,
    `U_features`.`prev_ord_as_pred_F1_AVG`,
    `U_features`.`prev_ord_as_pred_PRECISION_AVG`,
    `U_features`.`prev_ord_as_pred_RECALL_AVG`,
    `U_features`.`prev_ord_as_pred_F1_STD`,
    `U_features`.`prev_ord_as_pred_PRECISION_STD`,
    `U_features`.`prev_ord_as_pred_RECALL_STD`,
    test_train_orders.order_id,
    IF(order_products__all.order_id IS NULL, 0, 1) AS target

FROM
	instacart_features.PU_features
    
    INNER JOIN instacart_features.P_features ON
		P_features.product_id = PU_features.product_id
    
    INNER JOIN instacart_features.U_features ON
		U_features.user_id = PU_features.user_id
        
	INNER JOIN instacart_intermediate.test_train_orders ON
		test_train_orders.user_id = PU_features.user_id
        
	LEFT JOIN order_products__all ON
		order_products__all.order_id = test_train_orders.order_id
        AND order_products__all.product_id = PU_features.product_id
