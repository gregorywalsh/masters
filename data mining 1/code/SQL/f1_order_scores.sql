INSERT INTO  instacart_intermediate.prev_order_prediction_f1
	SELECT
			opa_cur.user_id AS user_id,
			opa_cur.order_number AS predicted_order_num,
            opa_cur.order_id AS predicted_order_id,
			opa_cur.product_id AS product_id,
			opa_prev.product_id IS NOT NULL AS in_prev_order,
			TRUE AS in_cur_order
			
	FROM
		order_products__all AS opa_cur
		
		LEFT JOIN order_products__all AS opa_prev ON (
			opa_prev.order_number = opa_cur.order_number - 1
			AND opa_prev.user_id = opa_cur.user_id
			AND opa_prev.product_id = opa_cur.product_id
		)
		
	WHERE
		opa_cur.order_number >= 2

-- 	ORDER BY
-- 		user_id,
-- 		predicted_order_num,
-- 		IF(opa_prev.product_id IS NULL, 0, 1)
;

INSERT INTO  instacart_intermediate.prev_order_prediction_f1
	SELECT
			opa_prev.user_id AS user_id,
			opa_prev.order_number + 1 AS predicted_order_num,
            o.order_id as predicted_order_id,
			opa_prev.product_id AS product_id,
			TRUE AS in_prev_order,
			FALSE AS in_cur_order
			
	FROM
		order_products__all AS opa_prev
        
        INNER JOIN orders AS o ON (
			o.user_id = opa_prev.user_id
            AND o.order_number = opa_prev.order_number + 1
		)
		
		LEFT JOIN order_products__all AS opa_cur ON (
			opa_cur.order_number = opa_prev.order_number + 1
			AND opa_prev.user_id = opa_cur.user_id
			AND opa_prev.product_id = opa_cur.product_id
		)
		
	WHERE
		opa_cur.user_id IS NULL
        
;

# REMOVE TEST AND TRAIN DATA
SET SQL_SAFE_UPDATES = 0;
DELETE FROM instacart_intermediate.prev_order_prediction_f1
USING instacart_intermediate.prev_order_prediction_f1, orders 
WHERE instacart_intermediate.prev_order_prediction_f1.predicted_order_id = orders.order_id AND orders.eval_set IN ('test', 'train');
;

# INSERT DATA INTO THE CORRECT 
INSERT INTO instacart_intermediate.OU_features_F1
	SELECT
		user_id,
		predicted_order_id AS order_id,
        predicted_order_num AS order_num,
		recall AS prev_ord_as_pred_RECALL,
		`precision` prev_ord_as_pred_PRECISION,
		IF(recall + `precision` = 0, 0, 2 * (recall * `precision`) / (recall + `precision`) ) AS prev_ord_as_pred_F1
    
	FROM
		(SELECT
			user_id,
 			predicted_order_id,
 			predicted_order_num,
-- 			SUM(in_cur_order) AS actual_count,
-- 			SUM(in_prev_order) AS predicted_count,
-- 			SUM(in_cur_order * in_prev_order) AS true_positives,
-- 			SUM((NOT in_cur_order) * in_prev_order) AS false_positives,
-- 			SUM(in_cur_order * (NOT in_prev_order)) AS false_negatives,
			SUM(in_cur_order * in_prev_order) / SUM(in_cur_order) AS recall,
			SUM(in_cur_order * in_prev_order) / SUM(in_prev_order) AS `precision`
			
		FROM
			instacart_intermediate.prev_order_prediction_f1 AS f1
			
		GROUP BY
			user_id,
            predicted_order_id,
			predicted_order_num
			
		) AS f1

	ORDER BY
		user_id,
        predicted_order_num