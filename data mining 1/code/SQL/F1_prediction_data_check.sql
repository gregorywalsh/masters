# CHECK ONLY PRIOR DATA INCLUDED FOR F1 PREDICTIONS
SELECT
	user_id,
	predicted_order_id,
	predicted_order_num,
	SUM(in_cur_order) AS actual_count,
	SUM(in_prev_order) AS predicted_count
	
FROM
	instacart_intermediate.prev_order_prediction_f1 AS f1
	
GROUP BY
	user_id,
	predicted_order_id,
	predicted_order_num

HAVING
	actual_count = 0
    OR predicted_count = 0