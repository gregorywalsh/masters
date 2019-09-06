# DATA ENHANCEMENT

# Add user_id and order_num to order_products__all

SET SQL_SAFE_UPDATES = 0;

UPDATE order_products__all AS opa
	INNER JOIN orders AS o ON
		o.order_id = opa.order_id
SET
	opa.order_number = o.order_number,
    opa.user_id = o.user_id
    
WHERE
	opa.order_id <> 0
    AND opa.product_id <> 0
;

# Add order_number to order_products__all
UPDATE
	order_products__all AS opa
    
    INNER JOIN orders AS o ON
		o.order_id = opa.order_id
    
SET
	opa.order_number = o.order_number

WHERE
	opa.order_id >= 0
;


# Add days since first order to orders
SET
	@csum := 0,
    @prior_user_id := 0 
;
    
UPDATE
	orders
SET
	days_since_first_order = IF(user_id = @prior_user_id, @csum := @csum + ifnull(days_since_prior_order, 0), @csum := 0),
    temp = @prior_user_id := user_id
WHERE
	order_id >= 0
    AND user_id < 10
ORDER BY
	user_id, order_number
;