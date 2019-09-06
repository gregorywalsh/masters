# user_product_timeBetweenPurchaseSET
SET
    @prior_user_id := 0,
    @prior_product_id := 0,
    @prior_days_since_first_order := 0
    
;
    
SELECT
	opa.product_id,
    opa.order_id,
    o.user_id,
    o.order_number,
    o.days_since_first_order,
    IF(
		opa.reordered = 1,
		IF(
			o.user_id = @prior_user_id AND opa.product_id = @prior_product_id,
			o.days_since_first_order - @prior_days_since_first_order,
            @prior_days_since_first_order := 0
		),
        NULL
	) AS days_since_prior_product_order,
    @prior_user_id := o.user_id,
	@prior_days_since_first_order := o.days_since_first_order,
    @prior_product_id := opa.product_id
    
FROM 
	order_products__all AS opa
    
    INNER JOIN orders AS o ON
		o.order_id = opa.order_id

WHERE
	o.order_id >= 0
    AND o.user_id < 10
    
ORDER BY
	opa.product_id,
    o.user_id,
    o.order_number
;

# F1 Score for user baskets
SELECT
	1
FROM
	order_products__all
	
	