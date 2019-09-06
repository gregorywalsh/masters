#VALIDATION

# Confirm order_numbers are accurate 

SELECT
	*
FROM (
	SELECT
		user_id,
		COUNT(*) AS count,
		MAX(order_number) AS max
	FROM
		orders AS o
	GROUP BY
		user_id
	) AS counts
    
WHERE
	counts.count != counts.max
;


# TBD Confirm reorder flag is accurate