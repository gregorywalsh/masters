TRUNCATE instacart.order_products__prior;
LOAD DATA LOCAL 
	INFILE '~/Downloads/order_products__prior.csv'
	INTO TABLE instacart.order_products__prior
	FIELDS TERMINATED BY ','
	ENCLOSED BY ''
	ESCAPED BY '\\'
	LINES TERMINATED BY '\n'
    IGNORE 1 LINES
;

TRUNCATE instacart.order_products__train;
LOAD DATA LOCAL 
	INFILE '~/Downloads/order_products__train.csv'
	INTO TABLE instacart.order_products__train
	FIELDS TERMINATED BY ','
	ENCLOSED BY ''
	ESCAPED BY '\\'
	LINES TERMINATED BY '\n'
    IGNORE 1 LINES
;

TRUNCATE instacart.orders;
LOAD DATA LOCAL 
	INFILE '~/Downloads/orders.csv'
	INTO TABLE instacart.orders
	FIELDS TERMINATED BY ','
	ENCLOSED BY ''
	ESCAPED BY '\\'
	LINES TERMINATED BY '\n'
    IGNORE 1 LINES
;

TRUNCATE instacart.products;
LOAD DATA LOCAL 
	INFILE '~/Downloads/products.csv'
	INTO TABLE instacart.products
	FIELDS TERMINATED BY ','
	ENCLOSED BY '"'
	ESCAPED BY '\\'
	LINES TERMINATED BY '\n'
    IGNORE 1 LINES
;

TRUNCATE instacart.order_products__all;
INSERT INTO instacart.order_products__all
SELECT *, "prior"
FROM instacart.order_products__prior;

INSERT INTO instacart.order_products__all
SELECT *, "train"
FROM instacart.order_products__train