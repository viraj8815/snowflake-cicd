CREATE OR REPLACE VIEW STAGE_DB.PUBLIC.CUSTOMERS_VIEW AS
SELECT
"customerid",
"companyname",
"contactname",
"contacttitle",
"address",
"city",
"region",
"postalcode",
"country",
FROM STAGE_DB.PUBLIC.CUSTOMERS;
