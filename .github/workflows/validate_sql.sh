USE DATABASE {{ secrets.SNOWSQL_DATABASE_STAGE }};
USE SCHEMA {{ secrets.SNOWSQL_SCHEMA }};

-- Simulate applying all SQL
!source sql/tables/sample_table.sql;
!source sql/views/sample_view.sql;
!source sql/UDF/sample_udf.sql;
