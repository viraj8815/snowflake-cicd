CREATE OR REPLACE TABLE STAGE_DB.PUBLIC.sample_stage_table (
  id INT,
  name STRING,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);
