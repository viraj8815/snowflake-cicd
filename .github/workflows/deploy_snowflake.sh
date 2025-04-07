#!/bin/bash

echo "Deploying SQL scripts to Snowflake..."

for file in sql/tables/*.sql sql/views/*.sql sql/UDF/*.sql; do
  echo "🔹 Executing $file..."

  snow sql \
    --account-name "${SNOWFLAKE_ACCOUNT}" \
    --username "${SNOWFLAKE_USERNAME}" \
    --password "${SNOWFLAKE_PASSWORD}" \
    --role "${SNOWFLAKE_ROLE}" \
    --warehouse "${SNOWFLAKE_WAREHOUSE}" \
    --database "${SNOWFLAKE_DATABASE}" \
    --schema "${SNOWFLAKE_SCHEMA}" \
    --filename "$file"
done

echo "Deployment to Snowflake complete."
