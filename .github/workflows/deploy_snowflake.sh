#!/bin/bash
echo "🚀 Deploying SQL scripts to Snowflake..."

for file in sql/tables/*.sql sql/views/*.sql sql/UDF/*.sql; do
  echo "🔹 Executing $file..."

  snow sql \
    --account "${SNOWFLAKE_ACCOUNT}" \
    --user "${SNOWFLAKE_USERNAME}" \
    --password "${SNOWFLAKE_PASSWORD}" \
    --role "${SNOWFLAKE_ROLE}" \
    --warehouse "${SNOWFLAKE_WAREHOUSE}" \
    --database "${SNOWFLAKE_DATABASE}" \
    --schema "${SNOWFLAKE_SCHEMA}" \
    --file "$file"
done

echo "✅ Deployment to Snowflake complete."
