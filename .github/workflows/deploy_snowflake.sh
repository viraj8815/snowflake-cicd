#!/bin/bash

echo "Deploying SQL scripts to Snowflake..."

for file in $(find sql/ -type f -name "*.sql"); do
  echo "Executing $file..."
  snowsql -a "$SNOWFLAKE_ACCOUNT" \
          -u "$SNOWFLAKE_USER" \
          -r "$SNOWFLAKE_ROLE" \
          -w "$SNOWFLAKE_WH" \
          -d "$SNOWFLAKE_DB" \
          -s "$SNOWFLAKE_SCHEMA" \
          -q "$(cat $file)"
done

echo "Deployment to Snowflake complete."
