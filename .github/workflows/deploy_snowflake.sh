#!/bin/bash

echo "🔐 Connecting to Snowflake..."
echo "📄 Executing SQL file: $1"

snowsql -a $SNOWFLAKE_ACCOUNT \
        -u $SNOWFLAKE_USER \
        -p $SNOWFLAKE_PWD \
        -r $SNOWFLAKE_ROLE \
        -w $SNOWFLAKE_WAREHOUSE \
        -d $SNOWFLAKE_DATABASE \
        -s $SNOWFLAKE_SCHEMA \
        -f "$1"

echo "✅ SQL deployment complete."
