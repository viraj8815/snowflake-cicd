#!/bin/bash

echo "Looking for SQL files under 'sql/'..."
SQL_FILES=$(find sql -type f -name "*.sql")

echo "Found SQL files:"
echo "$SQL_FILES"

echo "Running sqlfluff linter..."
for file in $SQL_FILES; do
  echo "Linting $file"
  sqlfluff lint "$file" --dialect snowflake
done

echo "SQL lint check completed (simulated dry run)"
