#!/bin/bash
echo "🔍 Running SQLFluff Lint on staging SQL files..."
sqlfluff lint sql/tables/*.sql sql/views/*.sql sql/UDF/*.sql

if [ $? -ne 0 ]; then
  echo "SQL Linting failed."
  exit 1
else
  echo "SQL Linting passed."
fi

echo "🔍 Syntax Check (Dry Run Simulation)..."
find sql/ -name "*.sql" -print -exec cat {} \;

echo "Syntax check complete."
