#!/usr/bin/env python3
"""
Check schema structure of all 3 parquet files
"""

import duckdb
import pandas as pd
from pathlib import Path

# Connect to the parquet database
try:
    # Initialize DuckDB connection
    conn = duckdb.connect()

    # List of parquet files to check
    parquet_files = [
        'parquet_data/floats.parquet',
        'parquet_data/profiles.parquet',
        'parquet_data/measurements.parquet'
    ]

    for file_path in parquet_files:
        print(f"\n{'='*60}")
        print(f"SCHEMA: {file_path}")
        print(f"{'='*60}")

        # Get schema information
        schema_query = f"DESCRIBE SELECT * FROM '{file_path}' LIMIT 1"
        schema_result = conn.execute(schema_query).fetchdf()

        print(f"Total Columns: {len(schema_result)}")
        print(f"\nColumn Details:")
        print("-" * 40)

        for idx, row in schema_result.iterrows():
            print(f"{idx+1:2d}. {row['column_name']:<25} | {row['column_type']}")

        # Get sample record count
        count_query = f"SELECT COUNT(*) as total_records FROM '{file_path}'"
        count_result = conn.execute(count_query).fetchdf()
        print(f"\nTotal Records: {count_result.iloc[0]['total_records']:,}")

        # Show first few sample values
        sample_query = f"SELECT * FROM '{file_path}' LIMIT 3"
        sample_result = conn.execute(sample_query).fetchdf()

        print(f"\nSample Data (first 3 rows):")
        print("-" * 60)
        print(sample_result.to_string(index=False, max_cols=6))

except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()