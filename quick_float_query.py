#!/usr/bin/env python3
"""
Quick query for float 5901373 - profile count and last measurement date
"""

import duckdb
import pandas as pd
from pathlib import Path

# Connect to the parquet database
data_dir = Path(".")

try:
    # Initialize DuckDB connection
    conn = duckdb.connect()

    # Query for float 5901373
    float_id = "5901373"

    query = f"""
    SELECT
        p.float_id,
        COUNT(DISTINCT p.profile_id) as total_profiles,
        MAX(p.profile_date) as last_measurement_date,
        COUNT(m.measurement_id) as total_measurements
    FROM 'parquet_data/profiles.parquet' p
    LEFT JOIN 'parquet_data/measurements.parquet' m ON p.profile_id = m.profile_id
    WHERE p.float_id = '{float_id}'
    GROUP BY p.float_id
    """

    result = conn.execute(query).fetchdf()

    if not result.empty:
        row = result.iloc[0]
        print(f"Float {float_id}:")
        print(f"- Total Profiles: {row['total_profiles']:,}")
        print(f"- Last Measurement: {row['last_measurement_date']}")
        print(f"- Total Measurements: {row['total_measurements']:,}")
    else:
        print(f"No data found for float {float_id}")

except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()