#!/usr/bin/env python3
"""
Quick test of argopy functionality for hackathon
"""

import argopy
import pandas as pd
from datetime import datetime, timedelta

def test_argopy_basic():
    """Test basic argopy functionality"""
    print("Testing argopy basic functionality...")

    try:
        # Initialize fetcher
        loader = argopy.DataFetcher()
        print("✅ Argopy DataFetcher initialized")

        # Test small region fetch (Indian Ocean subset)
        print("🌊 Fetching small Indian Ocean region...")
        ds = loader.region([65, 75, 15, 25, 0, 100]).load()  # Small Arabian Sea region
        print(f"✅ Data fetched: {ds.dims}")

        # Convert to DataFrame
        df = ds.to_dataframe()
        print(f"✅ DataFrame created: {len(df)} rows")

        # Check available columns
        print(f"📊 Available columns: {list(df.columns)}")

        # Check float count
        if 'PLATFORM_NUMBER' in df.columns:
            float_count = df['PLATFORM_NUMBER'].nunique()
            print(f"🛟 Active floats found: {float_count}")

            # Show sample float IDs
            sample_floats = sorted(df['PLATFORM_NUMBER'].unique())[:5]
            print(f"📋 Sample float IDs: {sample_floats}")

        # Check data ranges
        if 'TEMP' in df.columns:
            temp_data = df['TEMP'].dropna()
            if len(temp_data) > 0:
                print(f"🌡️ Temperature range: {temp_data.min():.2f} to {temp_data.max():.2f}°C")

        if 'LATITUDE' in df.columns:
            lat_range = f"{df['LATITUDE'].min():.2f} to {df['LATITUDE'].max():.2f}"
            lon_range = f"{df['LONGITUDE'].min():.2f} to {df['LONGITUDE'].max():.2f}"
            print(f"📍 Geographic coverage: Lat {lat_range}°N, Lon {lon_range}°E")

        return True, df

    except Exception as e:
        print(f"❌ Argopy test failed: {e}")
        return False, None

def test_single_float():
    """Test fetching data for a single float"""
    print("\n🛟 Testing single float data fetch...")

    try:
        loader = argopy.DataFetcher()
        # Use a known active float (this might work)
        test_float = 2902746  # Common in Indian Ocean

        ds = loader.float([test_float]).load()
        df = ds.to_dataframe()

        print(f"✅ Float {test_float} data: {len(df)} measurements")

        if 'TEMP' in df.columns:
            temp_data = df['TEMP'].dropna()
            print(f"🌡️ Temperature measurements: {len(temp_data)}")

        return True, df

    except Exception as e:
        print(f"❌ Single float test failed: {e}")
        return False, None

if __name__ == "__main__":
    print("="*60)
    print("ARGOPY FUNCTIONALITY TEST FOR HACKATHON")
    print("="*60)

    # Test 1: Basic region fetch
    success1, data1 = test_argopy_basic()

    # Test 2: Single float fetch
    success2, data2 = test_single_float()

    print(f"\n🎯 TEST RESULTS:")
    print(f"Basic region fetch: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Single float fetch: {'✅ PASS' if success2 else '❌ FAIL'}")

    if success1:
        print("\n🚀 READY FOR HACKATHON IMPLEMENTATION!")
    else:
        print("\n⚠️ Need to debug argopy connection")