
---

## The Orchestrator Master Prompt

**Copy the entire text below and paste it into the Gemini CLI.**

```python
# Step 1: Interact with the user to get the category
category = input("Please enter the query category you will be working on (e.g., 'Geographic/Spatial', 'Temporal'): ")
Basic Data Retrieval: 50 samples
Geographic/Spatial: 100 samples
Temporal: 200 samples
Depth/Vertical Structure: 150 samples
Parameter-Specific: 300 samples
Statistical and Analytical: 250 samples
Comparison: 180 samples
Quality Control and Data Assessment: 75 samples
Visualization-Oriented: 120 samples
Operational/Monitoring: 40 samples
Export and Data Management: 30 samples
Scientific Research: 80 samples
Total Required Samples: 1,575
print(f"Initiating orchestration for category: {category}")

# Step 2: Define the optimal JSON structure for the category
for the selected category , define the best structure in the same format as the .json file in the semantic_sample folder and get the best possible content to increase the semantical search 
# (Logic for selecting the optimal structure is embedded in the next step)

# Step 3: Generate the first batch of 10 prompts
print("---"Generating Batch 1 (Prompts 1-10) ---") 
the prompt should be generated in such a way that it covers most of the user queries and it should be uniqe and different from each other and it should be well defined and it is formed in such a way that it have unique semantically similarity , which can be easily extracted using rag

# Step 3a: Generate the JSON prompts
# (You will now generate the 10 JSON prompts for the specified category.
# Use the optimal structure for the category and the database schema below.
# The output should be a single JSON object with a "queries" list.)

# Step 3b: Create the JSON file
# (You will now use the write_file tool to create a new JSON file.
# The file path will be f"semantic_samples/{category.lower().replace(' ', '_')}.json".
# The content of the file will be the JSON object you just generated.)

# Step 3c: Generate the testing script
# (You will now generate a Python script named "test_batch.py".
# This script will test the similarity scores of the 10 prompts you just created.
# It must test at least 20 variations of the 10 prompts.)

# Step 3d: Clean the ChromaDB directory
# (You will now run a shell command to delete the old ChromaDB directory to ensure a clean build.
# For Windows, the command is: rd /s /q "new_comprehensive_chromadb"
# For macOS/Linux, the command is: rm -rf "new_comprehensive_chromadb")

# Step 3e: Run the populating script
# (You will now run the command: python new_comprehensive_rag_system.py)

# Step 3e: Run the testing script
# (You will now run the command: python test_batch.py)

# Step 3f: Analyze and refine
# (You will now analyze the "imagined" output of the test script.
# If any prompts have an average similarity score below 0.85, you will generate a new, improved version of the JSON object.
# In the new version, you will modify the `content` of the low-scoring prompts.)

# Step 3g: Update the JSON file with refined prompts
# (If you generated a refined version, you will now use the write_file tool again to overwrite the JSON file.)

print("---"Batch 1 Complete"---")
print("If the similarity scores are good, you can now process the next batch by pasting this entire prompt again.")
print("If the scores were low, the prompts have been refined. Please test again.")


#starting the next batch read the schema to write the sql for this schema 
# Database Schema Documentation

## Overview

This database contains oceanographic data from ARGO floats in the Indian Ocean region. The data is stored in 3 main parquet files representing different aspects of the float deployment and measurement systems.

**Data Summary:**
- **Total Records:** 1,428,795 records across 3 tables
- **Data Size:** 36.0 MB total storage
- **Time Period:** 2015-2025 (deployment to processing)
- **Geographic Coverage:** Indian Ocean (-21.7°N to 35.8°N, 39.9°E to 154.6°E)
- **Float Count:** 17 ARGO floats
- **Profile Count:** 3,130 vertical profiles
- **Measurement Count:** 1,425,648 individual measurements



## Table Structure

### 1. FLOATS Table
**File:** `floats.parquet` (0.02 MB)
**Records:** 17 rows
**Description:** Master table containing metadata about each ARGO float deployment

#### Columns

| Column | Type | Null % | Description | Value Range |
|--------|------|--------|-------------|-------------|
| **float_id** | string | 0.0% | Primary identifier for float | 17 unique values (2902182-2902748) |
| **wmo_number** | int64 | 0.0% | WMO identification number | 2902182 - 2902748 |
| **program_name** | string | 0.0% | ARGO program designation | 5 programs (INDIAN-OCEAN-ARGO, etc.) |
| **platform_type** | string | 0.0% | Float hardware type | UNKNOWN, APEX |
| **data_assembly_center** | string | 0.0% | Data processing center | argo-erddap, indian-ocean-consortium |
| **deployment_date** | string | 0.0% | ISO timestamp of deployment | 2015-10-15 to 2018-05-31 |
| **deployment_latitude** | float64 | 0.0% | Initial deployment latitude | -13.97° to 33.96°N |
| **deployment_longitude** | float64 | 0.0% | Initial deployment longitude | 66.87° to 147.84°E |
| **deployment_depth** | string | 100.0% | Deployment depth (not recorded) | NULL |
| **current_status** | string | 0.0% | Operational status | DEAD, INACTIVE |
| **last_latitude** | float64 | 0.0% | Last recorded position latitude | -20.84° to 29.56°N |
| **last_longitude** | float64 | 0.0% | Last recorded position longitude | 46.31° to 144.97°E |
| **last_update** | string | 0.0% | Last data transmission | 2021-05-11 to 2023-09-28 |
| **cycle_time_days** | int64 | 0.0% | Profile cycle interval | 10 days (standard) |
| **park_pressure_dbar** | string | 100.0% | Parking depth pressure | NULL |
| **profile_pressure_dbar** | string | 100.0% | Profile depth pressure | NULL |
| **total_profiles** | int64 | 0.0% | Number of profiles collected | 3 - 308 profiles |
| **quality_profiles** | int64 | 0.0% | Number of quality-controlled profiles | 0 (all profiles) |
| **metadata_text** | string | 0.0% | Detailed float description | Rich metadata summaries |
| **created_at** | string | 0.0% | Database record creation | 2025-09-07 |
| **updated_at** | string | 0.0% | Database record update | 2025-09-07 |

---

### 2. PROFILES Table
**File:** `profiles.parquet` (0.29 MB)
**Records:** 3,130 rows
**Description:** Vertical profile metadata for each float descent/ascent cycle

#### Columns

| Column | Type | Null % | Description | Value Range |
|--------|------|--------|-------------|-------------|
| **profile_id** | int64 | 0.0% | Primary key for profile | 1 - 3130 |
| **float_id** | string | 0.0% | Foreign key to floats table | 17 unique float IDs |
| **cycle_number** | int64 | 0.0% | Sequential cycle number | 1 - 351 |
| **profile_direction** | string | 0.0% | Ascent/descent direction | A (Ascending) |
| **profile_date** | string | 0.0% | ISO timestamp of profile | 2018-05-24 to latest |
| **latitude** | float64 | 0.0% | Profile location latitude | -21.66° to 35.76°N |
| **longitude** | float64 | 0.0% | Profile location longitude | 39.88° to 154.62°E |
| **max_pressure** | float64 | 0.0% | Maximum depth reached | 943.0 - 2025.5 dbar |
| **num_levels** | int64 | 0.0% | Number of measurement levels | 22 - 1000 levels |
| **vertical_sampling_scheme** | string | 100.0% | Sampling methodology | NULL |
| **data_mode** | string | 0.0% | Processing mode | R (Real-time) |
| **data_quality_flag** | int64 | 0.0% | Overall quality indicator | 1 (Good quality) |
| **processing_date** | string | 100.0% | Data processing timestamp | NULL |
| **netcdf_filename** | string | 100.0% | Source NetCDF file | NULL |
| **file_checksum** | string | 100.0% | Data integrity hash | NULL |
| **profile_summary** | string | 0.0% | Human-readable profile description | Rich profile summaries |
| **created_at** | string | 0.0% | Database record creation | 2025-09-07 |
| **updated_at** | string | 0.0% | Database record update | 2025-09-07 |

---

### 3. MEASUREMENTS Table
**File:** `measurements.parquet` (35.65 MB)
**Records:** 1,425,648 rows
**Description:** Individual oceanographic measurements at specific depths

#### Primary Data Columns

| Column | Type | Null % | Description | Value Range |
|--------|------|--------|-------------|-------------|
| **measurement_id** | int64 | 0.0% | Primary key for measurement | 1 - 1,425,648 |
| **profile_id** | int64 | 0.0% | Foreign key to profiles table | 1 - 3130 |
| **pressure** | float64 | 0.0% | Sea pressure (dbar) | 0.0 - 2025.5 dbar |
| **depth** | float64 | 0.0% | Calculated depth (meters) | 0.0 - 2065.4 m |
| **temperature** | float64 | 0.0% | Sea temperature (°C) | 1.90 - 31.86°C |
| **salinity** | float64 | 0.0% | Practical salinity (PSU) | 28.23 - 36.56 PSU |

#### Quality Control Columns

| Column | Type | Null % | Description | Value Range |
|--------|------|--------|-------------|-------------|
| **pressure_qc** | int64 | 0.0% | Pressure quality flag | 1 (Good) |
| **temperature_qc** | int64 | 0.0% | Temperature quality flag | 1 (Good) |
| **salinity_qc** | int64 | 0.0% | Salinity quality flag | 1 (Good) |
| **spike_test_flag** | int64 | 0.0% | Spike detection result | 1 (Pass) |
| **gradient_test_flag** | int64 | 0.0% | Gradient test result | 1 (Pass) |

#### Extended Parameters (Not Available)

| Column | Type | Null % | Description | Status |
|--------|------|--------|-------------|---------|
| **dissolved_oxygen** | string | 100.0% | Dissolved oxygen concentration | Not measured |
| **ph_in_situ** | string | 100.0% | In-situ pH measurements | Not measured |
| **chlorophyll_a** | string | 100.0% | Chlorophyll-a concentration | Not measured |
| **particle_backscattering** | string | 100.0% | Optical backscattering | Not measured |
| **downward_irradiance** | string | 100.0% | Light irradiance | Not measured |
| **potential_temperature** | string | 100.0% | Calculated potential temperature | Not calculated |
| **potential_density** | string | 100.0% | Calculated potential density | Not calculated |
| **buoyancy_frequency** | string | 100.0% | Buoyancy frequency | Not calculated |
| **mixed_layer_depth** | string | 100.0% | Mixed layer depth | Not calculated |

#### Processing Metadata

| Column | Type | Null % | Description | Value Range |
|--------|------|--------|-------------|-------------|
| **processing_level** | string | 0.0% | Data processing level | L1 (Level 1) |
| **interpolated** | int64 | 0.0% | Interpolation flag | 0 (Not interpolated) |
| **parameter_summary** | string | 0.0% | Human-readable measurement summary | 785,284 unique summaries |
| **created_at** | string | 0.0% | Database record creation | 2025-09-07 |

---

## Data Relationships

### Primary Keys and Foreign Keys

```
floats.float_id (1) ←→ (many) profiles.float_id
profiles.profile_id (1) ←→ (many) measurements.profile_id
```

### Table Relationships

- **1 Float** → **184 Profiles** (average)
- **1 Profile** → **456 Measurements** (average)
- **1 Float** → **83,862 Measurements** (average)

---

## Data Quality Summary

### Available Data
- **Core Parameters:** Temperature, Salinity, Pressure/Depth ✅
- **Quality Flags:** All measurements have QC flags ✅
- **Geolocation:** Complete lat/lon for all profiles ✅
- **Temporal Coverage:** 2015-2023 deployment period ✅

### Missing Data
- **Biogeochemical:** No oxygen, pH, chlorophyll data ❌
- **Derived Parameters:** No potential temperature/density ❌
- **Processing Metadata:** Limited NetCDF source info ❌

### Data Statistics

#### Temperature Distribution
- **Range:** 1.90°C - 31.86°C
- **Mean:** 10.0°C
- **Coverage:** 100% of measurements

#### Salinity Distribution
- **Range:** 28.23 - 36.56 PSU
- **Mean:** 34.88 PSU
- **Coverage:** 100% of measurements

#### Depth Coverage
- **Maximum Depth:** 2,065m
- **Mean Depth:** 839m
- **Vertical Resolution:** Variable (22-1000 levels per profile)

#### Geographic Coverage
- **Latitude:** -21.7°N to 35.8°N
- **Longitude:** 39.9°E to 154.6°E
- **Ocean Region:** Indian Ocean
- **Active Floats:** 17 deployments

---

## Usage Notes

### For SQL Queries
1. **Join Pattern:** `floats → profiles → measurements`
2. **Primary Keys:** Use integer IDs for performance
3. **Date Filtering:** Use ISO string format for dates
4. **Geographic Queries:** Use decimal degrees for coordinates

### Data Limitations
1. **Temporal Gaps:** Some floats stopped transmitting early
2. **Parameter Availability:** Only T/S data available
3. **Quality Control:** All data marked as QC=1 (basic quality)
4. **Processing Level:** L1 data only (no advanced corrections)

### Recommended Analyses
- **Thermohaline Properties:** Temperature-salinity relationships
- **Vertical Structure:** Depth profiles and stratification
- **Spatial Patterns:** Geographic distribution analysis
- **Temporal Trends:** Long-term changes (2015-2023)



*Generated: September 13, 2025*
*Data Source: ARGO Float Network, Indian Ocean*
*Processing: RAG System Database Schema Analysis*

# Step 4: Final testing with the web interface
print("---"After all batches for the category are complete"--- "")
print("You can start the web interface for final testing by running: python new_web_interface.py")
```

