# 🌊 ARGO Database Structure Analysis
## Comprehensive Research Document for Real Data Integration

**Target URL**: https://data-argo.ifremer.fr/
**Purpose**: Understand ARGO data structure for proper database design
**Date**: 2025-09-06

---

## 📁 ARGO Database Folder Structure Analysis

### **Main Directory Structure**
```
https://data-argo.ifremer.fr/
├── dac/                    # Data Assembly Centers
├── geo/                    # Geographic organization  
├── profiles/               # Profile data by float
├── aux/                    # Auxiliary data
├── doc/                    # Documentation
└── ftp/                    # FTP access
```

### **DAC (Data Assembly Centers) Structure**
```
dac/
├── aoml/                   # Atlantic Oceanographic and Meteorological Laboratory (USA)
├── bodc/                   # British Oceanographic Data Centre (UK)  
├── coriolis/               # Coriolis (France)
├── csio/                   # CSIO (China)
├── csiro/                  # CSIRO (Australia)
├── incois/                 # INCOIS (India) - **IMPORTANT FOR INDIAN OCEAN**
├── jma/                    # Japan Meteorological Agency
├── kma/                    # Korea Meteorological Administration
└── meds/                   # Marine Environmental Data Service (Canada)
```

### **Geographic Organization**
```
geo/
├── indian_ocean/           # Indian Ocean region data
├── atlantic_ocean/         # Atlantic Ocean region
├── pacific_ocean/          # Pacific Ocean region
├── arctic_ocean/           # Arctic region
└── southern_ocean/         # Southern Ocean
```

### **Profile Data Structure**
```
profiles/
├── [DAC]/                  # By Data Assembly Center
│   ├── [FLOAT_ID]/         # Individual float directory
│   │   ├── profiles/       # Profile NetCDF files
│   │   │   ├── D[FLOAT_ID]_[CYCLE].nc     # Delayed mode data
│   │   │   └── R[FLOAT_ID]_[CYCLE].nc     # Real-time data
│   │   ├── [FLOAT_ID]_meta.nc             # Metadata file
│   │   ├── [FLOAT_ID]_tech.nc             # Technical data
│   │   └── [FLOAT_ID]_Sprof.nc            # Synthetic profile
```

---

## 🗃️ Data Types and Storage Recommendations

### **1. PostgreSQL 2D Vector Database Storage**

#### **Suitable for 2D Spatial-Temporal Data:**
- **Float positions (Lat, Lon)**
- **Time series data (Date, Value)**
- **Depth profiles (Depth, Parameter)**
- **Surface measurements**

**Recommended Tables:**
```sql
-- Float tracking (2D position over time)
CREATE TABLE float_positions (
    id SERIAL PRIMARY KEY,
    float_id VARCHAR(20),
    latitude FLOAT,
    longitude FLOAT,
    measurement_date TIMESTAMP,
    cycle_number INTEGER,
    position_vector POINT -- 2D PostGIS point
);

-- Parameter measurements (depth-value pairs)
CREATE TABLE measurements (
    id SERIAL PRIMARY KEY,
    float_id VARCHAR(20),
    cycle_number INTEGER,
    depth FLOAT,
    temperature FLOAT,
    salinity FLOAT,
    pressure FLOAT,
    measurement_vector POINT -- (depth, value) as 2D
);
```

### **2. 3D Vector Database Storage**

#### **Suitable for 3D Spatial Data:**
- **3D Ocean positions (Lat, Lon, Depth)**
- **3D parameter distributions**
- **Volumetric ocean analysis**

**Recommended Storage:**
```sql
-- 3D ocean measurements
CREATE TABLE measurements_3d (
    id SERIAL PRIMARY KEY,
    float_id VARCHAR(20),
    latitude FLOAT,
    longitude FLOAT,  
    depth FLOAT,
    temperature FLOAT,
    salinity FLOAT,
    position_3d POINT_3D, -- (lat, lon, depth)
    measurement_time TIMESTAMP
);

-- 3D parameter vectors for ML/AI analysis
CREATE TABLE parameter_vectors_3d (
    id SERIAL PRIMARY KEY,
    float_id VARCHAR(20),
    cycle_number INTEGER,
    spatial_vector VECTOR(3),      -- [lat, lon, depth]
    parameter_vector VECTOR(3),    -- [temp, sal, pressure]
    quality_vector VECTOR(3)       -- [temp_qc, sal_qc, pres_qc]
);
```

---

## 📊 ARGO Data File Types and Contents

### **NetCDF Profile Files**
**File naming**: `D1901393_001.nc` or `R1901393_001.nc`
- **D** = Delayed mode (quality controlled)
- **R** = Real-time mode (preliminary)
- **1901393** = Float WMO number
- **001** = Cycle number

**Key Variables in Profile Files:**
```python
VARIABLES = {
    # Position and time
    'LATITUDE': 'Float latitude',
    'LONGITUDE': 'Float longitude', 
    'JULD': 'Julian day (time)',
    
    # Measurements
    'PRES': 'Pressure (decibar)',
    'TEMP': 'Temperature (Celsius)',
    'PSAL': 'Practical salinity (PSU)',
    
    # Quality flags
    'PRES_QC': 'Pressure quality flag',
    'TEMP_QC': 'Temperature quality flag', 
    'PSAL_QC': 'Salinity quality flag',
    
    # Additional parameters
    'DOXY': 'Dissolved oxygen (μmol/kg)',
    'CHLA': 'Chlorophyll-A (mg/m³)',
    'BBP': 'Particle backscattering',
}
```

### **Metadata Files**
**File naming**: `1901393_meta.nc`
**Contains**:
- Float configuration
- Sensor information
- Deployment details
- Mission parameters

### **Technical Files**
**File naming**: `1901393_tech.nc`
**Contains**:
- Battery voltage
- Pump activity
- System diagnostics
- Engineering data

---

## 🌏 Indian Ocean Specific Data Sources

### **INCOIS DAC (India)**
**Path**: `dac/incois/`
**Relevant for Chennai/Indian Ocean**:
- Float deployments in Bay of Bengal
- Arabian Sea measurements
- Monsoon oceanography data
- Regional quality control

### **Geographic Filtering for Indian Ocean**
**Coordinate Bounds**:
```python
INDIAN_OCEAN_BOUNDS = {
    'latitude': {'min': -60.0, 'max': 30.0},
    'longitude': {'min': 20.0, 'max': 120.0}
}

CHENNAI_REGION = {
    'latitude': {'min': 8.0, 'max': 18.0},
    'longitude': {'min': 75.0, 'max': 85.0}
}
```

---

## 💾 Data Processing Pipeline Recommendations

### **Step 1: Index Discovery**
1. Scan ARGO index files for Indian Ocean floats
2. Filter by geographic bounds
3. Identify active/recent floats

### **Step 2: Profile Download**
1. Download NetCDF profile files
2. Parse using `xarray` or `netCDF4`
3. Extract measurement arrays

### **Step 3: Data Validation**
1. Check quality flags
2. Remove bad data points
3. Apply ARGO QC standards

### **Step 4: Database Population**
```python
PROCESSING_STEPS = [
    "Download ARGO index files",
    "Filter for Indian Ocean region", 
    "Download profile NetCDF files",
    "Parse NetCDF data structure",
    "Extract measurements with QC flags",
    "Transform to database schema",
    "Populate 2D tables (lat/lon, depth/value)",
    "Populate 3D tables (lat/lon/depth)",
    "Create vector embeddings for similarity search",
    "Index for fast queries"
]
```

---

## 🔍 Key ARGO APIs and Access Methods

### **FTP Access**
- **Main FTP**: `ftp://data-argo.ifremer.fr/`
- **HTTPS Access**: `https://data-argo.ifremer.fr/`

### **Index Files**
- **Global Index**: `ar_index_global_prof.txt`
- **Metadata Index**: `ar_index_global_meta.txt`
- **Technical Index**: `ar_index_global_tech.txt`

### **Python Libraries for ARGO**
```python
REQUIRED_LIBRARIES = {
    'argopy': 'Official ARGO Python library',
    'xarray': 'NetCDF data handling', 
    'netCDF4': 'NetCDF file parsing',
    'pandas': 'Data manipulation',
    'numpy': 'Numerical computations'
}
```

---

## 🎯 Implementation Strategy

### **Phase 1: Data Discovery**
- [ ] Parse ARGO global index files
- [ ] Identify Indian Ocean floats
- [ ] Filter for quality and recency

### **Phase 2: Data Download** 
- [ ] Download profile NetCDF files
- [ ] Parse measurement data
- [ ] Validate data quality

### **Phase 3: Database Design**
- [ ] Create 2D vector tables for spatial-temporal
- [ ] Create 3D vector tables for volumetric analysis
- [ ] Design indexes for fast queries

### **Phase 4: Data Population**
- [ ] Transform NetCDF to database format
- [ ] Populate measurement tables
- [ ] Create vector embeddings

### **Phase 5: Integration**
- [ ] Connect to FloatChat RAG system
- [ ] Enable real-time queries
- [ ] Implement caching strategy

---

## ⚠️ Important Considerations

### **Data Volume**
- **Single float**: ~100-200 profiles over 4-5 years  
- **Each profile**: 50-200 measurement levels
- **Indian Ocean**: ~500-1000 active floats
- **Total records**: ~10-50 million measurements

### **Update Frequency**
- **Real-time data**: Updated every 10 days (float cycle)
- **Delayed mode**: Updated quarterly with QC
- **New floats**: Deployed continuously

### **Quality Control**
- Use only QC flag 1 (good) and 2 (probably good)
- Reject QC flags 3, 4, 8, 9 (bad/missing data)
- Prefer delayed mode (D) over real-time (R)

---

## 📋 Next Steps for Implementation

1. **Create ARGO data fetcher script** based on this analysis
2. **Design proper database schema** for 2D/3D storage
3. **Implement NetCDF parsing** with quality control
4. **Build data pipeline** for continuous updates  
5. **Integrate with RAG system** for intelligent queries

This analysis provides the foundation for building a proper, real-data FloatChat system without any fake/dummy data.