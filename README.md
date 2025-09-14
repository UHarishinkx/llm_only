

## Gener
### Your Task: Initiate the Workflow

1. read the setup.md

2. install and open the gemini cli in clone project folder

3. choose a query type from the 12 given query type and give okay for everything it will do everything


thats all you need to do and verify it is working at the it will open a web and test the query type you choose
































**Step 1: Define the Optimal JSON Structure**

*   Based on the mission category, select one of the following pre-defined optimal JSON structures.

    *   **For "Geographic/Spatial":**
        ```json
        {
          "id": "string", "content": "string",
          "metadata": { "category": "Geographic/Spatial", "spatial_operation": "string", "geo_parameters": {"center_lat": "float", "center_lon": "float", "radius_km": "float"}, "sql_template": "string", "visualization_config": { "primary": "interactive_map" } }
        }
        ```
    *   **For "Temporal":**
        ```json
        {
          "id": "string", "content": "string",
          "metadata": { "category": "Temporal", "time_operation": "string", "time_parameters": {"start_date": "string", "end_date": "string"}, "sql_template": "string", "visualization_config": { "primary": "timeline_chart" } }
        }
        ```
    *   **For "Statistical and Analytical":**
        ```json
        {
          "id": "string", "content": "string",
          "metadata": { "category": "Statistical and Analytical", "analysis_type": "string", "analysis_parameters": {"target_variable": "string", "group_by_columns": "array"}, "sql_template": "string", "visualization_config": { "primary": "bar_chart" } }
        }
        ```

**Step 2: Generate the JSON Prompts**

*   Generate a JSON object with a "queries" list of 10 prompts using the optimal structure from Step 1.
*   Use the database schema below to write the `sql_template`.

**DATABASE SCHEMA:**
*   **`floats` table:** `float_id`, `wmo_number`, `program_name`, `platform_type`, `data_assembly_center`, `deployment_date`, `deployment_latitude`, `deployment_longitude`, `current_status`, `last_latitude`, `last_longitude`, `last_update`, `total_profiles`.
*   **`profiles` table:** `profile_id`, `float_id`, `cycle_number`, `profile_date`, `latitude`, `longitude`, `max_pressure`, `num_levels`, `data_mode`, `data_quality_flag`.
*   **`measurements` table:** `measurement_id`, `profile_id`, `pressure`, `depth`, `temperature`, `salinity`, `pressure_qc`, `temperature_qc`, `salinity_qc`.
*   **JOINs:** `floats.float_id` -> `profiles.float_id`, `profiles.profile_id` -> `measurements.profile_id`.

**Step 3: Create the JSON File**

*   Use the `write_file` tool to create a new JSON file at `semantic_samples/[category_name].json` with the content from Step 2.

**Step 4: Generate the Python Testing Script**

*   Generate a `test_batch.py` script to test the prompts you just created, including at least 20 variations.

**Step 5: Analyze and Refine**

*   "Imagine" running the test script. If any prompts have an average similarity score below 0.85, generate a new, improved version of the JSON object.

**Step 6: Update the JSON File**

*   If refinements were made, use the `write_file` tool to overwrite the JSON file with the improved content.

---

The system is composed of the following key components:

*   **`new_comprehensive_rag_system.py`:** The core RAG engine. It handles semantic search, LLM interaction, SQL execution, and visualization.
*   **`new_web_interface.py`:** A web interface that uses the RAG system to provide a user-friendly UI.
*   **ChromaDB:** A vector database used to store and search semantic representations of sample queries.
*   **DuckDB:** An in-process SQL OLAP database management system used to query the Parquet data.
*   **Groq API:** The service providing the Large Language Model for generating SQL and providing natural language responses.
*   **`parquet_data/`:** Directory containing the ARGO data in Parquet format.
*   **`semantic_samples/`:** Directory containing JSON files with sample queries to populate the ChromaDB.

## Setup and Installation

1.  **Python:** This project requires Python 3.13.7. You can check your Python version by running `python --version`.

    If you don't have Python 3.13.7, you can download it from the [official Python website](https://www.python.org/downloads/release/python-3137/).

    For Windows users, it is recommended to use the installer from the website. For macOS and Linux users, you can use a version manager like `pyenv` to install and manage different Python versions.

2.  **Install Dependencies:** Install the required Python packages using the `requirements_new_system.txt` file:
    ```bash
    pip install -r requirements_new_system.txt
    ```


    ```

## Running the Project

You can run the project in two ways:

### 1. Web Interface

To start the web interface, run the `new_web_interface.py` script:

```bash
python new_web_interface.py
```

This will start a web server on `http://localhost:8001`. The first time you run it, it will initialize the ChromaDB with the data from the `semantic_samples` directory.

### 2. Interactive CLI

To use the interactive command-line interface, run the `new_comprehensive_rag_system.py` script:

```bash
python new_comprehensive_rag_system.py
```

This will also initialize the ChromaDB and then drop you into an interactive shell where you can type your queries.

## Deployment

This project is configured for deployment on [Railway](https://railway.app/). The `railway.json`, `railway.toml`, and `Dockerfile` (not present in the file listing, but referenced) are used for this purpose. The start command for the deployed service is `python main.py`, which refers to `web/backend/main.py`.

## Data

*   **`parquet_data/`**: This directory contains the core ARGO data, split into three Parquet files:
    *   `floats.parquet`: Information about each float.
    *   `profiles.parquet`: Data for each measurement profile.
    *   `measurements.parquet`: Individual sensor measurements.
*   **`semantic_samples/`**: This directory contains JSON files with sample queries. These are used to populate the ChromaDB with a variety of questions, which the system uses to find the best way to answer a user's query.

## Code Structure

```
.
├── new_comprehensive_rag_system.py # Core RAG engine
├── new_web_interface.py            # Web interface
├── web/
│   └── backend/
│       ├── main.py                 # Production web backend for deployment
│       └── requirements.txt
├── parquet_data/                   # ARGO data in Parquet format
├── semantic_samples/               # Sample queries for ChromaDB
├── new_comprehensive_chromadb/     # ChromaDB database
├── requirements_new_system.txt     # Python dependencies
├── railway.json                    # Railway deployment configuration
├── railway.toml                    # Railway deployment configuration
└── README.md                       # This file
```
