#!/usr/bin/env python3
"""
NEW COMPREHENSIVE RAG SYSTEM
- Loads all semantic_samples JSON files into new ChromaDB
- Best embedding model for semantic similarity
- Groq API LLM integration
- SQL execution on parquet data
- Visualization generation
"""

import json
import chromadb
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from pathlib import Path
import os
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
import duckdb
from typing import Dict, List, Tuple, Any

class ComprehensiveRAGSystem:
    """Complete RAG system with ChromaDB, Groq API, and visualization"""

    def __init__(self, groq_api_key: str = None):
        self.embedding_model_name = "https://tfhub.dev/google/universal-sentence-encoder/4"
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = hub.load(self.embedding_model_name)

        # Initialize ChromaDB in new directory
        self.chroma_path = "./new_comprehensive_chromadb"
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection_name = "comprehensive_argo_rag"
        self.collection = None

        # Initialize Groq API
        self.groq_client = None
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)

        # Data paths
        self.semantic_samples_dir = "semantic_samples"
        self.parquet_data_dir = "parquet_data"

        # Loaded samples
        self.all_samples = []

        # SQL execution setup
        self.db_connection = self.setup_sql_execution()

    def setup_sql_execution(self):
        """Setup SQL execution environment for parquet data using DuckDB"""
        try:
            con = duckdb.connect(database=':memory:', read_only=False)
            
            # Register parquet files as virtual tables
            for name, path in {
                'profiles': f"{self.parquet_data_dir}/profiles.parquet",
                'measurements': f"{self.parquet_data_dir}/measurements.parquet",
                'floats': f"{self.parquet_data_dir}/floats.parquet"
            }.items():
                if Path(path).exists():
                    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM parquet_scan('{path}')")
                    print(f"Registered '{name}' view from {path}")
                else:
                    print(f"WARNING: {path} not found")
            
            return con
        except Exception as e:
            print(f"Error setting up DuckDB: {e}")
            return None

    def load_all_semantic_samples(self):
        """Load all JSON files from semantic_samples directory"""
        samples_dir = Path(self.semantic_samples_dir)
        json_files = list(samples_dir.glob("*.json"))

        print(f"Found {len(json_files)} JSON files in {self.semantic_samples_dir}/")

        all_samples = []
        file_stats = {}

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract samples from different JSON structures
                samples = []
                if 'queries' in data:
                    samples = data['queries']
                elif 'samples' in data:
                    samples = data['samples']
                elif isinstance(data, list):
                    samples = data

                # Add source file info to each sample
                for sample in samples:
                    if isinstance(sample, dict):
                        sample['source_file'] = json_file.name
                        all_samples.append(sample)

                file_stats[json_file.name] = len(samples)
                print(f"  - {json_file.name}: {len(samples)} samples")

            except Exception as e:
                print(f"ERROR loading {json_file}: {e}")

        self.all_samples = all_samples
        print(f"\nTotal loaded samples: {len(all_samples)}")

        # Show category distribution
        categories = {}
        for sample in all_samples:
            if 'metadata' in sample and 'category' in sample['metadata']:
                cat = sample['metadata']['category']
                categories[cat] = categories.get(cat, 0) + 1

        if categories:
            print("\nCategory distribution:")
            for cat, count in categories.items():
                print(f"  - {cat}: {count} samples")

        return all_samples

    def create_new_chromadb(self):
        """Create new ChromaDB collection with all samples"""
        print(f"\nCreating new ChromaDB collection: {self.collection_name}")

        # Delete existing collection if it exists
        try:
            self.client.delete_collection(self.collection_name)
            print("Deleted existing collection")
        except:
            pass

        # Create new collection
        self.collection = self.client.create_collection(self.collection_name)

        # Prepare data for ChromaDB
        documents = []
        ids = []
        metadatas = []

        for i, sample in enumerate(self.all_samples):
            # Create unique ID
            sample_id = sample.get('id', f"sample_{i}")
            if sample.get('source_file'):
                sample_id = f"{sample['source_file'].replace('.json', '')}_{sample_id}"

            # Get content
            content = sample.get('content', '')
            if not content:
                # Try alternative content fields
                content = sample.get('query', sample.get('text', str(sample)))

            # Prepare metadata (ChromaDB only allows str, int, float, bool, None)
            metadata = {}
            if 'metadata' in sample:
                for key, value in sample['metadata'].items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        metadata[key] = value
                    elif isinstance(value, dict):
                        # Convert dict to JSON string
                        metadata[key] = json.dumps(value)
                    else:
                        metadata[key] = str(value)
            metadata['source_file'] = sample.get('source_file', 'unknown')

            documents.append(content)
            ids.append(sample_id)
            metadatas.append(metadata)

        # Generate embeddings
        print("Generating embeddings for all documents...")
        embeddings = self.embedding_model(documents)
        print("Embeddings generated.")

        # Add to ChromaDB in batches
        batch_size = 100
        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_embeds = embeddings[i:i+batch_size]

            self.collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metas,
                embeddings=batch_embeds.numpy().tolist()
            )
            total_added += len(batch_docs)
            print(f"Added batch {i//batch_size + 1}: {total_added}/{len(documents)} samples")

        print(f"\nSuccessfully created ChromaDB with {total_added} samples")
        return self.collection

    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Perform semantic search using ChromaDB"""
        if not self.collection:
            print("ERROR: ChromaDB collection not initialized")
            return []

        start_time = time.time()

        query_embedding = self.embedding_model([query])

        results = self.collection.query(
            query_embeddings=query_embedding.numpy().tolist(),
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

        search_time = time.time() - start_time

        # Process results
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                similarity = 1 - results['distances'][0][i]

                result = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': similarity,
                    'search_time': search_time
                }
                search_results.append(result)

        return search_results

    def execute_sql_query(self, sql_template: str, parameters: Dict = None) -> pd.DataFrame:
        """Execute SQL query on parquet data using DuckDB"""
        if not self.db_connection:
            print("ERROR: DuckDB connection not available")
            return pd.DataFrame()

        try:
            print(f"Executing SQL template: {sql_template[:200]}...")
            print(f"With parameters: {parameters}")
            # Replace parameters in SQL template
            sql_query = sql_template
            if parameters:
                for param, value in parameters.items():
                    placeholder = "{" + param + "}"
                    sql_query = sql_query.replace(placeholder, str(value))

            print(f"Executing SQL: {sql_query[:200]}...")

            # Execute query using DuckDB
            result_df = self.db_connection.execute(sql_query).fetchdf()

            print(f"Query returned {len(result_df)} rows")
            return result_df

        except Exception as e:
            print(f"SQL execution error: {e}")
            return pd.DataFrame()

    def generate_visualization(self, data: pd.DataFrame, plot_config: Dict) -> str:
        """Generate visualization based on plot configuration"""
        if data.empty:
            return "No data to visualize"

        try:
            plt.figure(figsize=(10, 6))

            plot_type = plot_config.get('type', 'line_plot')
            title = plot_config.get('title', 'Data Visualization')

            if plot_type == 'line_plot':
                if len(data.columns) >= 2:
                    plt.plot(data.iloc[:, 0], data.iloc[:, 1])
                    plt.xlabel(plot_config.get('x_axis', data.columns[0]))
                    plt.ylabel(plot_config.get('y_axis', data.columns[1]))

            elif plot_type == 'scatter':
                if len(data.columns) >= 2:
                    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], alpha=0.6)
                    plt.xlabel(plot_config.get('x_axis', data.columns[0]))
                    plt.ylabel(plot_config.get('y_axis', data.columns[1]))

            elif plot_type == 'histogram':
                if len(data.columns) >= 1:
                    plt.hist(data.iloc[:, 0], bins=30, alpha=0.7)
                    plt.xlabel(plot_config.get('x_axis', data.columns[0]))
                    plt.ylabel('Frequency')

            elif plot_type == 'heatmap':
                if len(data.columns) >= 3:
                    # Pivot data for heatmap if needed
                    pivot_data = data.pivot_table(
                        index=data.columns[0],
                        columns=data.columns[1],
                        values=data.columns[2],
                        aggfunc='mean'
                    )
                    sns.heatmap(pivot_data, cmap='viridis', cbar=True)

            plt.title(title)
            plt.tight_layout()

            # Save plot
            plot_filename = f"plot_{int(time.time())}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            return f"Visualization saved as: {plot_filename}"

        except Exception as e:
            plt.close()
            return f"Visualization error: {e}"

    def query_groq_llm(self, prompt: str, context: str = "") -> str:
        """Query Groq API for LLM response"""
        if not self.groq_client:
            return "Groq API not configured. Please provide API key."

        try:
            full_prompt = f"""
You are an expert oceanographic data analyst. Use the following context to answer the user's question.

Context: {context}

User Question: {prompt}

Please provide a comprehensive answer based on the context provided. If the context contains SQL templates or visualization instructions, incorporate them into your response.
"""

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt,
                    }
                ],
                model="llama-3.1-8b-instant",  # Fast and capable model
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            return f"Groq API error: {e}"

    def comprehensive_query(self, user_query: str, execute_sql: bool = True,
                          generate_viz: bool = True) -> Dict:
        """Complete query processing pipeline"""
        print(f"\nProcessing query: '{user_query}'")
        print("=" * 60)

        # Step 1: Semantic search
        print("1. Performing semantic search...")
        search_results = self.semantic_search(user_query, n_results=3)

        if not search_results:
            return {"error": "No search results found"}

        best_match = search_results[0]
        print(f"   Best match: {best_match['id']} (similarity: {best_match['similarity']:.3f})")

        # Step 2: Extract SQL and plot config
        sql_template = best_match['metadata'].get('sql_template', '')
        plot_config = best_match['metadata'].get('plot_config', {})
        parameters = best_match['metadata'].get('parameters', {})

        result = {
            'query': user_query,
            'best_match': best_match,
            'sql_template': sql_template,
            'plot_config': plot_config,
            'parameters': parameters
        }

        # Step 3: Execute SQL if requested and available
        sql_data = pd.DataFrame()
        if execute_sql and sql_template:
            print("2. Executing SQL query...")
            # Use default parameters for demo
            default_params = {}
            if isinstance(parameters, dict):
                for param, param_info in parameters.items():
                    if 'default' in param_info:
                        if 'integer' in param_info:
                            default_params[param] = 100
                        elif 'float' in param_info:
                            default_params[param] = 50.0
                        elif 'date' in param_info:
                            default_params[param] = '2020-01-01'
                        else:
                            default_params[param] = 'profile_001'

            sql_data = self.execute_sql_query(sql_template, default_params)
            result['sql_data'] = sql_data

        # Step 4: Generate visualization if requested
        if generate_viz and not sql_data.empty and plot_config:
            print("3. Generating visualization...")
            viz_result = self.generate_visualization(sql_data, plot_config)
            result['visualization'] = viz_result

        # Step 5: Generate LLM response
        print("4. Generating LLM response...")
        context = f"Best match: {best_match['content']}\n"
        if not sql_data.empty:
            context += f"Data summary: {len(sql_data)} rows returned\n"
        if plot_config:
            context += f"Visualization type: {plot_config.get('type', 'unknown')}\n"

        llm_response = self.query_groq_llm(user_query, context)
        result['llm_response'] = llm_response

        return result

    def simple_query(self, user_query: str) -> Dict:
        """Simplified query method for web interface with intelligent SQL generation"""
        result = {
            "best_match_id": "",
            "similarity": 0.0,
            "matched_sample": {},
            "llm_response": "",
            "sql_executed": False,
            "sql": "",
            "sql_data": [],
            "visualization_created": False,
            "plot_type": "",
            "plot_html": ""
        }

        try:
            # Step 1: Semantic search
            search_results = self.semantic_search(user_query, n_results=3)
            if not search_results:
                result["llm_response"] = "No similar queries found in the database."
                return result

            best_match = search_results[0]
            result["best_match_id"] = best_match['id']
            result["similarity"] = best_match['similarity']

            # Try to parse JSON content, fallback to raw content
            try:
                result["matched_sample"] = json.loads(best_match['content'])
            except (json.JSONDecodeError, TypeError):
                result["matched_sample"] = {"raw_content": best_match['content'], "metadata": best_match.get('metadata', {})}

            # Step 2: Check if LLM thinks plotting is needed
            plot_decision = self.llm_decide_plotting(user_query, best_match['content'])

            # Step 3: Generate SQL intelligently using LLM
            sql_generated = self.generate_intelligent_sql(user_query, search_results)

            if sql_generated["success"]:
                try:
                    sql_data = self.execute_sql_query(sql_generated["sql"])
                    result["sql_executed"] = True
                    result["sql"] = sql_generated["sql"]
                    result["sql_data"] = sql_data.to_dict('records') if not sql_data.empty else []

                    # If no data, try a fallback query
                    if len(result["sql_data"]) == 0:
                        fallback_sql = self.generate_fallback_sql(user_query)
                        if fallback_sql:
                            fallback_data = self.execute_sql_query(fallback_sql)
                            if not fallback_data.empty:
                                result["sql"] = f"Original returned 0 rows. Using fallback: {fallback_sql}"
                                result["sql_data"] = fallback_data.to_dict('records')
                            else:
                                result["sql"] = f"No data found. Fallback also returned 0 rows."

                except Exception as e:
                    result["sql_executed"] = False
                    result["sql"] = f"SQL execution failed: {str(e)}"
            else:
                result["sql"] = f"SQL generation failed: {sql_generated['error']}"

            # Step 4: Create visualization if needed
            if plot_decision["needs_plotting"] and result["sql_executed"] and result["sql_data"]:
                try:
                    plot_info = self.create_visualization(
                        result["sql_data"],
                        plot_decision["plot_type"],
                        user_query
                    )
                    result["visualization_created"] = plot_info.get("success", False)
                    result["plot_type"] = plot_info["type"]
                    result["plot_html"] = plot_info.get("html", "")
                except Exception as e:
                    result["plot_html"] = f"<p>Visualization failed: {str(e)}</p>"

            # Step 5: Generate LLM response
            context = f"Query: {user_query}\n"
            context += f"Best match: {best_match['id']} (similarity: {best_match['similarity']:.3f})\n"
            if result["sql_executed"]:
                context += f"SQL executed successfully, returned {len(result['sql_data'])} rows\n"
            if result["visualization_created"]:
                context += f"Visualization created: {result['plot_type']}\n"

            result["llm_response"] = self.query_groq_llm(user_query, context)

        except Exception as e:
            result["llm_response"] = f"Error processing query: {str(e)}"

        return result

    def llm_decide_plotting(self, user_query: str, matched_content: str) -> Dict:
        """Use LLM to decide if plotting is needed and what type"""
        if not self.groq_client:
            return {"needs_plotting": False, "plot_type": "none"}

        prompt = f"""
        Analyze this user query and determine if a visualization/plot is needed:

        User Query: "{user_query}"
        Matched Sample: {matched_content}

        Respond with JSON format:
        {{
            "needs_plotting": true/false,
            "plot_type": "line|scatter|heatmap|histogram|none",
            "reason": "brief explanation"
        }}

        Plot types:
        - line: for time series, depth profiles
        - scatter: for correlations, relationships
        - heatmap: for geographic data, temperature distributions
        - histogram: for distributions, frequency analysis
        - none: for simple data queries

        Look for keywords like: plot, graph, chart, heatmap, visualization, show, display
        """

        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1
            )

            response_text = response.choices[0].message.content.strip()

            # Try to parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: simple keyword detection
                plot_keywords = ['plot', 'graph', 'chart', 'heatmap', 'visualization', 'show', 'display']
                needs_plot = any(keyword in user_query.lower() for keyword in plot_keywords)
                return {
                    "needs_plotting": needs_plot,
                    "plot_type": "line" if needs_plot else "none",
                    "reason": "keyword detection fallback"
                }
        except Exception as e:
            # Fallback to simple keyword detection
            plot_keywords = ['plot', 'graph', 'chart', 'heatmap', 'visualization', 'show', 'display']
            needs_plot = any(keyword in user_query.lower() for keyword in plot_keywords)
            return {
                "needs_plotting": needs_plot,
                "plot_type": "line" if needs_plot else "none",
                "reason": f"LLM failed, used keyword detection: {str(e)}"
            }

    def create_visualization(self, data: List[Dict], plot_type: str, query: str) -> Dict:
        """Create interactive visualization from data using Plotly"""
        if not data:
            return {"type": "none", "html": "<p>No data to plot</p>"}

        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.offline import plot

            df = pd.DataFrame(data)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) < 1:
                return {"type": "error", "html": "<p>No numeric data found for plotting</p>"}

            fig = None

            # Temperature vs Pressure plot (oceanographic standard)
            if 'temperature' in df.columns and 'pressure' in df.columns:
                # Filter out invalid data (zeros and nulls)
                df_clean = df[(df['temperature'] > 0) & (df['pressure'] > 0)].copy()

                if len(df_clean) == 0:
                    return {"type": "error", "html": "<p>No valid temperature/pressure data found (all values are zero or null)</p>"}

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_clean['temperature'],
                    y=df_clean['pressure'],
                    mode='markers+lines',
                    name='Temperature Profile',
                    marker=dict(size=6, color='red', opacity=0.7),
                    line=dict(width=2, color='red'),
                    hovertemplate='<b>Temperature</b>: %{x:.2f}°C<br><b>Pressure</b>: %{y:.1f} dbar<extra></extra>'
                ))
                fig.update_layout(
                    title=f"Temperature vs Pressure Profile<br><sub>{query}</sub><br><span style='font-size:12px'>({len(df_clean)} data points)</span>",
                    xaxis_title="Temperature (°C)",
                    yaxis_title="Pressure (dbar) - Depth increases downward",
                    yaxis=dict(autorange="reversed"),  # Invert Y-axis (deeper = higher pressure)
                    template="plotly_white",
                    height=500
                )

            # Temperature vs Salinity scatter plot
            elif 'temperature' in df.columns and 'salinity' in df.columns:
                fig = px.scatter(
                    df,
                    x='temperature',
                    y='salinity',
                    title=f"Temperature vs Salinity Scatter Plot<br><sub>{query}</sub>",
                    labels={"temperature": "Temperature (°C)", "salinity": "Salinity (PSU)"},
                    template="plotly_white",
                    height=500
                )
                fig.update_traces(marker=dict(size=8, opacity=0.7))

            # Generic scatter plot for any two numeric columns
            elif plot_type == "scatter" and len(numeric_cols) >= 2:
                fig = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}<br><sub>{query}</sub>",
                    template="plotly_white",
                    height=500
                )

            # Line plot
            elif plot_type == "line" and len(numeric_cols) >= 1:
                if len(numeric_cols) >= 2:
                    fig = px.line(
                        df,
                        x=numeric_cols[0],
                        y=numeric_cols[1],
                        title=f"Line Plot: {numeric_cols[0]} vs {numeric_cols[1]}<br><sub>{query}</sub>",
                        template="plotly_white",
                        height=500
                    )
                else:
                    fig = px.line(
                        df,
                        y=numeric_cols[0],
                        title=f"Line Plot: {numeric_cols[0]}<br><sub>{query}</sub>",
                        template="plotly_white",
                        height=500
                    )

            # Histogram
            elif plot_type == "histogram":
                fig = px.histogram(
                    df,
                    x=numeric_cols[0],
                    title=f"Distribution: {numeric_cols[0]}<br><sub>{query}</sub>",
                    template="plotly_white",
                    height=500
                )

            # Default: line plot of first numeric column
            else:
                fig = px.line(
                    df,
                    y=numeric_cols[0],
                    title=f"Data Plot: {numeric_cols[0]}<br><sub>{query}</sub>",
                    template="plotly_white",
                    height=500
                )

            if fig:
                # Convert to HTML with embedded JavaScript
                html_plot = plot(fig, output_type='div', include_plotlyjs='cdn')
                return {"type": plot_type, "html": html_plot, "success": True}
            else:
                return {"type": "error", "html": "<p>Could not create plot with available data</p>"}

        except ImportError:
            # Fallback to matplotlib if plotly not available
            return self.create_matplotlib_fallback(data, plot_type, query)
        except Exception as e:
            return {"type": "error", "html": f"<p>Plotting failed: {str(e)}</p>"}

    def create_matplotlib_fallback(self, data: List[Dict], plot_type: str, query: str) -> Dict:
        """Fallback to matplotlib if plotly is not available"""
        try:
            import base64
            from io import BytesIO

            df = pd.DataFrame(data)
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            plt.figure(figsize=(10, 6))

            if 'temperature' in df.columns and 'pressure' in df.columns:
                plt.plot(df['temperature'], df['pressure'], 'ro-', markersize=4, linewidth=2)
                plt.xlabel("Temperature (°C)")
                plt.ylabel("Pressure (dbar)")
                plt.gca().invert_yaxis()  # Invert Y-axis for oceanographic convention
            elif len(numeric_cols) >= 2:
                plt.plot(df[numeric_cols[0]], df[numeric_cols[1]], 'bo-', markersize=4)
                plt.xlabel(numeric_cols[0])
                plt.ylabel(numeric_cols[1])
            elif len(numeric_cols) >= 1:
                plt.plot(df[numeric_cols[0]], 'b-', linewidth=2)
                plt.ylabel(numeric_cols[0])
                plt.xlabel("Index")

            plt.title(f"ARGO Data Visualization\n{query}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Convert to base64 for embedding in HTML
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()

            html_img = f'<img src="data:image/png;base64,{image_base64}" style="width:100%; max-width:800px;" alt="ARGO Data Plot">'
            return {"type": plot_type, "html": html_img, "success": True}

        except Exception as e:
            return {"type": "error", "html": f"<p>Fallback plotting failed: {str(e)}</p>"}

    def generate_intelligent_sql(self, user_query: str, search_results: List[Dict]) -> Dict:
        """Generate SQL using LLM based on user query and search results"""
        if not self.groq_client:
            return {"success": False, "error": "LLM not available"}

        # Get database schema information
        schema_info = self.get_database_schema()

        # Prepare context from search results
        context_samples = []
        for result in search_results[:3]:  # Use top 3 matches
            try:
                sample = json.loads(result['content'])
                if 'metadata' in result and 'sql_template' in result['metadata']:
                    context_samples.append({
                        "id": result['id'],
                        "similarity": result['similarity'],
                        "sql_template": result['metadata']['sql_template']
                    })
            except:
                pass

        prompt = f"""
        Generate a SQL query for ARGO oceanographic data based on the user's request.

        User Query: "{user_query}"

        Database Schema:
        {schema_info}

        Similar examples from database:
        {json.dumps(context_samples, indent=2)}

        Instructions:
        1. Create a SQL query that answers the user's question
        2. Use proper table joins (profiles JOIN measurements ON profiles.profile_id = measurements.profile_id)
        3. Add appropriate WHERE clauses and LIMIT to avoid massive results
        4. For temperature vs depth queries, use temperature and pressure columns
        5. Include QC filters (temperature_qc IN ('1', '2') for good data)
        6. IMPORTANT: Filter out invalid data - exclude NULL values and zeros
        7. For temperature: WHERE temperature IS NOT NULL AND temperature > 0
        8. For pressure: WHERE pressure IS NOT NULL AND pressure > 0
        9. Return only valid SQL without explanation

        SQL Query:
        """

        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1
            )

            sql_query = response.choices[0].message.content.strip()

            # Clean up the response - remove markdown formatting if present
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].split("```")[0].strip()

            # Basic validation
            if not sql_query.upper().startswith('SELECT'):
                return {"success": False, "error": "Generated query is not a SELECT statement"}

            return {"success": True, "sql": sql_query}

        except Exception as e:
            return {"success": False, "error": f"LLM SQL generation failed: {str(e)}"}

    def generate_fallback_sql(self, user_query: str) -> str:
        """Generate simple fallback SQL queries for common cases"""
        query_lower = user_query.lower()

        # Temperature vs salinity scatter plot
        if "temperature" in query_lower and "salinity" in query_lower and "scatter" in query_lower:
            return """
            SELECT m.temperature, m.salinity, p.profile_date, p.latitude, p.longitude
            FROM profiles p
            JOIN measurements m ON p.profile_id = m.profile_id
            WHERE m.temperature IS NOT NULL
            AND m.salinity IS NOT NULL
            AND m.temperature_qc IN ('1', '2')
            AND m.salinity_qc IN ('1', '2')
            ORDER BY p.profile_date DESC
            LIMIT 1000
            """

        # Temperature vs depth/pressure queries
        if "temperature" in query_lower and ("depth" in query_lower or "pressure" in query_lower):
            # Check if specific float_id is mentioned
            import re
            float_id_match = re.search(r'\b(\d{7})\b', user_query)
            if float_id_match:
                float_id = float_id_match.group(1)
                return f"""
                SELECT m.temperature, m.pressure
                FROM profiles p
                JOIN measurements m ON p.profile_id = m.profile_id
                WHERE p.float_id = '{float_id}'
                AND m.temperature IS NOT NULL AND m.temperature > 0
                AND m.pressure IS NOT NULL AND m.pressure > 0
                AND m.temperature_qc IN ('1', '2')
                ORDER BY m.pressure
                LIMIT 1000
                """
            else:
                return """
                SELECT m.temperature, m.pressure
                FROM profiles p
                JOIN measurements m ON p.profile_id = m.profile_id
                WHERE m.temperature IS NOT NULL AND m.temperature > 0
                AND m.pressure IS NOT NULL AND m.pressure > 0
                AND m.temperature_qc IN ('1', '2')
                ORDER BY m.pressure
                LIMIT 1000
                """

        # Float information queries
        if "float" in query_lower and any(word in query_lower for word in ["info", "about", "details"]):
            return "SELECT * FROM floats LIMIT 10"

        # Profile queries
        if "profile" in query_lower:
            return """
            SELECT p.profile_id, p.profile_date, p.latitude, p.longitude,
                   COUNT(m.measurement_id) as measurement_count
            FROM profiles p
            LEFT JOIN measurements m ON p.profile_id = m.profile_id
            GROUP BY p.profile_id
            ORDER BY p.profile_date DESC
            LIMIT 50
            """

        # General data queries
        if any(word in query_lower for word in ["data", "measurements", "show"]):
            return """
            SELECT p.profile_date, p.latitude, p.longitude,
                   m.temperature, m.salinity, m.pressure
            FROM profiles p
            JOIN measurements m ON p.profile_id = m.profile_id
            WHERE m.temperature IS NOT NULL
            AND m.temperature_qc IN ('1', '2')
            ORDER BY p.profile_date DESC
            LIMIT 200
            """

        return None

    def get_database_schema(self) -> str:
        """Get database schema information for LLM context"""
        schema_info = """
        Tables:
        1. profiles: profile_id, float_id, profile_date, latitude, longitude
        2. measurements: measurement_id, profile_id, pressure, temperature, salinity, temperature_qc, salinity_qc
        3. floats: float_id, wmo_number, deployment_date, deployment_latitude, deployment_longitude

        Common Query Patterns:
        - Join profiles and measurements on profile_id
        - Use temperature_qc IN ('1', '2') for good quality data
        - pressure represents depth (higher pressure = deeper)
        - Always use LIMIT to prevent large result sets
        """
        return schema_info

    

    def interactive_mode(self):
        """Interactive mode for querying the system"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE RAG SYSTEM - INTERACTIVE MODE")
        print("=" * 80)
        print("Commands:")
        print("  - Enter any oceanographic query")
        print("  - 'stats' - Show system statistics")
        print("  - 'test [query]' - Test semantic search only")
        print("  - 'exit' - Quit")
        print()

        while True:
            user_input = input("Enter your query: ").strip()

            if user_input.lower() == 'exit':
                print("Goodbye!")
                break

            elif user_input.lower() == 'stats':
                print(f"System Statistics:")
                print(f"  - Total samples in ChromaDB: {len(self.all_samples)}")
                print(f"  - Parquet files loaded: {len(self.dataframes)}")
                print(f"  - Groq API configured: {'Yes' if self.groq_client else 'No'}")
                continue

            elif user_input.lower().startswith('test '):
                query = user_input[5:].strip()
                if query:
                    results = self.semantic_search(query, n_results=3)
                    print(f"\nTop 3 matches for '{query}':")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. {result['id']} (similarity: {result['similarity']:.3f})")
                continue

            elif user_input == '':
                continue

            # Full comprehensive query
            result = self.comprehensive_query(user_input)

            print(f"\nRESULTS:")
            print(f"Best match: {result['best_match']['id']}")
            print(f"Similarity: {result['best_match']['similarity']:.3f}")

            if 'sql_data' in result and not result['sql_data'].empty:
                print(f"SQL data: {len(result['sql_data'])} rows")
                print(result['sql_data'].head())

            if 'visualization' in result:
                print(f"Visualization: {result['visualization']}")

            if 'llm_response' in result:
                print(f"\nLLM Response:")
                print(result['llm_response'])

            print("\n" + "-" * 60)

def main():
    """Initialize and run the comprehensive RAG system"""
    from dotenv import load_dotenv
    load_dotenv()
    print("Comprehensive RAG System Initialization")
    print("=" * 60)

    # Get Groq API key from environment or user
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        print("WARNING: GROQ_API_KEY not found in environment variables")
        print("LLM responses will be disabled")
    else:
        print(f"Found GROQ_API_KEY: {groq_api_key[:5]}...")

    # Initialize system
    rag_system = ComprehensiveRAGSystem(groq_api_key)

    # Load all semantic samples
    samples = rag_system.load_all_semantic_samples()

    # Create new ChromaDB
    collection = rag_system.create_new_chromadb()

    print(f"\nSystem ready!")
    print(f"ChromaDB path: {rag_system.chroma_path}")
    print(f"Total samples: {len(samples)}")

    # Start interactive mode
    rag_system.interactive_mode()

if __name__ == "__main__":
    main()