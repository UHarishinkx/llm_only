#!/usr/bin/env python3
"""
New Web Interface using the new ChromaDB RAG System
- Uses new_comprehensive_rag_system.py
- ChromaDB semantic similarity search
- LLM decides if plotting is needed
- Calls visualization functions when needed
"""

import os
import sys
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import webbrowser
import traceback

# Import the new comprehensive RAG system
try:
    from new_comprehensive_rag_system import ComprehensiveRAGSystem
except ImportError as e:
    print(f"Error importing ComprehensiveRAGSystem: {e}")
    print("Make sure new_comprehensive_rag_system.py is in the same directory")
    sys.exit(1)

class NewRAGWebHandler(BaseHTTPRequestHandler):
    """
    Handles HTTP requests for the ARGO RAG System web interface.
    """

    def do_GET(self):
        """Handles GET requests."""
        if self.path == '/':
            self.serve_main_page()
        elif self.path == '/api/status':
            self.serve_status()
        elif self.path.startswith('/api/query'):
            self.handle_query_get()
        else:
            self.send_error(404)

    def do_POST(self):
        """Handles POST requests."""
        if self.path == '/api/query':
            self.handle_query_post()
        else:
            self.send_error(404)

    def serve_main_page(self):
        """Serves the main HTML page."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ARGO RAG System - New ChromaDB</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
                .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; }
                .status { padding: 10px; margin: 10px 0; border-radius: 5px; text-align: center; }
                .loading { background: #fff3cd; color: #856404; }
                .ready { background: #d4edda; color: #155724; }
                .error { background: #f8d7da; color: #721c24; }
                .query-section { margin: 20px 0; }
                #queryInput { width: 100%; padding: 12px; font-size: 16px; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }
                #queryBtn { padding: 12px 24px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 10px 0; }
                #queryBtn:hover { background: #0056b3; }
                #queryBtn:disabled { background: #ccc; cursor: not-allowed; }
                .results { margin-top: 20px; }
                .result-item { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 4px; border-left: 4px solid #007bff; }
                .sql-code { background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 4px; font-family: monospace; white-space: pre-wrap; margin: 10px 0; }
                .data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                .data-table th, .data-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .data-table th { background: #f8f9fa; }
                .pagination-controls { margin: 10px 0; text-align: center; }
                #showMoreBtn { padding: 8px 16px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; }
                #showMoreBtn:hover { background: #218838; }
                #showMoreBtn:disabled { background: #ccc; cursor: not-allowed; }
                .metadata { font-size: 12px; color: #6c757d; margin-top: 10px; }
                .plot-section { background: #e8f4f8; padding: 15px; border-radius: 4px; margin: 10px 0; border-left: 4px solid #17a2b8; }
                .llm-response { background: #f1f8ff; padding: 15px; border-radius: 4px; margin: 10px 0; border-left: 4px solid #0066cc; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ARGO RAG System - Enhanced with New ChromaDB</h1>
                <div id="status" class="status loading">Initializing new RAG system...</div>
                <div class="query-section">
                    <h3>Enter your query:</h3>
                    <input type="text" id="queryInput" placeholder="e.g., show temperature anomalies, plot temperature vs depth, create heatmap" />
                    <button id="queryBtn" onclick="submitQuery()" disabled>Submit Query</button>
                </div>
                <div id="results"></div>
            </div>
            <script>
                function checkStatus() {
                    fetch('/api/status')
                        .then(r => r.json())
                        .then(data => {
                            const statusDiv = document.getElementById('status');
                            const queryBtn = document.getElementById('queryBtn');
                            if (data.rag_loaded) {
                                statusDiv.className = 'status ready';
                                statusDiv.innerHTML = 'RAG System Ready! ChromaDB: ' + data.chromadb_count + ' samples loaded | Embedding Model: ' + data.model_name;
                                queryBtn.disabled = false;
                            } else {
                                statusDiv.className = 'status loading';
                                statusDiv.innerHTML = 'Loading new RAG system with ChromaDB...';
                                setTimeout(checkStatus, 2000);
                            }
                        })
                        .catch(e => {
                            document.getElementById('status').innerHTML = 'Checking status...';
                            setTimeout(checkStatus, 2000);
                        });
                }

                function submitQuery() {
                    const query = document.getElementById('queryInput').value;
                    const resultsDiv = document.getElementById('results');
                    const queryBtn = document.getElementById('queryBtn');

                    if (!query.trim()) return;

                    queryBtn.disabled = true;
                    queryBtn.textContent = 'Processing...';
                    resultsDiv.innerHTML = '<div class="status loading">Processing your query with semantic search and LLM...</div>';

                    fetch('/api/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: query})
                    })
                    .then(r => r.json())
                    .then(data => {
                        if (data.error) {
                            resultsDiv.innerHTML = '<div class="status error">Error: ' + data.error + '</div>';
                        } else {
                            displayResults(data);
                        }
                        queryBtn.disabled = false;
                        queryBtn.textContent = 'Submit Query';
                    })
                    .catch(e => {
                        resultsDiv.innerHTML = '<div class="status error">Request failed: ' + e + '</div>';
                        queryBtn.disabled = false;
                        queryBtn.textContent = 'Submit Query';
                    });
                }

                function displayResults(data) {
                    let html = '<div class="results">';
                    html += '<div class="result-item">';
                    html += '<h4>Query: "' + data.query + '"</h4>';
                    html += '<div class="metadata">';
                    html += 'Best Match: ' + data.best_match_id + ' | Similarity: ' + data.similarity.toFixed(3) + ' | ';
                    html += 'Processing Time: ' + data.processing_time.toFixed(2) + 's';
                    html += '</div>';

                    // Show matched sample info
                    if (data.matched_sample) {
                        html += '<h5>Matched Sample:</h5>';
                        html += '<div class="sql-code">' + JSON.stringify(data.matched_sample, null, 2) + '</div>';
                    }

                    // Show LLM response
                    if (data.llm_response) {
                        html += '<div class="llm-response">';
                        html += '<h5>LLM Analysis:</h5>';
                        html += '<p>' + data.llm_response + '</p>';
                        html += '</div>';
                    }

                    // Show SQL execution
                    if (data.sql_executed) {
                        html += '<h5>Generated SQL:</h5>';
                        html += '<div class="sql-code">' + data.sql + '</div>';

                        if (data.sql_data && data.sql_data.length > 0) {
                            html += '<h5>SQL Results (' + data.sql_data.length + ' total rows):</h5>';
                            html += '<div id="paginationContainer" data-page="0" data-total="' + data.sql_data.length + '">';
                            html += formatDataTable(data.sql_data.slice(0, 10));
                            if (data.sql_data.length > 10) {
                                html += '<div class="pagination-controls">';
                                html += '<button onclick="showMore()" id="showMoreBtn">Show More (10 more of ' + (data.sql_data.length - 10) + ' remaining)</button>';
                                html += '</div>';
                            }
                            html += '</div>';
                            // Store data globally for pagination
                            window.currentSqlData = data.sql_data;
                        } else if (data.sql_executed) {
                            html += '<p>SQL executed but no data returned</p>';
                        }
                    }

                    // Show interactive visualization if created
                    if (data.visualization_created && data.plot_html) {
                        html += '<div class="plot-section">';
                        html += '<h5>Interactive Visualization:</h5>';
                        html += '<div style="width:100%; overflow-x:auto;">' + data.plot_html + '</div>';
                        html += '</div>';
                    }

                    html += '</div>';
                    html += '</div>';
                    document.getElementById('results').innerHTML = html;
                }

                function formatDataTable(data) {
                    if (!data || data.length === 0) return '<p>No data returned</p>';

                    const headers = Object.keys(data[0]);
                    let html = '<table class="data-table"><thead><tr>';
                    headers.forEach(header => html += '<th>' + header + '</th>');
                    html += '</tr></thead><tbody>';

                    data.forEach(row => {
                        html += '<tr>';
                        headers.forEach(header => {
                            const value = row[header];
                            html += '<td>' + (value !== null ? value : 'NULL') + '</td>';
                        });
                        html += '</tr>';
                    });

                    html += '</tbody></table>';
                    return html;
                }

                // Allow Enter key
                document.getElementById('queryInput').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter' && !document.getElementById('queryBtn').disabled) {
                        submitQuery();
                    }
                });

                function showMore() {
                    const container = document.getElementById('paginationContainer');
                    const currentPage = parseInt(container.dataset.page);
                    const total = parseInt(container.dataset.total);
                    const pageSize = 10;
                    const nextPage = currentPage + 1;
                    const startIndex = nextPage * pageSize;
                    const endIndex = Math.min(startIndex + pageSize, total);

                    if (startIndex >= total) return;

                    // Add new rows to existing table
                    const newData = window.currentSqlData.slice(startIndex, endIndex);
                    const tableBody = document.querySelector('.data-table tbody');

                    newData.forEach(row => {
                        const tr = document.createElement('tr');
                        Object.keys(row).forEach(key => {
                            const td = document.createElement('td');
                            td.textContent = row[key] !== null ? row[key] : 'NULL';
                            tr.appendChild(td);
                        });
                        tableBody.appendChild(tr);
                    });

                    // Update pagination state
                    container.dataset.page = nextPage;
                    const remaining = total - endIndex;
                    const showMoreBtn = document.getElementById('showMoreBtn');

                    if (remaining > 0) {
                        showMoreBtn.textContent = `Show More (10 more of ${remaining} remaining)`;
                    } else {
                        showMoreBtn.textContent = 'All data loaded';
                        showMoreBtn.disabled = true;
                    }
                }

                // Start status check
                checkStatus();
            </script>
        </body>
        </html>
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def serve_status(self):
        """Serves the status of the RAG system."""
        status = {
            "rag_loaded": hasattr(server_state, 'rag_system') and server_state.rag_system is not None,
            "chromadb_count": 0,
            "model_name": "loading...",
            "status": "loading"
        }
        if hasattr(server_state, 'rag_system') and server_state.rag_system:
            try:
                if hasattr(server_state.rag_system, 'collection') and server_state.rag_system.collection:
                    status["chromadb_count"] = server_state.rag_system.collection.count()
                if hasattr(server_state.rag_system, 'embedding_model_name'):
                    status["model_name"] = server_state.rag_system.embedding_model_name
                status["status"] = "ready"
            except Exception as e:
                print(f"Error getting status: {e}")
                pass

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

    def handle_query_get(self):
        # Handle GET requests to /api/query if needed
        self.send_error(405, "Method not allowed. Use POST for queries.")

    def handle_query_post(self):
        """Handles the query from the user."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            query = data.get('query', '').strip()

            if not query:
                self.send_error(400, "No query provided")
                return

            if not hasattr(server_state, 'rag_system') or not server_state.rag_system:
                response = {"error": "RAG system not ready"}
            else:
                try:
                    start_time = time.time()
                    # Use the new comprehensive query method
                    result = server_state.rag_system.simple_query(query)
                    processing_time = time.time() - start_time

                    response = {
                        "query": query,
                        "processing_time": processing_time,
                        "best_match_id": result.get("best_match_id", "unknown"),
                        "similarity": result.get("similarity", 0.0),
                        "matched_sample": result.get("matched_sample", {}),
                        "llm_response": result.get("llm_response", ""),
                        "sql_executed": result.get("sql_executed", False),
                        "sql": result.get("sql", ""),
                        "sql_data": result.get("sql_data", []),
                        "visualization_created": result.get("visualization_created", False),
                        "plot_type": result.get("plot_type", ""),
                        "plot_html": result.get("plot_html", "")
                    }
                except Exception as e:
                    response = {"error": f"Processing failed: {str(e)}", "traceback": traceback.format_exc()}

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            error_response = {"error": f"Request handling failed: {str(e)}"}
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())

class ServerState:
    """
    A simple class to hold the state of the server.
    """
    def __init__(self):
        self.rag_system = None

server_state = ServerState()

def initialize_rag():
    """Initialize the new comprehensive RAG system"""
    try:
        print("Initializing New Comprehensive RAG System...")
        print("This will load the new ChromaDB with all semantic samples...")

        GROQ_API_KEY = "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"

        # Create the new RAG system
        server_state.rag_system = ComprehensiveRAGSystem(GROQ_API_KEY)

        # Load all semantic samples into ChromaDB
        print("Loading semantic samples into ChromaDB...")
        server_state.rag_system.load_all_semantic_samples()

        # Create ChromaDB collection (only if not exists)
        try:
            if (hasattr(server_state.rag_system, 'collection') and 
                server_state.rag_system.collection and 
                server_state.rag_system.collection.count() > 0):
                print(f"Using existing ChromaDB with {server_state.rag_system.collection.count()} samples")
            else:
                print("Creating ChromaDB collection...")
                server_state.rag_system.create_new_chromadb()
        except Exception as e:
            print(f"Error with ChromaDB collection, creating new one: {e}")
            server_state.rag_system.create_new_chromadb()

        print("New RAG System ready with enhanced capabilities!")
        if hasattr(server_state.rag_system, 'collection') and server_state.rag_system.collection:
            print(f"- ChromaDB samples: {server_state.rag_system.collection.count()}")
        if hasattr(server_state.rag_system, 'embedding_model_name'):
            print(f"- Embedding model: {server_state.rag_system.embedding_model_name}")
        print("- LLM integration: Groq API")
        print("- SQL execution: Ready")
        print("- Visualization: Ready")

    except Exception as e:
        print(f"RAG initialization failed: {e}")
        traceback.print_exc()
        server_state.rag_system = None

def main():
    """
    Starts the web server.
    """
    print("Starting ARGO RAG New Web Interface...")
    print("This uses the new ChromaDB system with semantic search and LLM integration")

    # Start RAG initialization in background
    rag_thread = threading.Thread(target=initialize_rag, daemon=True)
    rag_thread.start()

    # Start web server
    port = 8001  # Different port to avoid conflicts
    try:
        server = HTTPServer(('localhost', port), NewRAGWebHandler)
    except OSError as e:
        print(f"Error starting server on port {port}: {e}")
        print("Try a different port or check if the port is already in use")
        return

    print(f"Server running at http://localhost:{port}")
    print("Opening browser...")

    # Open browser after a short delay
    def open_browser():
        time.sleep(1)
        try:
            webbrowser.open(f'http://localhost:{port}')
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please manually open: http://localhost:{port}")

    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()

if __name__ == "__main__":
    main()