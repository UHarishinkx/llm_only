#!/usr/bin/env python3
"""
Production Web Backend for ARGO RAG System
Converts CLI (interactive_test.py) to Web Interface
Always-on RAG system with concurrent user support
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import threading
import time

# Web framework
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Add parent directories to import your existing RAG system
sys.path.append('../..')
sys.path.append('.')

# Import your existing RAG system
from working_enhanced_rag import WorkingRAGSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for RAG system (always loaded)
class AppState:
    def __init__(self):
        self.rag_system = None
        self.startup_complete = False
        self.connected_clients = set()
        self.lock = threading.Lock()

app_state = AppState()

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    sql: str
    data: List[Dict]
    method: str
    similarity: float
    execution_time: float
    metadata: Dict[str, Any]
    total_records: int

class SystemStatus(BaseModel):
    status: str
    rag_loaded: bool
    chromadb_count: int
    uptime_seconds: float
    connected_users: int

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        app_state.connected_clients.add(id(websocket))
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        app_state.connected_clients.discard(id(websocket))
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()
startup_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle - Load RAG system once and keep it running"""
    logger.info("Starting ARGO RAG Web System...")

    def initialize_rag():
        """Initialize RAG system in background thread (like interactive_test.py)"""
        try:
            # Use your existing API keys
            GROQ_API_KEY = "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"

            logger.info("Loading RAG System (this may take 2-3 minutes)...")

            # Initialize your RAG system (same as interactive_test.py)
            app_state.rag_system = WorkingRAGSystem(GROQ_API_KEY)

            # Setup ChromaDB (same as interactive_test.py)
            try:
                current_count = app_state.rag_system.chroma_manager.collection.count()
                if current_count > 0:
                    logger.info(f"Using existing ChromaDB with {current_count} queries")
                else:
                    logger.info("ChromaDB is empty - setting up...")
                    app_state.rag_system.setup_system()
            except Exception as e:
                logger.info(f"Setting up ChromaDB: {e}")
                app_state.rag_system.setup_system()

            app_state.startup_complete = True
            logger.info("RAG System loaded and ready! (Always-on mode activated)")

        except Exception as e:
            logger.error(f"RAG system initialization failed: {e}")
            app_state.startup_complete = True  # Continue anyway

    # Start RAG initialization in background
    rag_thread = threading.Thread(target=initialize_rag, daemon=True)
    rag_thread.start()

    # Server is ready to accept connections
    logger.info("Web server ready! RAG system loading in background...")

    yield

    logger.info("Shutting down ARGO RAG Web System...")

# FastAPI app with lifecycle management
app = FastAPI(
    title="ARGO RAG Web Interface",
    description="Web interface for your ARGO oceanographic RAG system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ARGO RAG System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; text-align: center; }
            .loading { background: #fff3cd; color: #856404; }
            .ready { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
            .query-section { margin: 20px 0; }
            #queryInput { width: 100%; padding: 12px; font-size: 16px; border: 1px solid #ddd; border-radius: 4px; }
            #queryBtn { padding: 12px 24px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 10px 0; }
            #queryBtn:hover { background: #0056b3; }
            #queryBtn:disabled { background: #ccc; cursor: not-allowed; }
            .results { margin-top: 20px; }
            .result-item { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 4px; border-left: 4px solid #007bff; }
            .sql-code { background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 4px; font-family: monospace; white-space: pre-wrap; margin: 10px 0; }
            .data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
            .data-table th, .data-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            .data-table th { background: #f8f9fa; }
            .metadata { font-size: 12px; color: #6c757d; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>ARGO RAG System - Web Interface</h1>
            <div id="status" class="status loading">Initializing RAG system...</div>

            <div class="query-section">
                <h3>Enter your query:</h3>
                <input type="text" id="queryInput" placeholder="e.g., show me temperature data for each profile" />
                <button id="queryBtn" onclick="submitQuery()" disabled>Submit Query</button>
            </div>

            <div id="results" class="results"></div>
        </div>

        <script>
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            const statusDiv = document.getElementById('status');
            const queryBtn = document.getElementById('queryBtn');
            const queryInput = document.getElementById('queryInput');
            const resultsDiv = document.getElementById('results');

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);

                if (data.type === 'status') {
                    if (data.rag_loaded) {
                        statusDiv.className = 'status ready';
                        statusDiv.innerHTML = `RAG System Ready! ChromaDB: ${data.chromadb_count} queries loaded`;
                        queryBtn.disabled = false;
                    } else {
                        statusDiv.className = 'status loading';
                        statusDiv.innerHTML = 'Loading RAG system...';
                    }
                } else if (data.type === 'query_result') {
                    displayResult(data);
                } else if (data.type === 'error') {
                    displayError(data.message);
                }
            };

            function submitQuery() {
                const query = queryInput.value.trim();
                if (!query) return;

                queryBtn.disabled = true;
                queryBtn.textContent = 'Processing...';
                resultsDiv.innerHTML = '<div class="result-item">Processing your query...</div>';

                ws.send(JSON.stringify({type: 'query', query: query}));
            }

            function displayResult(data) {
                const resultHtml = `
                    <div class="result-item">
                        <h4>Query: "${data.query}"</h4>
                        <div class="metadata">
                            Method: ${data.method} | Similarity: ${data.similarity.toFixed(3)} |
                            Time: ${data.execution_time.toFixed(2)}s | Records: ${data.total_records}
                        </div>

                        <h5>Generated SQL:</h5>
                        <div class="sql-code">${data.sql}</div>

                        <h5>Results (showing first 10):</h5>
                        ${formatDataTable(data.data.slice(0, 10))}

                        ${data.total_records > 10 ? `<p><em>... and ${data.total_records - 10} more records</em></p>` : ''}
                    </div>
                `;

                resultsDiv.innerHTML = resultHtml;
                queryBtn.disabled = false;
                queryBtn.textContent = 'Submit Query';
            }

            function formatDataTable(data) {
                if (!data || data.length === 0) return '<p>No data returned</p>';

                const headers = Object.keys(data[0]);
                let html = '<table class="data-table"><thead><tr>';
                headers.forEach(header => html += `<th>${header}</th>`);
                html += '</tr></thead><tbody>';

                data.forEach(row => {
                    html += '<tr>';
                    headers.forEach(header => {
                        const value = row[header];
                        html += `<td>${value !== null ? value : 'NULL'}</td>`;
                    });
                    html += '</tr>';
                });

                html += '</tbody></table>';
                return html;
            }

            function displayError(message) {
                resultsDiv.innerHTML = `<div class="result-item" style="border-color: #dc3545; background: #f8d7da;">Error: ${message}</div>`;
                queryBtn.disabled = false;
                queryBtn.textContent = 'Submit Query';
            }

            // Allow Enter key to submit
            queryInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !queryBtn.disabled) {
                    submitQuery();
                }
            });

            // Check status on load
            setTimeout(() => {
                fetch('/api/status').then(r => r.json()).then(data => {
                    if (data.rag_loaded) {
                        statusDiv.className = 'status ready';
                        statusDiv.innerHTML = `RAG System Ready! ChromaDB: ${data.chromadb_count} queries loaded`;
                        queryBtn.disabled = false;
                    }
                });
            }, 1000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/status")
async def get_status():
    """Get system status"""
    chromadb_count = 0
    if app_state.rag_system and app_state.startup_complete:
        try:
            chromadb_count = app_state.rag_system.chroma_manager.collection.count()
        except:
            chromadb_count = 0

    return SystemStatus(
        status="ready" if app_state.startup_complete else "loading",
        rag_loaded=app_state.rag_system is not None and app_state.startup_complete,
        chromadb_count=chromadb_count,
        uptime_seconds=time.time() - startup_time,
        connected_users=len(app_state.connected_clients)
    )

@app.post("/api/query")
async def process_query_api(request: QueryRequest):
    """REST API for query processing"""
    if not app_state.rag_system or not app_state.startup_complete:
        raise HTTPException(status_code=503, detail="RAG system not ready yet")

    try:
        # Process query (same as interactive_test.py)
        start_time = time.time()
        result = app_state.rag_system.process_query(request.query)

        # Execute SQL and get data
        data, success = app_state.rag_system.execute_query(result.enhanced_sql)

        if not success:
            raise HTTPException(status_code=400, detail="SQL execution failed")

        execution_time = time.time() - start_time

        return QueryResponse(
            query=request.query,
            sql=result.enhanced_sql,
            data=data,
            method=result.method,
            similarity=result.similarity,
            execution_time=execution_time,
            metadata=result.metadata,
            total_records=len(data)
        )

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time communication"""
    await manager.connect(websocket)

    try:
        # Send initial status
        status = await get_status()
        await websocket.send_text(json.dumps({
            "type": "status",
            "rag_loaded": status.rag_loaded,
            "chromadb_count": status.chromadb_count
        }))

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "query":
                query = message.get("query", "").strip()
                if not query:
                    continue

                if not app_state.rag_system or not app_state.startup_complete:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "RAG system not ready yet"
                    }))
                    continue

                try:
                    # Process query (same logic as interactive_test.py)
                    start_time = time.time()
                    result = app_state.rag_system.process_query(query)
                    data, success = app_state.rag_system.execute_query(result.enhanced_sql)
                    execution_time = time.time() - start_time

                    if success:
                        await websocket.send_text(json.dumps({
                            "type": "query_result",
                            "query": query,
                            "sql": result.enhanced_sql,
                            "data": data,
                            "method": result.method,
                            "similarity": result.similarity,
                            "execution_time": execution_time,
                            "metadata": result.metadata,
                            "total_records": len(data)
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "SQL execution failed"
                        }))

                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))

    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )