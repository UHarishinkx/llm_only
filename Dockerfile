# Production Dockerfile for ARGO RAG Web System
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY web/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your existing RAG system files
COPY working_enhanced_rag.py .
COPY working_enhanced_chroma_db/ ./working_enhanced_chroma_db/
COPY parquet_data/ ./parquet_data/
COPY optimized_chromadb_data.json .

# Copy web backend
COPY web/backend/main.py .

# Create necessary directories
RUN mkdir -p logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Start the web server
CMD ["python", "main.py"]