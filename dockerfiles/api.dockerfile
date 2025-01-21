# Base image
FROM python:3.11-slim

# Install necessary system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and application code
COPY requirements.txt .
COPY src/finance/ ./src/finance/
COPY model/ ./model/  

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn --no-cache-dir  # Ensure uvicorn is installed

# Expose the port for FastAPI
EXPOSE 8000

# Run FastAPI using uvicorn
ENTRYPOINT ["uvicorn", "src.finance.api:app", "--host", "0.0.0.0", "--port", "8000"]
