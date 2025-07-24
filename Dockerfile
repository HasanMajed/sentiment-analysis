# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port (change if your app uses a different port)
EXPOSE 8000

# Set PYTHONPATH so 'app' is a module
ENV PYTHONPATH="${PYTHONPATH}:/app/custom-ai-model"

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.best_api:app", "--host", "0.0.0.0", "--port", "8000"] 