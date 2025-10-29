# Use full Python image for heavy ML packages
FROM python:3.10

# Set working directory inside container
WORKDIR /app

# Install system dependencies for ML packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire application (all folders and files from ai-legal-assistant)
COPY . .

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Run the Flask application
CMD ["python", "app.py"]
