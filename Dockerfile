FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install dependencies
# Note: In a real prod environment, we'd pin versions strictly
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Start command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.api.app:app"]
