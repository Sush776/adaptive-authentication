# Use official Python slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Command to run the app (adjust if your app.py is inside src/)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
