# Dockerfile
# Use a slim Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files
# You'll need to include your config.ini
COPY . .

# Cloud Run requires the container to listen on the PORT environment variable
# Uvicorn para FastAPI async (concurrencia real)
CMD exec uvicorn app_async:api --host 0.0.0.0 --port $PORT --workers 1