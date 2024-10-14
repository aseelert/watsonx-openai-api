# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the openapi.py file into the container
COPY watsonxai-endpoint.py /app/watsonxai-endpoint.py

# Install any needed packages
RUN pip install --no-cache-dir fastapi uvicorn requests

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Set environment variables for Watsonx API key and URL (to be set at runtime)
ENV WATSONX_IAM_APIKEY=""
ENV WATSONX_PROJECT_ID=""
ENV WATSONX_REGION=""

# Define environment variable for Python to avoid buffering
ENV PYTHONUNBUFFERED=1

# Run the FastAPI app with Uvicorn when the container launches
CMD ["uvicorn", "watsonxai-endpoint:app", "--host", "0.0.0.0", "--port", "8000"]
