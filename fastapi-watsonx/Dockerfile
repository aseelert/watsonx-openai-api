# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

RUN pip install --upgrade pip

# Copy only the required files into the container
COPY requirements.txt ./

# Install any needed packages
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Set environment variables for Watsonx API key and URL (to be set at runtime)
ENV WATSONX_IAM_APIKEY="" \
    WATSONX_PROJECT_ID="" \
    WATSONX_REGION="" \
    WATSONX_VERSION="" \
# Defines if onprem install - set to 1 if on prem
    WATSONX_ON_PREM=0 \
    CPD_URL=""

# Define environment variable for Python to avoid buffering
ENV PYTHONUNBUFFERED=1

COPY watsonxai-endpoint.py /app/watsonxai-endpoint.py

# Run the FastAPI app with Uvicorn when the container launches
CMD ["uvicorn", "watsonxai-endpoint:app", "--host", "0.0.0.0", "--port", "8000"]
