# FastAPI Watsonx-Compatible Gateway

This repository hosts a FastAPI application that forwards OpenAI-style `/v1/completions` requests to IBM Watsonx.ai’s text generation service. The gateway allows users to send requests using the legacy OpenAI format while interacting with Watsonx.ai models.


## Why Use This?

This gateway provides a bridge between OpenAI’s API format and Watsonx.ai, allowing you to:
- **Seamlessly integrate with Watsonx.ai** without rewriting existing applications that rely on OpenAI’s API.
- **Ensure compatibility** with legacy codebases that use OpenAI’s completions endpoint.
- **Flexibility**: Leverage Watsonx.ai's capabilities with minimal changes to your existing infrastructure.

## Prerequisites

Before running this application, you need to set the following environment variables for both interactive mode and Docker:

### Required Environment Variables

- **`WATSONX_IAM_APIKEY`**: The IBM API key required to authenticate with Watsonx.ai.
- **`WATSONX_PROJECT_ID`**: The Porject ID for Watsonx.ai, where the requests will be forwarded.


Ensure these variables are properly set, or the application will fail to start.

### Setting Environment Variables

For Docker, pass the environment variables with the `-e` flag:

```bash
docker run -d -p 8080:8000 --name watsonxai-endpoint \
-e WATSONX_IAM_APIKEY="your-ibm-api-key" \
-e WATSONX_PROJECT_ID="your-watsonx-project-id" \
watsonxai-endpoint:1.0
```

For interactive mode, set them in your shell before starting the app:

```bash
export WATSONX_IAM_APIKEY="your-ibm-api-key"
export WATSONX_PROJECT_ID="your-watsonx-project-id"

uvicorn watsonxai-endpoint:app --reload --port 8080
```

## How to Run

### Running Locally

If running interactively, use `uvicorn` to start the FastAPI application after setting the environment variables:

```bash
uvicorn watsonxai-endpoint:app --reload --port 8080
```

#### Validate with Curl

After starting the application, you can test it with a curl command:

```bash
curl http://127.0.0.1:8080/v1/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer <your-IBM-token>" \
-d '{
  "prompt": "Explain Watsonx.ai advantages.",
  "max_tokens": 50,
  "temperature": 0.7
}'
```

### Running in Docker

If you prefer to run this application in a Docker container, follow these steps:

**1. Build the Docker image**

**Project Version**
```bash
docker build -t watsonxai-endpoint:1.0 .
```

**2. Run the Docker container**

```bash
docker run -d -p 8080:8000 --name watsonxai-endpoint \
-e WATSONX_IAM_APIKEY="your-ibm-api-key" \
-e WATSONX_PROJECT_ID="your-watsonx-project-id" \
watsonxai-endpoint:1.0
```

This will start the application in a container, listening on port 8080, and interacting with Watsonx.ai via the provided credentials.

#### Use Redhat instructLab (project version)
```bash
ilab data generate \
--pipeline full \
--sdg-scale-factor 100 \
--endpoint-url http://localhost:8080/v1 \
--output-dir ./outputdir-watsonxai-endpoint \
--chunk-word-count 1000 \
--num-cpus 8 \
--model mistralai/mistral-large
```

## API Reference

This gateway mimics the OpenAI `/v1/completions` endpoint while forwarding requests to IBM Watsonx.ai. It supports typical parameters like `prompt`, `max_tokens`, and `temperature`, while ensuring compatibility with the Watsonx.ai text generation model.

---

This gateway ensures compatibility between legacy applications and Watsonx.ai, providing a flexible solution for text generation services.
