[![Red Hat Instructlab](https://img.shields.io/badge/Redhat-Instructlab-purple)](https://instructlab.ai/)
[![watsonx.ai](https://img.shields.io/badge/IBM-watsonx.ai-blue)](https://dataplatform.cloud.ibm.com/wx/home?context=wx)
[![Docker Pulls](https://img.shields.io/docker/pulls/aseelert/watsonxai-endpoint)](https://hub.docker.com/r/aseelert/watsonxai-endpoint)

# FastAPI watsonx-Compatible Gateway

This repository hosts a **FastAPI** application that forwards requests in the OpenAI-style `/v1/completions` format to **IBM watsonx.ai**. The gateway allows users to send requests using a format compatible with OpenAI but interacts with **watsonx.ai** models instead.

---

## 🌐 What is FastAPI?

**FastAPI** is a modern web framework designed for speed and ease of use when building APIs. It automatically generates interactive documentation using OpenAPI and is known for its high performance in production environments. In this project, FastAPI is used to create and manage the watsonx-compatible API.

[Learn more about FastAPI](https://fastapi.tiangolo.com/)

---

## 🔵 What is watsonx.ai?

**watsonx.ai** is IBM's enterprise AI platform designed for building, training, and deploying AI and machine learning models. It integrates with IBM Cloud services, allowing for efficient handling of both generative and custom AI models. This project forwards OpenAI-style API requests to interact with watsonx.ai models for text generation.

[Learn more about watsonx.ai](https://www.ibm.com/watsonx)

---

## 🔴 What is iLab (Red Hat InstructLab)?

**Red Hat InstructLab (iLab)** simplifies the development of large language models (LLMs) by offering tools for data generation and model alignment. It supports fine-tuning of models like LLaMA and IBM Granite using both human-curated and synthetic data, improving model performance without the need for expensive retraining.

[Learn more about Red Hat InstructLab](https://instructlab.ai)

---

## 🟢 What is OpenAPI?

**OpenAPI** is a standard specification that defines the structure of REST APIs in a way that can be understood by both humans and machines. In this project, OpenAPI is used through **FastAPI** to automatically generate API documentation and testing tools, such as Swagger.

[Learn more about OpenAPI](https://swagger.io/specification/)

---

### Clarifications:
- **OpenAI Compatibility**: The API is structured similarly to OpenAI’s completion API but interacts with watsonx.ai models.
- **Swagger/OpenAPI**: FastAPI automatically generates Swagger documentation, accessible via `/docs`, providing interactive API testing and documentation.




## Why Use This?

This gateway provides a bridge between OpenAI’s API format and watsonx.ai, allowing you to:
- **Seamlessly integrate with watsonx.ai** without rewriting existing applications that rely on OpenAI’s API.
- **Ensure compatibility** with legacy codebases that use OpenAI’s completions endpoint.
- **Flexibility**: Leverage watsonx.ai's capabilities with minimal changes to your existing infrastructure.

## Installation

**Clone the repository:**
```bash
git clone https://github.com/aseelert/watsonx-openai-api
cd watsonx-openai-api
```

## Prerequisites

Before running this application, you need to set the following environment variables for both interactive mode and Docker:

- You have an IBM Cloud IAM key (watsonx_IAM_APIKEY) for authentication.
- Your watsonx.ai project has a Watson Machine Learning instance associated, which is required to manage machine learning models.
- You have a Project ID (watsonx_PROJECT_ID) of the watsonx.ai project.


**install python 3.11 venv:**
```bash
sudo dnf -y install python3.11 python3.11-devel jq
python3.11 -m venv venv
source ~/watsonx-openai-api/venv/bin/activate
pip install --upgrade pip
```

**install pip packages:**
```bash
pip install --no-cache-dir fastapi uvicorn requests streamlit
```

### Required Environment Variables

- **`watsonx_IAM_APIKEY`**: The IBM API key required to authenticate with watsonx.ai.
- **`watsonx_PROJECT_ID`**: The Porject ID for watsonx.ai, where the requests will be forwarded.

Start by setting the environment variables:

```bash
export watsonx_IAM_APIKEY="your-ibm-api-key"
export watsonx_PROJECT_ID="your-watsonx-project-id"
```

Ensure these variables are properly set, or the application will fail to start.
An explanation how to set them is given in the respective sections.

## How to Run

<details>
<summary><b>Run local (Interactive mode)</b></summary>




If running interactively, use `uvicorn` to start the FastAPI application after setting the environment variables:

```bash
cd fastapi-watsonx

uvicorn watsonxai-endpoint:app --reload --port 8080
```

</details>

<details>
<summary> <b>Run as docker</b></summary>

If you prefer to run this application in a Docker container, follow these steps:

**1. Execute Docker with hub.docker.com image and IBM Variables**
This will start the application in a container, listening on port 8080, and interacting with watsonx.ai via the provided credentials.

```bash
docker run -d -p 8080:8000 --name watsonxai-endpoint \
-e watsonx_IAM_APIKEY=${watsonx_IAM_APIKEY} \
-e watsonx_PROJECT_ID=${watsonx_PROJECT_ID} \
aseelert/watsonxai-endpoint:1.0
```
</details>

<details>
<summary> <b>Build own Docker image</b></summary>
**1. Build your own local Docker image**

**Project Version**
```bash
cd fastapi-watsonx
docker build -t watsonxai-endpoint:1.0 .
```

**2. Execute Docker with local image and IBM Variables**

For Docker, pass the environment variables with the `-e` flag:

```bash
docker run -d -p 8080:8000 --name watsonxai-endpoint \
-e watsonx_IAM_APIKEY=${watsonx_IAM_APIKEY} \
-e watsonx_PROJECT_ID=${watsonx_PROJECT_ID} \
watsonxai-endpoint:1.0
```



**4. Activate live logs**

```bash
docker logs -f watsonxai-endpoint
```

</details>

## How to use
### Use with Curl

After starting the application, you can test it with a curl command:

```bash
curl http://127.0.0.1:8080/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "prompt": "Explain watsonx.ai advantages.",
  "max_tokens": 50,
  "temperature": 0.7
}'|jq
```


### Use with Redhat instructLab

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

### **Parameter Description:**

- **`--pipeline full`**: Runs the entire data generation pipeline.
- **`--sdg-scale-factor 100`**: Scales synthetic data generation by a factor of 100.
- **`--endpoint-url http://localhost:8080/v1`**: Specifies the API endpoint for watsonx.ai.
- **`--output-dir ./outputdir-watsonxai-endpoint`**: Directory where the generated data will be saved.
- **`--chunk-word-count 1000`**: Splits the data into chunks of 1000 words.
- **`--num-cpus 8`**: Uses 8 CPU cores for processing.
- **`--model mistralai/mistral-large`**: Specifies the model to be used for data generation.



## API Reference

This gateway mimics the OpenAI `/v1/completions` endpoint while forwarding requests to IBM watsonx.ai. It supports typical parameters like `prompt`, `max_tokens`, and `temperature`, while ensuring compatibility with the watsonx.ai text generation model.

---

This gateway ensures compatibility between legacy applications and watsonx.ai, providing a flexible solution for text generation services.
