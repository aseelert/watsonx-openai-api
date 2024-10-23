[![Red Hat Instructlab](https://img.shields.io/badge/Redhat-Instructlab-purple)](https://instructlab.ai/)
[![watsonx.ai](https://img.shields.io/badge/IBM-watsonx.ai-blue)](https://dataplatform.cloud.ibm.com/wx/home?context=wx)
[![Docker Version](https://img.shields.io/docker/v/aseelert/watsonxai-endpoint)](https://hub.docker.com/r/aseelert/watsonxai-endpoint)
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

- You have an IBM Cloud IAM key **`(WATSONX_IAM_APIKEY)`** for authentication.
- You have a Project ID **`(WATSONX_PROJECT_ID)`** of the watsonx.ai project.
- You have a Server Regions **`(WATSONX_REGION)`** of the watsonx.ai project. (supported: ["`us-south`", "`eu-gb`", "`jp-tok`", "`eu-de`"])
- Your watsonx.ai project has a Watson Machine Learning instance associated, which is required to manage machine learning models.
- If instructlab data generation will be use, we assume ilab is already installed.

### Required Environment Variables

- **`WATSONX_IAM_APIKEY`**: The IBM API key required to authenticate with watsonx.ai.
- **`WATSONX_PROJECT_ID`**: The Project ID for watsonx.ai, where the requests will be forwarded.
- **`WATSONX_REGION`**: The ProjectRegion for watsonx.ai ("us-south", "eu-gb", "jp-tok", "eu-de")

Start by setting the environment variables:

```bash
export WATSONX_IAM_APIKEY="your-ibm-api-key"
export WATSONX_PROJECT_ID="your-watsonx-project-id"
export WATSONX_REGION="your-watsonx-project-region"
```

Ensure these variables are properly set, or the application will fail to start.
An explanation how to set them is given in the respective sections.

## How to Run

<details>
<summary><b>Run local (Interactive mode)</b></summary>

If running interactively, use `uvicorn` to start the FastAPI application after setting the environment variables:

**install python 3.11 venv:**
```bash
sudo dnf -y install python3.11 python3.11-devel jq
python3.11 -m venv venv
source ~/watsonx-openai-api/venv/bin/activate
pip install --upgrade pip
```

**install pip packages:**
```bash
pip install --no-cache-dir fastapi uvicorn requests streamlit tabulate
```

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
-e WATSONX_IAM_APIKEY=${WATSONX_IAM_APIKEY} \
-e WATSONX_PROJECT_ID=${WATSONX_PROJECT_ID} \
-e WATSONX_REGION=${WATSONX_REGION} \
aseelert/watsonxai-endpoint:1.1
```

**2. Activate live logs**

```bash
docker logs -f watsonxai-endpoint
```
</details>

<details>
<summary> <b>Build own Docker image</b></summary>

**1. Build a local docker image**
```bash
cd fastapi-watsonx
docker build -t watsonxai-endpoint:1.1 .
```

**2. Execute Docker with local image and IBM Variables**

For Docker, pass the environment variables with the `-e` flag:

```bash
docker run -d -p 8080:8000 --name watsonxai-endpoint \
-e WATSONX_IAM_APIKEY=${WATSONX_IAM_APIKEY} \
-e WATSONX_PROJECT_ID=${WATSONX_PROJECT_ID} \
-e WATSONX_REGION=${WATSONX_REGION} \
watsonxai-endpoint:1.1
```

**3. Activate live logs**

```bash
docker logs -f watsonxai-endpoint
```

</details>




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
--model ibm/granite-20b-multilingual
```


<details>
<summary> <b>Example output</b></summary>

```json
INFO:     172.17.0.1:44028 - "POST /v1/completions HTTP/1.1" 200 OK
2024-10-02 08:17:35 - INFO - watsonxai-endpoint - Received a Watsonx completion request.
2024-10-02 08:17:35 - DEBUG - watsonxai-endpoint - Prompt: '<|system|>
You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and ..., Max Tokens: 2048, Temperature: 0.7, Model ID: meta-llama/llama-3-405b-instruct
2024-10-02 08:17:35 - DEBUG - watsonxai-endpoint - Using cached IAM token.
2024-10-02 08:17:35 - DEBUG - watsonxai-endpoint - Sending request to Watsonx.ai: {
    "input": "'<|system|>\nYou are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.\n<|user|>\nYou are a very knowledgeable AI Assistant that will faithfully assist the user with their task.\nYou are asked to come up with a diverse context for - This skill provides the ability to summarize transcripts\n.\nPlease follow these guiding principles when generating responses:\n* Use proper grammar and punctuation.\n* Always generate safe and respectful content. Do not generate content that is harmful, abusive, or offensive.\n* Always generate content that is factually accurate and relevant to the prompt.\n* Strictly adhere to the prompt and generate responses in the same style and format as the example.\n* Return the context between [Start of Context] and [End of Context] tags.\n\nTo better assist you with this task, here is an example of a context:\n[Start of Context]\nSara: (Dialing customer care) Hello, this is Sara, and I'm having some issues with my broadband connection. The internet has been quite slow, and I've been experiencing frequent disconnections.\\n\\nMike: Hi Sara, I'm Mike, a customer care agent. I'm sorry to hear about the trouble you're facing with your broadband. Let me check that for you. Can you please provide me with your account number or the phone number associated with your account?\\n\\nSara: Sure, it's 204-555-1234.\\n\\nMike: Thank you, Sara. Let me pull up your account. While I'm doing that, can you tell me when you first started noticing these issues?\\n\\nSara: It started about a week ago. The internet speed has been inconsistent, and there are times when it just goes out completely.\\n\\nMike: I understand how frustrating that can be. I appreciate your patience. It looks like there might be some signal issues. Have you tried restarting your modem and router?\\n\\nSara: Yes, I've tried that a couple of times, but the problems persist.\\n\\nMike: Alright, thanks for trying that. I'll run a diagnostic on your connection now. While that's happening, could you let me know if there are specific times of the day when you notice these problems more frequently?\\n\\nSara: It seems to be worse during the evenings, especially when I'm trying to stream videos or have video calls.\\n\\nMike: Got it. It could be related to network congestion during peak hours. Let me check the signal strength in your area. While I'm doing that, have you noticed if your neighbors are experiencing similar issues?\\n\\nSara: I haven't had a chance to check with them, but I can do that. Hold on a moment.\\n\\n(Mike puts Sara on a brief hold while he checks the network status)\\n\\nMike: Thank you for waiting, Sara. It appears there might be an issue in your area affecting multiple customers. Our technicians are already working to resolve it. I apologize for the inconvenience.\\n\\nSara: Oh, okay. I appreciate the update. Do you have an estimated time for when it will be fixed?\\n\\nMike: I don't have an exact time, but our team is actively working on it, and we'll strive to resolve it as soon as possible. In the meantime, if you experience any further issues or if there are updates, feel free to reach out to us.\\n\\nSara: Alright, thank you, Mike. I hope it gets resolved soon.\\n\\nMike: You're welcome, Sara. I understand the importance of a reliable internet connection. If you have any other questions or concerns, please don't hesitate to contact us.\\n\\nSara: I will. Thanks for your help.\\n\\nMike: Have a great day, Sara!\n[End of Context]\n\nNow generate a context paragraph, remember to follow the principles mentioned above and use the same format as the examples. Remember to use the same style and format as the example above. Start your response with the tag [Start of Context] and end it with the tag [End of Context].\n<|assistant|>\n'",
    "parameters": {
        "decoding_method": "sample",
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 1,
        "repetition_penalty": 1
    },
    "model_id": "meta-llama/llama-3-405b-instruct",
    "project_id": "311cd3b7-876d-4028-b271-2469a433867f"
}
2024-10-02 08:17:35 - DEBUG - urllib3.connectionpool - Starting new HTTPS connection (1): us-south.ml.cloud.ibm.com:443
```
</details>

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

### Appendix
Get the current list of LLMs of Saas watsonx.ai instance

```bash
curl -X GET http://localhost:8080/v1/models | jq
```

```json
{
  "data": [
    {
      "id": "bigscience/mt0-xxl",
      "object": "model",
      "created": 1729690580,
      "owned_by": "BigScience / Hugging Face",
      "description": "An instruction-tuned iteration on mT5. Supports tasks like question_answering, summarization, classification, generation.",
      "max_tokens": 4095,
      "token_limits": {
        "max_sequence_length": 4096,
        "max_output_tokens": 4095
      }
    },
    {
      "id": "codellama/codellama-34b-instruct-hf",
      "object": "model",
      "created": 1729690580,
      "owned_by": "Code Llama / Hugging Face",
      "description": "Code Llama is an AI model built on top of Llama 2, fine-tuned for generating and discussing code. Supports tasks like code.",
      "max_tokens": 8192,
      "token_limits": {
        "max_sequence_length": 16384,
        "max_output_tokens": 8192
      }
    },
    {
      "id": "google/flan-t5-xl",
      "object": "model",
      "created": 1729690580,
      "owned_by": "Google / Hugging Face",
      "description": "A pretrained T5 - an encoder-decoder model pre-trained on a mixture of supervised / unsupervised tasks converted into a text-to-text format. Supports tasks like question_answering, summarization, retrieval_augmented_generation, classification, generation, extraction.",
      "max_tokens": 4095,
      "token_limits": {
        "max_sequence_length": 4096,
        "max_output_tokens": 4095
      }
    },
    {
      "id": "google/flan-t5-xxl",
      "object": "model",
      "created": 1729690580,
      "owned_by": "Google / Hugging Face",
      "description": "flan-t5-xxl is an 11 billion parameter model based on the Flan-T5 family. Supports tasks like question_answering, summarization, retrieval_augmented_generation, classification, generation, extraction.",
      "max_tokens": 4095,
      "token_limits": {
        "max_sequence_length": 4096,
        "max_output_tokens": 4095
      }
    },
......
```
