from fastapi import FastAPI, Request, HTTPException
import requests
import os
import time
import uuid
import logging
import json

app = FastAPI()

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Mapping of each region to URL
WATSONX_URLS = {
    "us-south": "https://us-south.ml.cloud.ibm.com",
    "eu-gb": "https://eu-gb.ml.cloud.ibm.com",
    "jp-tok": "https://jp-tok.ml.cloud.ibm.com",
    "eu-de": "https://eu-de.ml.cloud.ibm.com"
}

# Get region from env variable
region = os.getenv("WATSONX_REGION")

# If no region var is set, ask user to input it
if not region:
    print("Please select a region:")
    for idx, reg in enumerate(WATSONX_URLS.keys(), start=1):
        print(f"{idx}. {reg}")
    
    # Get user input and validate
    choice = input("Enter the number corresponding to your region: ")
    
    try:
        region = list(WATSONX_URLS.keys())[int(choice) - 1]
    except (IndexError, ValueError):
        raise ValueError("Invalid region selection. Please restart and select a valid option.")


# IBM Cloud IAM URL for fetching the token
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# Load IBM API key, Watsonx URL, and Project ID from environment variables
IBM_API_KEY = os.getenv("WATSONX_IAM_APIKEY")
WATSONX_URL = f"{WATSONX_URLS.get(region)}/ml/v1/text/generation?version=2023-05-29"
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

if not IBM_API_KEY:
    logger.error("IBM API key is not set. Please set the WATSONX_IAM_APIKEY environment variable.")
    raise SystemExit("IBM API key is required.")

if not PROJECT_ID:
    logger.error("Watsonx.ai project ID is not set. Please set the WATSONX_PROJECT_ID environment variable.")
    raise SystemExit("Watsonx.ai project ID is required.")

# Global variables to cache the IAM token and expiration time
cached_token = None
token_expiration = 0

# Function to fetch the IAM token
def get_iam_token():
    global cached_token, token_expiration
    current_time = time.time()

    if cached_token and current_time < token_expiration:
        logger.debug("Using cached IAM token.")
        return cached_token

    logger.debug("Fetching new IAM token from IBM Cloud...")

    try:
        response = requests.post(
            IAM_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": IBM_API_KEY,
            },
        )
        response.raise_for_status()
        token_data = response.json()
        cached_token = token_data["access_token"]
        expires_in = token_data["expires_in"]

        token_expiration = current_time + expires_in - 600
        logger.debug(f"IAM token fetched, expires in {expires_in} seconds.")

        return cached_token
    except requests.exceptions.RequestException as err:
        logger.error(f"Error fetching IAM token: {err}")
        raise HTTPException(status_code=500, detail=f"Error fetching IAM token: {err}")

# Route to handle Watsonx-compatible requests
@app.post("/v1/completions")
async def watsonx_completions(request: Request):
    logger.info("Received a Watsonx completion request.")

    request_data = await request.json()

    # Extract parameters from request
    prompt = request_data.get("prompt", "")
    max_tokens = request_data.get("max_tokens", 2000)
    temperature = request_data.get("temperature", 0.2)
    model_id = request_data.get("model", "mistralai/mistral-large")  # Default model_id if not provided
    presence_penalty = request_data.get("presence_penalty", 1)

    logger.debug(f"Prompt: {prompt[:200]}..., Max Tokens: {max_tokens}, Temperature: {temperature}, Model ID: {model_id}")

    # Get IBM IAM token
    iam_token = get_iam_token()

    # Prepare Watsonx.ai request payload with HAP disabled
    watsonx_payload = {
        "input": prompt,
        "parameters": {
            "decoding_method": "sample",
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 50,
            "top_p": 1,
            "repetition_penalty": presence_penalty
        },
        "model_id": model_id,
        "project_id": PROJECT_ID
    }

    logger.debug(f"Sending request to Watsonx.ai: {json.dumps(watsonx_payload, indent=4)}")

    headers = {
        "Authorization": f"Bearer {iam_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        response = requests.post(WATSONX_URL, json=watsonx_payload, headers=headers)
        response.raise_for_status()
        watsonx_data = response.json()
        logger.debug(f"Received response from Watsonx.ai: {json.dumps(watsonx_data, indent=4)}")
    except requests.exceptions.RequestException as err:
        logger.error(f"Error calling Watsonx.ai: {err}")
        raise HTTPException(status_code=500, detail=f"Error calling Watsonx.ai: {err}")

    # Extract the generated text from the Watsonx response
    results = watsonx_data.get("results", [])
    if results and "generated_text" in results[0]:
        generated_text = results[0]["generated_text"]
        logger.debug(f"Generated text from Watsonx.ai: {generated_text}")
    else:
        generated_text = "\n\nNo response available."
        logger.warning("No generated text found in Watsonx.ai response.")

    # Prepare the OpenAI-compatible response with model name "llama-3-405b-instruct"
    openai_response = {
        "id": f"cmpl-{str(uuid.uuid4())[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": "llama-3-405b-instruct",  # or whatever model name you'd prefer
        "system_fingerprint": f"fp_{str(uuid.uuid4())[:12]}",
        "choices": [
            {
                "text": generated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": results[0].get("stop_reason", "length")
            }
        ],
        "usage": {
            "prompt_tokens": results[0].get("input_token_count", 5),
            "completion_tokens": results[0].get("generated_token_count", 7),
            "total_tokens": results[0].get("input_token_count", 5) + results[0].get("generated_token_count", 7)
        }
    }

    # Pretty print the final response for better readability
    logger.debug(f"Returning OpenAI-compatible response: {json.dumps(openai_response, indent=4)}")
    return openai_response
