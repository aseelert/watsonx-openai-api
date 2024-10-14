from fastapi import FastAPI, Request, HTTPException
import requests
import os
import time
import uuid
import logging
import json
from tabulate import tabulate

app = FastAPI()

# Function to check if the app is running inside Docker
def is_running_in_docker():
    """Check if the app is running inside Docker by checking for specific environment variables or Docker files."""
    return os.path.exists('/.dockerenv') or os.getenv("DOCKER") == "true"

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Mapping of valid regions
valid_regions = ["us-south", "eu-gb", "jp-tok", "eu-de"]

# Get region from env variable
region = os.getenv("WATSONX_REGION")

# Handle behavior based on environment (Docker vs. Interactive mode)
if not region or region not in valid_regions:
    if is_running_in_docker():
        # In Docker, raise an error if WATSONX_REGION is missing or invalid
        logger.error(f"WATSONX_REGION key is not set or invalid. Supported regions are: {', '.join(valid_regions)}.")
        raise SystemExit(f"WATSONX_REGION is required. Supported regions are: {', '.join(valid_regions)}.")
    else:
        # In interactive mode, prompt the user for the region
        print("Please select a region from the following options:")
        for idx, reg in enumerate(valid_regions, start=1):
            print(f"{idx}. {reg}")
        
        choice = input("Enter the number corresponding to your region: ")
        
        try:
            region = valid_regions[int(choice) - 1]
        except (IndexError, ValueError):
            raise ValueError("Invalid region selection. Please restart and select a valid option.")

# Mapping of each region to URL
WATSONX_URLS = {
    "us-south": "https://us-south.ml.cloud.ibm.com",
    "eu-gb": "https://eu-gb.ml.cloud.ibm.com",
    "jp-tok": "https://jp-tok.ml.cloud.ibm.com",
    "eu-de": "https://eu-de.ml.cloud.ibm.com"
}

# Set the Watsonx URL based on the selected region
WATSONX_URL = f"{WATSONX_URLS.get(region)}/ml/v1/text/generation?version=2023-05-29"


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

def format_debug_output(request_data):
    headers = ["by API", "Parameter", "API Value", "Default Value", "Explanation"]
    table = []

    # Define the parameters excluding "Prompt" and add explanations
    parameters = [
        ("Model ID", request_data.get("model", "llama-3-405b-instruct"), "llama-3-405b-instruct", 
        "ID of the model to use for completion"),
        
        ("Max Tokens", request_data.get("max_tokens", 2000), 2000, 
        "Maximum number of tokens to generate in the completion. The total tokens, prompt + completion."),
        
        ("Temperature", request_data.get("temperature", 0.2), 0.2, 
        "Controls the randomness of the generated output. Higher values make the output more random."),
        
        ("Presence Penalty", request_data.get("presence_penalty", 1), 1, 
        "Penalizes new tokens based on whether they appear in the text so far. Positive values encourage the model to talk about new topics."),
        
        ("Frequency Penalty", request_data.get("frequency_penalty", 1), 1, 
        "Penalizes new tokens based on their frequency in the text so far. Reduces the likelihood of the model repeating the same lines."),
        
        ("top_p", request_data.get("top_p", 1), 1, 
        "Nucleus sampling parameter. For example, top_p = 0.1 means the model will consider only the top 10% probability tokens."),
        
        ("best_of", request_data.get("best_of", 1), 1, 
        "Generates multiple completions server-side, returning the 'best' one (the one with the highest log probability)."),
        
        ("echo", request_data.get("echo", False), False, 
        "If set to True, echoes the prompt back along with the completion. Useful for debugging purposes."),
        
        ("n", request_data.get("n", 1), 1, 
        "Number of completions to generate for each prompt. Note that this can quickly consume your token quota."),
        
        ("seed", request_data.get("seed", None), None, 
        "If specified, ensures deterministic outputs, meaning repeated requests with the same seed and parameters should return the same result."),
        
        ("stop", request_data.get("stop", None), None, 
        "Up to 4 sequences where the model will stop generating further tokens. The generated text will not contain the stop sequence."),
        
        ("logit_bias", request_data.get("logit_bias", None), None, 
        "A JSON object that adjusts the likelihood of specified tokens appearing in the completion. Maps token IDs to a bias value from -100 to 100."),
        
        ("logprobs", request_data.get("logprobs", None), None, 
        "Includes the log probabilities on the logprobs most likely tokens, as well as the chosen tokens. Useful for analyzing the model's decision process."),
        
        ("stream", request_data.get("stream", False), False, 
        "If set to True, streams back partial progress as the model generates tokens in real-time."),
        
        ("suffix", request_data.get("suffix", None), None, 
        "Specifies a suffix that comes after the generated text. Useful for inserting text after a completion.")
    ]


    # ANSI escape codes for colors
    green = "\033[92m"
    yellow = "\033[93m"
    reset = "\033[0m"

    for param, api_value, default_value, explanation in parameters:
        # Determine if the parameter was provided by the API (yellow) or using default (green)
        if api_value != default_value:
            # Use yellow for rows where the API value differs from the default
            color = yellow
            provided_by_api = f"{yellow}X{reset}"
        else:
            # Use green for rows with default values
            color = green
            provided_by_api = ""

        # Append the row with color applied to the entire line
        table.append([
            provided_by_api,  # First column: Provided by API
            f"{color}{param}{reset}", 
            f"{color}\"{api_value}\"{reset}" if isinstance(api_value, str) else f"{color}{api_value}{reset}", 
            f"{color}{default_value}{reset}", 
            f"{color}{explanation}{reset}"
        ])

    # Align the "Provided by API" and "Parameter" columns to the left, as well as "Explanation"
    return tabulate(table, headers, tablefmt="pretty", colalign=("center", "left", "center", "center", "left"))



# Route to handle Watsonx-compatible requests
@app.post("/v1/completions")
async def watsonx_completions(request: Request):
    logger.info("Received a Watsonx completion request.")

    request_data = await request.json()


    # Extract parameters from request, or set default values if not provided
    prompt = request_data.get("prompt", "")
    model_id = request_data.get("model", "llama-3-405b-instruct")  # Default model_id
    max_tokens = request_data.get("max_tokens", 2000)
    temperature = request_data.get("temperature", 0.2)
    best_of = request_data.get("best_of", 1)
    n = request_data.get("n", 1)
    presence_penalty = request_data.get("presence_penalty", 1)
    #frequency_penalty = request_data.get("frequency_penalty", 0)
    echo = request_data.get("echo", False)
    logit_bias = request_data.get("logit_bias", None)
    logprobs = request_data.get("logprobs", None)
    #stop = request_data.get("stop", [])
    suffix = request_data.get("suffix", None)
    stream = request_data.get("stream", False)
    seed = request_data.get("seed", None)
    top_p = request_data.get("top_p", 1)

    # Debugging information: Check if parameters are given by API or use defaults
    logger.debug("Parameter source debug:")
    logger.debug("\n" + format_debug_output(request_data))

    # Get IBM IAM token
    iam_token = get_iam_token()

    # Prepare Watsonx.ai request payload with HAP disabled
    watsonx_payload = {
        "input": prompt,  # Ensure 'prompt' does not have extra single quotes around it
        "parameters": {
            "decoding_method": "sample", # decoding_method = Greedy is not supported.
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 50,
            "top_p": top_p,
            "random_seed": seed,
            #"stop_sequences": stop,
            "repetition_penalty": presence_penalty
            #"frequency_penalty": frequency_penalty
        },
        "model_id": model_id,
        "project_id": PROJECT_ID
    }

    # Properly escape and format special characters using json.dumps
    formatted_payload = json.dumps(watsonx_payload, indent=4, ensure_ascii=False)

    # Log the prettified JSON with escaped characters
    logger.debug(f"Sending request to Watsonx.ai: {formatted_payload}")

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
        logger.error(f"Error: calling Watsonx.ai: {err}")
        raise HTTPException(status_code=500, detail=f"Error calling Watsonx.ai: {err}")

    # Extract the generated text from the Watsonx response
    results = watsonx_data.get("results", [])
    if results and "generated_text" in results[0]:
        generated_text = results[0]["generated_text"]
        logger.debug(f"Generated text from Watsonx.ai: \n{generated_text}")
    else:
        generated_text = "\n\nNo response available."
        logger.warning("No generated text found in Watsonx.ai response.")

    # Prepare the OpenAI-compatible response with model name "llama-3-405b-instruct"
    openai_response = {
        "id": f"cmpl-{str(uuid.uuid4())[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_id,
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
