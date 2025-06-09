from fastapi import FastAPI, Request, HTTPException
import requests
import os
import time
import uuid
import logging
import json
from tabulate import tabulate
from ibm_watsonx_ai import APIClient, Credentials
from fastapi.responses import StreamingResponse

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

api_version = os.getenv("WATSONX_VERSION") or "2023-05-29"

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


# IBM Cloud IAM URL for fetching the token
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# Load IBM API key, Watsonx URL, and Project ID from environment variables
IBM_API_KEY = os.getenv("WATSONX_IAM_APIKEY")
WATSONX_MODELS_URL = f"{WATSONX_URLS.get(region)}/ml/v1/foundation_model_specs"
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

# Construct Watsonx URLs with the version parameter
WATSONX_URL = f"{WATSONX_URLS.get(region)}/ml/v1/text/generation?version={api_version}"
WATSONX_URL_CHAT = f"{WATSONX_URLS.get(region)}/ml/v1/text/chat?version={api_version}"

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
        ("Model ID", request_data.get("model", "ibm/granite-20b-multilingual"), "ibm/granite-20b-multilingual",
        "ID of the model to use for completion"),

        ("Max Tokens", request_data.get("max_tokens", 2000), 2000,
        "Maximum number of tokens to generate in the completion. The total tokens, prompt + completion."),

        ("Temperature", request_data.get("temperature", 0.2), 0.2,
        "Controls the randomness of the generated output. Higher values make the output more random."),

        ("Presence Penalty", request_data.get("presence_penalty", 1), 1,
        "Penalizes new tokens based on whether they appear in the text so far. Positive values encourage the model to talk about new topics."),

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
        if api_value is not None:
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


# Fetch the models from Watsonx
def get_watsonx_models():
    try:
        token = get_iam_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Send request to Watsonx models endpoint
        params = {
            "version": "2024-09-16",  # Ensure the version is correct
            "filters": "function_text_generation,!lifecycle_withdrawn:and",
            "limit": 200
        }
        response = requests.get(
            WATSONX_MODELS_URL,
            headers=headers,
            params=params
        )

        if response.status_code == 404:
            logger.error("404 Not Found: The endpoint or version might be incorrect.")
            raise HTTPException(status_code=404, detail="Watsonx Models API endpoint not found.")

        response.raise_for_status()  # Raise exception for any non-200 status codes
        models_data = response.json()
        return models_data
    except requests.exceptions.RequestException as err:
        logger.error(f"Error fetching models from Watsonx.ai: {err}")
        raise HTTPException(status_code=500, detail=f"Error fetching models from Watsonx.ai: {err}")


# Convert Watsonx models to OpenAI-like format
def convert_watsonx_to_openai_format(watsonx_data):
    openai_models = []

    for model in watsonx_data['resources']:
        limits = model.get('model_limits', {}) or {}
        max_output = limits.get('max_output_tokens')
        max_seq   = limits.get('max_sequence_length')

        openai_model = {
            "id": model['model_id'],  # Watsonx's model_id maps to OpenAI's id
            "object": "model",  # Hardcoded, as OpenAI uses "model" as the object type
            "created": int(time.time()),  # Optional: use current timestamp or a fixed one if available
            "owned_by": f"{model.get('provider')} / {model.get('source')}",  # Combine Watsonx's provider and source
						"description": (
                f"{model.get('short_description', '').strip()} "
                f"Supports tasks like {', '.join(model.get('task_ids', []))}."
            ).strip(),	  # Watsonx's short description
            "max_tokens": max_output,  # Map Watsonx's max_output_tokens to OpenAI's max_tokens
            "token_limits": {
                "max_sequence_length": max_seq,  # Watsonx's max_sequence_length
                "max_output_tokens": max_output  # Watsonx's max_output_tokens
            }
        }
        openai_models.append(openai_model)

    return {
        "data": openai_models
    }


# FastAPI route for /v1/models
@app.get("/v1/models")
async def fetch_models():
    try:
        models = get_watsonx_models()  # Fetch the Watsonx models data
        model_names = [m['model_id'] for m in models['resources']]
        logger.debug(f"Available models: {model_names}")
        # logger.debug(f"Available models: {models}")

        # Convert Watsonx output to OpenAI-like format
        openai_like_models = convert_watsonx_to_openai_format(models)

        # Return the OpenAI-like formatted models
        return openai_like_models
    except Exception as err:
        logger.error(f"Error fetching models: {err}")
        raise HTTPException(status_code=500, detail=f"Error fetching models: {err}")

# FastAPI route for /v1/models/{model_id}
@app.get("/v1/models/{model_id}")
async def fetch_model_by_id(model_id: str):
    try:
        # Fetch the full list of models from Watsonx
        models = get_watsonx_models()

        # Search for the model with the specific model_id
        model = next((m for m in models['resources'] if m['model_id'] == model_id), None)

        if model:
            # Convert the model details to OpenAI-like format
            openai_model = {
                "id": model['model_id'],  # Watsonx's model_id maps to OpenAI's id
                "object": "model",  # Hardcoded, as OpenAI uses "model" as the object type
                "created": int(time.time()),  # Optional: use current timestamp or a fixed one if available
                "owned_by": f"{model['provider']} / {model['source']}",  # Combine Watsonx's provider and source
                "description": f"{model['short_description']} Supports tasks like {', '.join(model.get('task_ids', []))}.",  # Watsonx's short description
                "max_tokens": model['model_limits']['max_output_tokens'],  # Map Watsonx's max_output_tokens to OpenAI's max_tokens
                "token_limits": {
                    "max_sequence_length": model['model_limits']['max_sequence_length'],  # Watsonx's max_sequence_length
                    "max_output_tokens": model['model_limits']['max_output_tokens']  # Watsonx's max_output_tokens
                }
            }
            return {"data": [openai_model]}

        # If model not found, raise a 404 error
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found.")

    except Exception as err:
        logger.error(f"Error fetching model by ID: {err}")
        raise HTTPException(status_code=500, detail=f"Error fetching model by ID: {err}")


def extract_tool_context(content: str):
    """Extract tool context from content if present"""
    if "<context>" in content and "</context>" in content:
        start_idx = content.index("<context>") + len("<context>")
        end_idx = content.index("</context>")
        tool_context = content[start_idx:end_idx].strip()
        # cleaned_content = content.replace(f"<context>{tool_context}</context>", "").strip()
        cleaned_content = content
        return tool_context, cleaned_content
    return None, content


def generate_watsonx_stream(headers, payload, model_id):
	payload["stream"] = True
	headers["Accept"] = "text/event-stream"
	
	# open the Watsonx stream
	try:
		watson_resp = requests.post(
				WATSONX_URL_CHAT,
				headers=headers,
				json=payload,
				stream=True,
				timeout=(5, None)
		)
		watson_resp.raise_for_status()
	except requests.RequestException as e:
		logger.error(f"Error opening Watsonx stream: {e}")
		raise HTTPException(status_code=500, detail="Could not open Watsonx stream")

	logger.debug(watson_resp)

	def iter_events():
		try:
			for line in watson_resp.iter_lines(decode_unicode=True):
				logger.debug(line)
				if not line:
					continue
				if line.startswith("data:"):
					line = line[len("data:"):].strip()
				try:
					chunk = json.loads(line)
				except json.JSONDecodeError:
					continue

				# extract text in whatever shape Watsonx gave us
				if "message" in chunk and isinstance(chunk["message"], dict):
					text = chunk["message"].get("content", "")
				elif "choices" in chunk:
					# Detect the summary chunk with no choices
					if not chunk.get("choices"):
						# send the OpenAI‐style end‐of‐stream
						yield "data: [DONE]\n\n"
						break
					text = chunk["choices"][0].get("delta", {}).get("content", "")
				else:
					text = chunk.get("content", "")

				# check for final finish_reason
				finish = chunk["choices"][0].get("finish_reason")
				is_done = finish is not None or chunk.get("done")
				
				out = {
					"id": chunk.get("id"),
					"object": "chat.completion.chunk",
					"model": model_id,
					"choices": [
						{
							"delta": {"content": text},
							"index": 0,
							"finish_reason": finish if is_done else None
						}
					]
				}
				yield f"data: {json.dumps(out)}\n\n"

				# note: NO break here—just keep reading until the server closes
		finally:
			watson_resp.close()

	return StreamingResponse(iter_events(), media_type="text/event-stream")


def generate_watsonx_non_stream(headers, payload, model_id):
	try:
		response = requests.post(WATSONX_URL_CHAT, headers=headers, json=payload)
		response.raise_for_status()
		data = response.json()
		logger.debug(f"Watsonx.ai response:\n{json.dumps(data, indent=4)}")
	except requests.exceptions.HTTPError as err:
		logger.error(f"Watsonx HTTPError: {err}\nResponse: {response.text}")
		raise HTTPException(status_code=response.status_code, detail=f"Error from Watsonx.ai: {response.text}")
	except requests.exceptions.RequestException as err:
		logger.error(f"Watsonx RequestException: {err}")
		raise HTTPException(status_code=500, detail=f"Error calling Watsonx.ai: {err}")
	except Exception as err:
		logger.error(f"Unexpected error handling Watsonx.ai response: {err}")
		raise HTTPException(status_code=500, detail="Unexpected error from Watsonx.ai")

	try:        
		# Process response: translate back to OpenAI format
		choices = []
		for choice in data.get("choices", []):
			raw_msg = choice.get("message", {})
			role    = raw_msg.get("role", "assistant")
			content = raw_msg.get("content", "")
			tool_context, clean_content = extract_tool_context(content)
			
			message = {"role": role, "content": clean_content}
			if tool_context:
				message["context"] = tool_context
					
			choices.append({
				"index": len(choices),
				"message": message,
				"done_reason": choice.get("done_reason", "stop")
			})
				
		response_payload = {
			"id": data.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
			"model": data.get("model_id", model_id),
			"created_at": data.get("created_at", int(time.time())),
			"object": "chat.completion",
			"choices": choices,
			"usage": data.get("usage", {})
		}
		# Log final OpenAI-style response
		logger.debug(f"Returning OpenAI-compatible chat response:\n{json.dumps(response_payload, indent=4)}")
		return response_payload
			
	except Exception as e:
		logger.error(f"Watsonx API error: {e}")
		raise HTTPException(status_code=500, detail="Error processing request")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
	logger.info("Received a Watsonx chat completion request.")
	
	# Parse request
	try:
		req = await request.json()
		logger.debug(f"Raw request JSON: {json.dumps(req, indent=2)}")
	except Exception as e:
		logger.error(f"Invalid request: {e}")
		raise HTTPException(status_code=400, detail="Invalid JSON request format")

	try:
		model_id     = req["model"]
		messages     = req["messages"]
	except KeyError as e:
		logger.error(f"Missing required field in request: {e}")
		raise HTTPException(status_code=400, detail=f"Missing required field: {e}")

	stream       = req.get("stream", False)
	max_tokens   = req.get("max_tokens")
	temperature  = req.get("temperature")
	top_p        = req.get("top_p")

	# Prepare Watsonx payload
	watson_messages = []
	for msg in messages:
		content = msg["content"]
		watson_content = [{"type": "text", "text": content}] if isinstance(content, str) else content
		watson_messages.append({
			"role": msg["role"],
			"content": watson_content
		})

	payload = {
		"model_id": model_id,
		"project_id": PROJECT_ID,
		"messages": watson_messages,
	}

	if max_tokens is not None:
		payload["max_tokens"] = max_tokens

	params = {}
	if temperature is not None:
		params["temperature"] = temperature
	if top_p is not None:
		params["top_p"] = top_p
	if params:
		payload["parameters"] = params

	# Log payload sent to Watsonx
	logger.debug(f"Watsonx.ai request payload:\n{json.dumps(payload, indent=4)}")
	
	headers = {
		"Authorization": f"Bearer {get_iam_token()}",
		"Content-Type": "application/json"
	}

	if stream:
		# Handle streaming response
		return generate_watsonx_stream(headers, payload, model_id)
	else:
		# Handle non-streaming response
		return generate_watsonx_non_stream(headers, payload, model_id)
	


