# Watsonx OpenAPI Project

This repository provides a FastAPI-based API that mimics the OpenAI `/v1/completions` endpoint and forwards requests to IBM Watsonx.ai for text generation. The default model used is `llama-3-405b-instruct`.

## Features

- Receives text prompts and forwards them to IBM Watsonx.ai for text generation.
- Returns the generated text in a format compatible with the OpenAI API.
- Utilizes environment variables to configure the Watsonx API key and Watsonx deployment URL.

## Subdirectories

- **[`fastapi-watsonx`](fastapi-watsonx/)**: Main FastAPI application to interact with Watsonx.ai.
- **[`openai-new`](openai-new/)**: Converts legacy OpenAI completion commands to the new format.
- **[`streamlit`](streamlit/)**: A simple UI built with Streamlit for interacting with the API.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/aseelert/watsonx-openapi
    cd watsonx-openapi
    ```

2. For detailed setup instructions, refer to the specific subdirectory READMEs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
