```bash
cd ~/watsonx-openai-api/fastapi-watsonx

docker build -t fastapi-watsonx:1.0-amd64 --platform linux/amd64 .
docker build -t fastapi-watsonx:1.0-arm64 --platform linux/arm64/v8 .

docker tag fastapi-watsonx:1.0-amd64 aseelert/watsonxai-endpoint:1.0-amd64
docker tag fastapi-watsonx:1.0-arm64 aseelert/watsonxai-endpoint:1.0-arm64

docker push aseelert/watsonxai-endpoint:1.0-amd64
docker push aseelert/watsonxai-endpoint:1.0-arm64

docker manifest create aseelert/watsonxai-endpoint:1.0 aseelert/watsonxai-endpoint:1.0-arm64 aseelert/watsonxai-endpoint:1.0-amd64 --amend
docker manifest push aseelert/watsonxai-endpoint:1.0
```
