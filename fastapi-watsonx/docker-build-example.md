```bash
cd ~/watsonx-openai-api/fastapi-watsonx

docker build --no-cache -t fastapi-watsonx:1.1-amd64 --platform linux/amd64 .
docker build --no-cache -t fastapi-watsonx:1.1-arm64 --platform linux/arm64/v8 .

docker tag fastapi-watsonx:1.1-amd64 aseelert/watsonxai-endpoint:1.1-amd64
docker tag fastapi-watsonx:1.1-arm64 aseelert/watsonxai-endpoint:1.1-arm64

docker push aseelert/watsonxai-endpoint:1.1-amd64
docker push aseelert/watsonxai-endpoint:1.1-arm64

docker manifest create aseelert/watsonxai-endpoint:1.1 aseelert/watsonxai-endpoint:1.1-arm64 aseelert/watsonxai-endpoint:1.1-amd64 --amend
docker manifest push aseelert/watsonxai-endpoint:1.1
```


Local build
```bash
docker build --no-cache -t fastapi-watsonx .
docker tag fastapi-watsonx quay.io/gas_stocktrader/fastapi-watsonx:latest
docker push quay.io/gas_stocktrader/fastapi-watsonx:latest
docker run -d --rm --env-file=.env -p 8000:8000 fastapi-watsonx
```