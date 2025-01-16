# PII Detection and Masking Service

![PII-Ranha Logo](logo.png)

A FastAPI-based service for detecting and masking Personally Identifiable Information (PII) in text using the PII-Ranha model.

## Features

- PII Detection: Check if text contains any PII
- PII Masking: Mask detected PII with either aggregated or detailed redaction
- Health Check: Verify service status
- Token Chunking: Handles long texts by splitting into 256-token chunks

## API Endpoints

### Check PII
`POST /check-pii`
- Request body: `{"text": "your text here"}`
- Returns: 200 if no PII detected, 400 if PII detected

### Mask PII
`POST /mask-pii`
- Request body: `{"text": "your text here"}`
- Optional query param: `aggregate_redaction=true|false` (default: true)
- Returns: Original and masked text

### Health Check
`GET /health`
- Returns: Service status

## Running the Service

1. Install dependencies:
```bash
pip install fastapi uvicorn transformers torch
```

2. Run the server:
```bash
python app.py
```

3. The service will be available at `http://localhost:8000`

## Testing with curl

Check for PII:
```bash
curl -X POST "http://localhost:8000/check-pii" -H "Content-Type: application/json" -d '{"text":"Your text here"}'
```

Mask PII:
```bash
curl -X POST "http://localhost:8000/mask-pii" -H "Content-Type: application/json" -d '{"text":"Your text here"}'
```

## Development

The service uses the [PII-Ranha](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information) model from Hugging Face.

## License

[MIT License](LICENSE)
