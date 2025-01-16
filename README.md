# Piiranha Server - PII Detection and Masking Service

![PII-Ranha Logo](logo.png)


A FastAPI-based service for detecting and masking Personally Identifiable Information (PII) in text using the PII-Ranha model.

## Features

- PII Detection: Check if text contains any PII
- PII Masking: Mask detected PII with either aggregated or detailed redaction
- Health Check: Verify service status
- Token Chunking: Handles long texts by splitting into 256-token chunks

## Examples


```bash
% curl -X POST "http://localhost:8001/check-pii" -i  \
-H "Content-Type: application/json" \
-d '{"text":"Your text here, my name is Jean-Claude Dusse"}'
HTTP/1.1 400 Bad Request
date: Thu, 16 Jan 2025 11:40:49 GMT
server: uvicorn
content-length: 34
content-type: application/json

{"detail":"PII detected in input"
```
```bash
% curl -X POST "http://localhost:8001/check-pii" -i \
-H "Content-Type: application/json" \
-d '{"text":"Your text here, Lorem ipsum dolor sit amet, consectetur adipiscing elit"}'
HTTP/1.1 200 OK
date: Thu, 16 Jan 2025 11:40:07 GMT
server: uvicorn
content-length: 43
content-type: application/json

{"status":"OK","message":"No PII detected"
```



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


## Author & Enterprise Support

**Author**: Sébastien Campion 

Sébastien Campion - sebastien.campion@foss4.eu

**Note**: This project is under active development. Please report any issues or feature requests through the issue tracker.

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
