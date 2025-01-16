from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

app = FastAPI()

# Initialize PII detection model with better error handling
try:
    model_name = "iiiorg/piiranha-v1-detect-personal-information"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("PII detection model loaded successfully")
except Exception as e:
    print(f"Failed to load PII detection model: {e}")
    raise


class TextInput(BaseModel):
    text: str


def contains_pii(text: str) -> bool:
    """Check if text contains any PII using the PII detection model"""
    MAX_TOKENS = 256
    try:
        # Split text into chunks of MAX_TOKENS tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = [tokens[i:i + MAX_TOKENS] for i in range(0, len(tokens), MAX_TOKENS)]
        
        for chunk in chunks:
            # Convert chunk back to text
            chunk_text = tokenizer.decode(chunk)
            
            # Process each chunk
            inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            if any(label != model.config.label2id['O'] for label in predictions[0]):
                return True
                
        return False
    except Exception as e:
        print(f"Error in contains_pii: {e}")
        return True  # Assume PII exists if we can't check


def mask_pii(text: str, aggregate_redaction: bool = True) -> str:
    """Mask PII in text using the PII detection model"""
    MAX_TOKENS = 256
    masked_text = list(text)
    
    # Split text into chunks of MAX_TOKENS tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + MAX_TOKENS] for i in range(0, len(tokens), MAX_TOKENS)]
    
    for chunk in chunks:
        # Convert chunk back to text
        chunk_text = tokenizer.decode(chunk)
        
        # Process each chunk
        inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1)
        encoded_inputs = tokenizer.encode_plus(chunk_text, return_offsets_mapping=True, add_special_tokens=True)
        offset_mapping = encoded_inputs['offset_mapping']

        is_redacting = False
        redaction_start = 0
        current_pii_type = ''

        for i, (start, end) in enumerate(offset_mapping):
            if start == end:  # Special token
                continue

            label = predictions[0][i].item()
            if label != model.config.label2id['O']:  # Non-O label
                pii_type = model.config.id2label[label]
                if not is_redacting:
                    is_redacting = True
                    redaction_start = start
                    current_pii_type = pii_type
                elif not aggregate_redaction and pii_type != current_pii_type:
                    apply_redaction(masked_text, redaction_start, start, current_pii_type, aggregate_redaction)
                    redaction_start = start
                    current_pii_type = pii_type
            else:
                if is_redacting:
                    apply_redaction(masked_text, redaction_start, end, current_pii_type, aggregate_redaction)
                    is_redacting = False

        if is_redacting:
            apply_redaction(masked_text, redaction_start, len(masked_text), current_pii_type, aggregate_redaction)

    return ''.join(masked_text)


def apply_redaction(masked_text: list, start: int, end: int, pii_type: str, aggregate_redaction: bool):
    """Apply redaction to a portion of text with better handling"""
    try:
        # Ensure we don't go out of bounds
        start = max(0, min(start, len(masked_text)))
        end = max(0, min(end, len(masked_text)))

        # Clear the text range
        for j in range(start, end):
            masked_text[j] = ''

        # Apply the redaction marker
        if start < len(masked_text):
            if aggregate_redaction:
                masked_text[start] = '[redacted]'
            else:
                masked_text[start] = f'[{pii_type}]'
    except Exception as e:
        print(f"Error in apply_redaction: {e}")


@app.post("/check-pii")
async def check_pii(input_data: TextInput):
    """Endpoint to check for PII in input text"""
    if contains_pii(input_data.text):
        raise HTTPException(status_code=400, detail="PII detected in input")
    return {"status": "OK", "message": "No PII detected"}


@app.post("/mask-pii")
async def mask_pii_endpoint(input_data: TextInput, aggregate_redaction: bool = True):
    """Endpoint to mask PII in input text"""
    masked_text = mask_pii(input_data.text, aggregate_redaction)
    return {
        "status": "OK",
        "original_text": input_data.text,
        "masked_text": masked_text,
        "aggregate_redaction": aggregate_redaction
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "OK"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
