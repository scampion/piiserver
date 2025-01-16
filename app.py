from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import re

app = FastAPI()

# Simple regex patterns for common PII types
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'phone': r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
}

class TextInput(BaseModel):
    text: str

def contains_pii(text: str) -> bool:
    """Check if text contains any PII patterns"""
    for pattern in PII_PATTERNS.values():
        if re.search(pattern, text):
            return True
    return False

@app.post("/check-pii")
async def check_pii(input_data: TextInput):
    """Endpoint to check for PII in input text"""
    if contains_pii(input_data.text):
        raise HTTPException(status_code=400, detail="PII detected in input")
    return {"status": "OK", "message": "No PII detected"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
