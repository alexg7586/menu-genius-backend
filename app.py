import os
import openai
import base64
import json
import aiohttp
import asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helpers ---
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- OCR ---
def extract_text(file_bytes: bytes) -> str:
    base64_img = base64.b64encode(file_bytes).decode("utf-8")
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an OCR assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract text from this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]},
        ]
    )
    return response.choices[0].message.content.strip()

# --- Prompt ---
PROMPT = """
The following is a restaurant menu. For each line:
- Extract the dish name (omit prices or numbering)
- If a description is included, rewrite it naturally
- If not, create one based on the dish name
Respond in this JSON format:
[
  {"name": "...", "description": "..."},
  ...
]
Menu:
{chunk}
"""

# --- GPT Call ---
async def gpt_generate(session, chunk):
    prompt = PROMPT.format(chunk=chunk)
    headers = {
        "Authorization": f"Bearer {openai.api_key}" ,
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a food expert."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500
    }
    try:
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as resp:
            result = await resp.json()
            content = result["choices"][0]["message"]["content"].strip("```json").strip("```")
            return json.loads(content)
    except:
        return []

# --- Chunking ---
def split_lines(text: str, size: int = 6) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return ["\n".join(lines[i:i+size]) for i in range(0, len(lines), size)]

# --- Main Route ---
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        return {"error": "Invalid file type."}

    content = await file.read()
    text = extract_text(content)
    if not text.strip():
        return {"error": "OCR failed or empty result."}

    chunks = split_lines(text)
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [gpt_generate(session, chunk) for chunk in chunks]
        for result in await asyncio.gather(*tasks):
            results.extend(result)

    return {"menu": results}

# --- Uvicorn Entry ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5001))
    uvicorn.run("app_minimal:app", host="0.0.0.0", port=port)
