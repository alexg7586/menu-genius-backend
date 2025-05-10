import os
import base64
import json
import aiohttp
import asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# OpenAI credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def split_menu_text(menu_text: str, chunk_size: int = 12):
    lines = [line.strip() for line in menu_text.splitlines() if line.strip()]
    return ["\n".join(lines[i:i + chunk_size]) for i in range(0, len(lines), chunk_size)]

async def extract_text_from_image_async(session, image_data: bytes) -> str:
    base64_image = base64.b64encode(image_data).decode("utf-8")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an OCR assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract text from this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as resp:
        result = await resp.json()
        return result["choices"][0]["message"]["content"].strip()

async def generate_chunk(session, chunk: str):
    prompt = f"""
You are analyzing a restaurant menu. Each line may include a dish name, an optional description, and sometimes a price.

Instructions:
- Extract the actual dish name (omit numbering or category labels).
- Extract any available price as a string. If no price is found, you may omit the field or return an empty string.
- If a description exists, rewrite it into 1–2 clear, natural English sentences.
- If no description exists, generate one based on the dish name and common culinary context.
- Return only real dishes, not section headers.

Output format:
[
  {{
    "name": "Dish Name",
    "description": "Short English description.",
    "price": "Optional price as a string, e.g., '$12.99', '120 THB', or '₫45,000'"
  }},
  ...
]

Menu:
{chunk}
"""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a food expert."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as resp:
        result = await resp.json()
        return json.loads(result["choices"][0]["message"]["content"])  # List[dict] with name, description, price

@app.post("/upload")
async def upload_menu(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        return {"error": "Invalid file type."}

    image_data = await file.read()

    async with aiohttp.ClientSession() as session:
        try:
            ocr_text = await extract_text_from_image_async(session, image_data)
            chunks = split_menu_text(ocr_text)
            tasks = [generate_chunk(session, chunk) for chunk in chunks]
            results = await asyncio.gather(*tasks)

            # Flatten the list of lists
            menu_items = [item for sublist in results for item in sublist]
            return {"menu": menu_items}
        except Exception as e:
            return {"error": str(e)}