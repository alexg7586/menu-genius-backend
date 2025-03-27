import os
import base64
import json
import aiohttp
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Logging setup
logging.basicConfig(level=logging.INFO)

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
    try:
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
            content = result.get("choices", [{}])[0].get("message", {}).get("content")
            if not content:
                raise ValueError("OpenAI OCR: No content returned")
            return content.strip()
    except Exception as e:
        raise RuntimeError(f"OCR failed: {str(e)}")

async def generate_chunk(session, chunk: str):
    try:
        prompt = f"""
You are analyzing a restaurant menu. Each line may include a dish name, or a dish name followed by a description.

Instructions:
- Extract the actual dish name (omit prices, numbering, category labels, and combo options).
- If a description exists, rewrite it into one or two clear, natural English sentences.
- If no description exists, generate one based on the dish name and common culinary context.
- Return dishes only, not headers.

Output format:
[{{"name": "Dish Name", "description": "Short English description."}}, ...]

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
            content = result.get("choices", [{}])[0].get("message", {}).get("content")
            if not content:
                raise ValueError("OpenAI GPT: No content returned")
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                raise ValueError("OpenAI GPT: Invalid JSON response")
    except Exception as e:
        raise RuntimeError(f"GPT generation failed: {str(e)}")

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
            logging.error("Upload processing failed", exc_info=True)
            return {"error": str(e)}