
import os
import openai
import base64
import json
import aiohttp
import asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Set OpenAI API key and model from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("GPT_MODEL", "gpt-4o")

# Enable CORS for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allowed image types for upload
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Clean raw OCR text before sending to GPT
def clean_ocr_text(raw_text: str) -> str:
    lines = raw_text.strip().splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 2:
            continue
        if any(c.isdigit() for c in line) and ('$' in line or len(line) < 4):
            continue
        if line.lower() in ["menu", "starter", "main", "dessert"]:
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)

# Extract text using OpenAI Vision API
async def extract_text_from_image(file_data: bytes) -> str:
    base64_image = base64.b64encode(file_data).decode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai.api_key}"}
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please extract all visible text from this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 1000
    }
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as resp:
            result = await resp.json()
            return result["choices"][0]["message"]["content"]

# Use GPT to generate descriptions
async def generate_menu_descriptions(menu_text: str, language: str = "English") -> str:
    prompt = f"""
You are a professional food expert. The following text was extracted from a restaurant menu image. For each line that seems like a dish name, generate a short, accurate, 1-2 sentence description. If the line is not a food item, skip it. Only output real dishes.

Output format:
Dish Name: Description

Respond in {language}.

Menu text:
{menu_text}
"""
    response = await openai.ChatCompletion.acreate(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# Parse GPT output
def parse_items(text: str):
    items = []
    for line in text.strip().split("\n"):
        if ":" in line:
            name, desc = line.split(":", 1)
            items.append({
                "name": name.strip(),
                "description": desc.strip()
            })
    return items

# Upload endpoint
@app.post("/upload")
async def upload_image(file: UploadFile = File(...), language: str = "English"):
    if not allowed_file(file.filename):
        return {"error": "Invalid file type."}

    file_data = await file.read()
    try:
        raw_text = await extract_text_from_image(file_data)
        cleaned_text = clean_ocr_text(raw_text)
        gpt_result = await generate_menu_descriptions(cleaned_text, language)
        items = parse_items(gpt_result)
        return {"items": items}
    except Exception as e:
        return {"error": str(e)}
