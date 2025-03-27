import os
import openai
import base64
import json
import aiohttp
import asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

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

def extract_text_from_image(file_data: bytes) -> str:
    base64_image = base64.b64encode(file_data).decode("utf-8")
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an OCR assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract text from this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]
    )
    return response.choices[0].message.content.strip()

def split_menu_text(menu_text: str, chunk_size: int = 6):
    lines = [line.strip() for line in menu_text.splitlines() if line.strip()]
    return ["\n".join(lines[i:i + chunk_size]) for i in range(0, len(lines), chunk_size)]

async def generate_chunk(session, chunk: str):
    prompt = f"""
The following is a restaurant menu. For each dish:
- Extract the name (omit prices, labels, numbering)
- If description exists, rephrase it; else generate one
- Be concise (1–2 sentences)
Respond in JSON: [{{"name": "...", "description": "..."}}, ...]
Menu:
{chunk}
"""
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
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
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30) as resp:
            data = await resp.json()
            content = data["choices"][0]["message"]["content"].strip("```json").strip("```")
            return json.loads(content)
    except:
        return []

async def get_menu(menu_text: str):
    chunks = split_menu_text(menu_text)
    async with aiohttp.ClientSession() as session:
        tasks = [generate_chunk(session, chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
    return [item for group in results for item in group if "name" in item and "description" in item]

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        return {"error": "Unsupported file type."}

    file_data = await file.read()
    text = extract_text_from_image(file_data)
    if not text.strip():
        return {"error": "OCR failed or empty."}

    menu = await get_menu(text)
    return {"menu": menu}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 5001)))

