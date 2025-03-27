
import os
import openai
import base64
import json
import aiohttp
import asyncio
import re
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------- OCR 提取 ----------------------
def extract_text_from_image(file_data: bytes) -> str:
    base64_image = base64.b64encode(file_data).decode("utf-8")
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an OCR assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract text from this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]
    )
    return response.choices[0].message.content.strip()

# ---------------------- 菜名清洗 ----------------------
def preprocess_menu_lines(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_lines = []
    for line in lines:
        line = re.sub(r"^\d+[\.|\-]\s*", "", line)  # 去前缀编号
        line = re.sub(r"(\s*[-–—]*\s*)?\$?\d+(\.\d+)?\s*(元|rmb|usd)?$", "", line, flags=re.IGNORECASE)  # 去价格
        clean_lines.append(line.strip())
    return clean_lines

# ---------------------- 智能分组 ----------------------
def split_menu_text(menu_text: str) -> List[str]:
    lines = preprocess_menu_lines(menu_text)
    total = len(lines)

    if total <= 8:
        chunk_size = total
    elif total <= 15:
        chunk_size = 5
    elif total <= 30:
        chunk_size = 6
    else:
        chunk_size = 5

    chunks = []
    for i in range(0, total, chunk_size):
        chunk = "\n".join(lines[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# ---------------------- 异步 GPT 调用 ----------------------
async def generate_chunk_descriptions(session, chunk_text: str, output_language: str):
    prompt = f"""
You are a culinary expert. For each of the following menu items, generate a concise description in {output_language}, limited to 1–2 complete sentences. The description must include key details such as ingredients, flavor, and preparation style, while staying natural and informative.

Only output in {output_language}. Return a valid JSON array like:
[{{"name": "...", "description": "..."}}]

Menu items:
{chunk_text}
"""
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a food expert."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4
    }

    try:
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30) as resp:
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            if "```json" in content:
                content = content.split("```json")[-1].split("```")[0].strip()
            try:
                return json.loads(content)
            except Exception as e:
                return [{"name": "Error", "description": f"Failed to process chunk: {str(e)}"}]
    except Exception as e:
        return [{"name": "Error", "description": f"OpenAI API error: {str(e)}"}]

# ---------------------- 主逻辑 ----------------------
async def get_menu_descriptions_async(menu_text: str, output_language: str):
    chunks = split_menu_text(menu_text)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = [generate_chunk_descriptions(session, chunk, output_language) for chunk in chunks]
        completed = await asyncio.gather(*tasks)
        for result in completed:
            results.extend(result)

    return results

# ---------------------- 接口 ----------------------
from fastapi.responses import JSONResponse

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), language: str = Form("English")):
    if not allowed_file(file.filename):
        return JSONResponse(content={"error": "Unsupported file type. Please upload JPG, JPEG, PNG, or WEBP."}, status_code=400)

    file_data = await file.read()
    menu_text = extract_text_from_image(file_data)

    if not menu_text.strip() or len(menu_text.strip()) < 10:
        return JSONResponse(content={"error": "OCR failed or returned invalid text"}, status_code=400)

    menu_descriptions = await get_menu_descriptions_async(menu_text, language)
    return {"menu": menu_descriptions}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5001))
    uvicorn.run("app:app", host="0.0.0.0", port=port)