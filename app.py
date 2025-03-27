import os
import openai
import base64
import json
import aiohttp
import asyncio
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

# CORS 
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

# ---------------------- OCR ----------------------
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

# ---------------------- Split ----------------------
def split_menu_text(menu_text: str) -> List[str]:
    lines = [line.strip() for line in menu_text.splitlines() if line.strip()]
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

# ---------------------- Prompt Templates ----------------------
EN_PROMPT_TEMPLATE = """
The following is part of a restaurant menu. For each actual dish, provide:

- A translated name (omit numbering, category labels, and prices)
- A short description (ingredients, flavor, preparation)

Ignore any price information (e.g., numbers like "$12.99", "25元") when identifying dish names.
If an item is part of a set meal (e.g., optional dishes listed under a combo), do not list it as a standalone dish.
Avoid long descriptions or unnecessary details. Use 1-2 short sentences.
Respond only in English.

Format your response as a valid JSON array:
[
  {{
    "name": "...",
    "description": "..."
  }}
]

Menu:
{chunk_text}
"""

ZH_PROMPT_TEMPLATE = """
以下是餐厅菜单的一部分。请提取其中每一道真实的菜品，并给出以下信息：

- 菜名（翻译为中文，不包括编号、栏目或价格）
- 简短的中文描述（包含主要食材、口味和制作方式）

请忽略价格（如 "$12.99"、"25元"）以及套餐中的可选项目，不要将这些当作独立菜品。
描述应简洁自然，控制在 1-2 句话之内。只用中文作答。

请以以下 JSON 格式返回：
[
  {{
    "name": "...",
    "description": "..."
  }}
]

菜单内容：
{chunk_text}
"""

# ---------------------- GPT Async ----------------------
async def generate_chunk_descriptions(session, chunk_text: str, output_language: str):
    if output_language.lower() == "chinese":
        prompt = ZH_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
    else:
        prompt = EN_PROMPT_TEMPLATE.format(chunk_text=chunk_text)

    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a food expert."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30) as resp:
            data = await resp.json()
            content = data["choices"][0]["message"]["content"].strip("```json").strip("```")
            return json.loads(content)
    except Exception as e:
        return [{"name": "Error", "description": f"Failed to process chunk: {str(e)}"}]

# ---------------------- Async Merge ----------------------
async def get_menu_descriptions_async(menu_text: str, output_language: str):
    chunks = split_menu_text(menu_text)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = [generate_chunk_descriptions(session, chunk, output_language) for chunk in chunks]
        completed = await asyncio.gather(*tasks)
        for result in completed:
            results.extend(result)

    return results

# ---------------------- Upload API ----------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), language: str = Form("English")):
    if not allowed_file(file.filename):
        return {"error": "Unsupported file type. Please upload JPG, JPEG, PNG, or WEBP."}

    file_data = await file.read()
    menu_text = extract_text_from_image(file_data)

    if not menu_text.strip() or len(menu_text.strip()) < 10:
        return {"error": "OCR failed or returned invalid text"}

    menu_descriptions = await get_menu_descriptions_async(menu_text, language)
    return {"menu": menu_descriptions}

# ---------------------- Uvicorn Entry ----------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5001))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
