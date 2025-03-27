import os
import openai
import base64
import json
import asyncio
import aiohttp
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------- OCR 同步提取 ----------------------
def extract_text_from_image(file_data):
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

# ---------------------- 智能分组菜单 ----------------------
def split_menu_text(menu_text):
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

# ---------------------- Async GPT 生成描述 ----------------------
async def generate_chunk_descriptions(session, chunk_text, output_language="English"):
    prompt = f"""
Translate the following menu items into {output_language} and write a short, rich description for each (ingredients, flavor, prep).
Use only {output_language}. Return valid JSON: [{{"name": "...", "description": "..."}}]

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
        ]
    }

    try:
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30) as resp:
            data = await resp.json()
            content = data["choices"][0]["message"]["content"].strip("```json").strip("```")
            return json.loads(content)
    except Exception as e:
        return [{"name": "Error", "description": f"Failed to process chunk: {str(e)}"}]

# ---------------------- Async 执行全部任务 ----------------------
async def get_menu_descriptions_async(menu_text, output_language="English"):
    chunks = split_menu_text(menu_text)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = [generate_chunk_descriptions(session, chunk, output_language) for chunk in chunks]
        completed = await asyncio.gather(*tasks)
        for result in completed:
            results.extend(result)

    return results

# ---------------------- Flask 路由入口 ----------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a JPG, JPEG, PNG, or WEBP image."}), 400

    file_data = file.read()
    menu_text = extract_text_from_image(file_data)

    if not menu_text.strip() or len(menu_text.strip()) < 10:
        return jsonify({"error": "OCR failed or returned invalid text"}), 400

    output_language = request.form.get("language", "English")

    # ⏳ 启动异步任务
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    menu_descriptions = loop.run_until_complete(get_menu_descriptions_async(menu_text, output_language))

    return jsonify({"menu": menu_descriptions})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

