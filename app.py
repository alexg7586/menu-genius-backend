import os
import openai
import base64
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")  # 默认更快的模型

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------- OCR 提取菜单文本 ----------------------
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

# ---------------------- 菜单智能分组 ----------------------
def split_menu_text(menu_text):
    lines = [line.strip() for line in menu_text.splitlines() if line.strip()]
    total = len(lines)

    # 智能决定每组条数
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

# ---------------------- 单个小段调用 GPT 生成 ----------------------
def generate_chunk_descriptions(chunk_text, output_language="English"):
    prompt = f"""
Here is a part of a restaurant menu:

{chunk_text}

Translate each dish name into {output_language}, and provide a short but rich description (ingredients, flavor, prep).
Use only {output_language}, and return result in this JSON format:
[
  {{
    "name": "...",
    "description": "..."
  }}
]
"""
    try:
        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a food expert."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content.strip("```json").strip("```")
        return json.loads(content)
    except Exception as e:
        return [{"name": "Error", "description": f"Failed to process chunk: {str(e)}"}]

# ---------------------- 合并并发执行所有 chunk ----------------------
def get_menu_descriptions(menu_text, output_language="English"):
    chunks = split_menu_text(menu_text)
    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(generate_chunk_descriptions, chunk, output_language) for chunk in chunks]
        for future in as_completed(futures):
            try:
                result = future.result()
                results.extend(result)
            except Exception as e:
                results.append({"name": "Error", "description": f"Unexpected error: {str(e)}"})

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
    menu_descriptions = get_menu_descriptions(menu_text, output_language)

    return jsonify({"menu": menu_descriptions})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

