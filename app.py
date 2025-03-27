import os
import openai
import base64
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

openai.api_key = os.getenv("OPENAI_API_KEY")

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_image(file_data):
    base64_image = base64.b64encode(file_data).decode("utf-8")

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an OCR assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract text from this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]
    )

    return response.choices[0].message.content

def get_menu_descriptions(menu_text, output_language="English"):
    prompt = f"""
Here's a menu text:

{menu_text}

For each dish, return its translated name and a short, rich description (ingredients, flavor, preparation).
Use only {output_language}, and output a valid JSON array: [{{"name": "...", "description": "..."}}]
"""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a food expert."},
            {"role": "user", "content": prompt}
        ]
    )

    menu_response = response.choices[0].message.content.strip("```json").strip("```")

    return json.loads(menu_response)

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
    if not menu_text.strip():
        return jsonify({"error": "OCR failed to extract text"}), 400

    output_language = request.form.get("language", "English")
    menu_descriptions = get_menu_descriptions(menu_text, output_language)

    return jsonify({"menu": menu_descriptions})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
