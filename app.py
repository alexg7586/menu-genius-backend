import os
import openai
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("GPT_MODEL", "gpt-4o")

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

MENU_FUNCTION = {
    "name": "add_menu_item",
    "description": "Extract a dish from the menu with name and description.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the dish"
            },
            "description": {
                "type": "string",
                "description": "Short description of the dish"
            }
        },
        "required": ["name", "description"]
    }
}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        return {"error": "Unsupported file type."}

    file_data = await file.read()
    base64_image = base64.b64encode(file_data).decode("utf-8")

    vision_response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an OCR assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract text from this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]},
        ]
    )
    extracted_text = vision_response.choices[0].message.content.strip()

    lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]
    results = []

    for line in lines:
        try:
            response = openai.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a food menu parser."},
                    {"role": "user", "content": line}
                ],
                tools=[
                    {
                        "type": "function",
                        "function": MENU_FUNCTION
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "add_menu_item"}}
            )
            call = response.choices[0].message.tool_calls[0].function
            args = json.loads(call.arguments)
            results.append({"name": args["name"], "description": args["description"]})
        except Exception as e:
            continue

    return {"menu": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 5001)))