
import base64
from config import OPENAI_API_KEY, MODEL

# Call OpenAI Vision model to extract text from image
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
