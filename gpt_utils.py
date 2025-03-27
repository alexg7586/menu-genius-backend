
import json
from config import OPENAI_API_KEY, MODEL

# Use GPT to generate structured dish descriptions from menu chunk
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
