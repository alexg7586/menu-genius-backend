
import logging
import asyncio
import aiohttp
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from utils import allowed_file, split_menu_text
from ocr_utils import extract_text_from_image_async
from gpt_utils import generate_chunk

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload endpoint
@app.post("/upload")
async def upload_menu(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        return {"error": "Invalid file type."}

    image_data = await file.read()

    async with aiohttp.ClientSession() as session:
        try:
            # Extract text from image using OCR
            ocr_text = await extract_text_from_image_async(session, image_data)
            # Split long menu into chunks
            chunks = split_menu_text(ocr_text)
            # Generate menu descriptions in parallel
            tasks = [generate_chunk(session, chunk) for chunk in chunks]
            results = await asyncio.gather(*tasks)
            # Flatten and return result
            menu_items = [item for sublist in results for item in sublist]
            return {"menu": menu_items}
        except Exception as e:
            logging.error("Upload processing failed", exc_info=True)
            return {"error": str(e)}
