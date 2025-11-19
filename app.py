# app.py

import io
import os
from dotenv import load_dotenv
load_dotenv()

from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import torch

from transformers import BlipProcessor, BlipForConditionalGeneration

import google.generativeai as genai


# ------------------------------------
# 1. Load Gemini API Key from ENV
# ------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Set it before running the server.")

genai.configure(api_key=GEMINI_API_KEY)


# ------------------------------------
# 2. Initialize Gemini Model
# ------------------------------------

gemini_model = genai.GenerativeModel("gemini-flash-latest")

# Choose CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------
# 3. Load BLIP For Image Captioning
# ------------------------------------

print("Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
print("BLIP model loaded!")


# ------------------------------------
# 4. Create FastAPI App
# ------------------------------------

app = FastAPI(title="AI Cooking Assistant (Gemini + BLIP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_headers=["*"],
    allow_methods=["*"],
)


# ------------------------------------
# 5. Caption Image (BLIP)
# ------------------------------------

def caption_image(image: Image.Image) -> str:
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs, max_new_tokens=40)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()


# ------------------------------------
# 6. Generate Recipes (Gemini)
# ------------------------------------

def generate_recipes_from_caption(caption: str, count: int = 1) -> List[str]:
    prompt = f"""
You are a professional chef.

Based on this image description: "{caption}"

Generate {count} high-quality recipe variation(s).
Each recipe must follow this structure:

Title:
Servings:
Estimated time:
Ingredients:
- item (with quantity)
Steps:
1.
2.
3.
Notes/Variations:

Separate each recipe using "---".
"""

    response = gemini_model.generate_content(prompt)
    text = response.text

    # Split multiple recipes
    recipes = [part.strip() for part in text.split("---") if part.strip()]

    return recipes[:count]


# ------------------------------------
# 7. API Endpoint
# ------------------------------------

@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...), count: int = Form(1)):
    try:
        # Read image file
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Step 1 — BLIP Caption
        caption = caption_image(pil_img)

        # Step 2 — Gemini Recipes
        recipes = generate_recipes_from_caption(caption, count)

        return JSONResponse({
            "caption": caption,
            "recipes": recipes
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------
# 8. Health Check
# ------------------------------------

@app.get("/")
def home():
    return {"status": "ok", "message": "AI Cooking Assistant with Gemini is running!"}
