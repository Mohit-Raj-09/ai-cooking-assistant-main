# list_models.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Please set GEMINI_API_KEY in your environment before running this script.")

genai.configure(api_key=API_KEY)

models = genai.list_models()
print("Available models:\n")

for model in models:
    name = model.name

    # Safely extract supported methods
    methods = []

    # Newer SDK
    if hasattr(model, "model_info") and hasattr(model.model_info, "supported_generation_methods"):
        methods = model.model_info.supported_generation_methods

    # Older SDK
    elif hasattr(model, "supported_methods"):
        methods = model.supported_methods

    print(name, "=>", methods)
