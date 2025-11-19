# models.py
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, T5ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

# BLIP (image -> caption)
print("Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Flan-T5 (caption -> recipe)
print("Loading Flan-T5 model...")
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)

print("Models loaded on", device)
