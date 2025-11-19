# ai-cooking-assistant

# error
No Python at "C:\Users\sumit\...


Step 1 — Deactivate current venv
deactivate

Step 2 — Delete the broken venv folder
rm -r .\venv

Step 3 — Create a new venv using your correct Python
where python

You will get something like:
C:\Users\SAHIL\AppData\Local\Programs\Python\Python312\python.exe

Now create venv:
python -m venv venv

Step 4 — Activate venv
.\venv\Scripts\activate

Step 5 — Install dependencies
pip install -r requirements.txt

Step 6 — Run the script
python list_models.py

pip install dotenv

pip install google-generativeai

pip install --upgrade pip setuptools wheel

pip install fastapi uvicorn transformers torch pillow python-multipart "numpy<2.0"

pip uninstall -y torch torchvision torchaudio

pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 




# run backend and model 
# open both in different terminal
uvicorn app:app --reload --host 0.0.0.0 --port 8000
python -m http.server 5500

# link of website
http://localhost:5500/sample.html#home