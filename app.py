from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (good for testing; restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

# 👇 Redirect root (/) → /docs
@app.get("/")
def root():
    return RedirectResponse(url="/docs")


#  Updated endpoint with lazy import
@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    # Save uploaded audio locally
    audio_path = file.filename
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    #  Lazy import here (loads only when API is called)
    from meeting_model import run_meeting_pipeline

    # Run your existing pipeline
    output = run_meeting_pipeline(audio_path, HF_TOKEN, summary_length="detailed")

    return {
        "summary": output["summary"],
        "sentiment": output["sentiment"],
        "zip_file": output["zip"]
    }


# Optional:endpoint to download the ZIP
@app.get("/download/{zip_name}")
def download_zip(zip_name: str):
    return FileResponse(zip_name)
