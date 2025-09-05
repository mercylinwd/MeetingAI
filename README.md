
````markdown
# MeetingAI Backend

This project provides APIs for:
- Transcription
- Speaker diarization
- Summarization
- Sentiment analysis
- Exporting results as ZIP

## Setup

1. Clone the repo or copy files.  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
````

## Configure

1. Open `app.py`.
2. Replace the placeholder with your Hugging Face token:

   ```python
   HF_TOKEN = "your_huggingface_token_here"
   ```

## Run the Server

```bash
uvicorn app:app --reload --port 8000
```

* API root: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Endpoints

### POST `/process_audio/`

Upload an audio file.

**Response:**

```json
{
  "summary": "...",
  "sentiment": "...",
  "zip_file": "meeting_results.zip"
}
```

### GET `/download/{zip_name}`

Download the ZIP file.
Example:

```
http://127.0.0.1:8000/download/meeting_results.zip
```

```

