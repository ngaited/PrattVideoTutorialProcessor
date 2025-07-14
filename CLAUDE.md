# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Flask-based video tutorial processor that converts screen-capture tutorial videos into markdown documentation. The system uses AI models (OpenAI GPT-4o, Whisper) to transcribe audio, extract text from frames via OCR, and generate step-by-step instructions.

## Development Commands

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Run Flask development server
python app.py
# Server runs on http://0.0.0.0:7870

# Run Celery worker (in separate terminal)
celery -A celery_worker worker --loglevel=info
```

### Environment Setup
The application requires a `.env` file with the following variables:
- `OPENAI_API_KEY` - OpenAI API key for GPT-4o and transcription
- `WHISPER_BASE_URL` - URL for Whisper transcription service
- `CELERY_BROKER_URL` - Redis URL for Celery (e.g., "redis://localhost:6379/0")
- `CELERY_RESULT_BACKEND` - Redis URL for Celery results
- `FLASK_SECRET_KEY` - Secret key for Flask sessions
- `SENDER_EMAIL`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` - Email configuration

## Architecture

### Core Components

1. **Flask Web Interface** (`app.py`):
   - Single-page upload interface at `/`
   - Handles video uploads and queues processing tasks
   - Uses unique filenames to prevent conflicts
   - Renders `templates/index.html` with Pratt Institute branding

2. **Celery Task Queue** (`celery_worker.py`):
   - Processes videos asynchronously using Redis broker
   - Sends results via email with markdown attachments
   - Handles cleanup of uploaded files and temporary outputs
   - Each job gets unique output directory in `job_outputs/`

3. **Video Processing Pipeline** (`processor.py`):
   - **HybridVideoProcessor** class coordinates the entire pipeline
   - Multi-stage frame extraction using scene detection, periodic sampling, motion detection, and histogram analysis
   - Whisper API integration for audio transcription with video tutorial optimizations
   - Frame deduplication using perceptual hashing
   - OCR via OpenAI GPT-4o-mini API calls
   - Segment alignment between transcript and frames
   - Final tutorial generation using GPT-4o

### Processing Pipeline Flow

1. **Audio Transcription**: Extract audio with FFmpeg, send to Whisper API with polling
2. **Frame Extraction**: Multi-method approach optimized for screen recordings
3. **Frame Deduplication**: Remove similar frames using perceptual hashing
4. **OCR Processing**: Extract text from frames using OpenAI vision models
5. **Segment Alignment**: Match transcript segments with relevant frames
6. **Tutorial Generation**: Process each segment with GPT-4o to create structured steps
7. **Finalization**: Merge all steps into coherent markdown tutorial

### Key Technical Details

- **Whisper Integration**: Uses custom transcription parameters optimized for video tutorials
- **Frame Extraction**: Combines scene detection, periodic sampling, motion detection, and histogram analysis
- **OCR Strategy**: Switched from local Tesseract to OpenAI vision models for better accuracy
- **Error Handling**: Comprehensive error handling with email notifications
- **Cleanup**: Automatic cleanup of uploaded files and temporary directories

### Directory Structure
```
/
├── app.py                 # Flask web application
├── celery_worker.py       # Celery task processing
├── processor.py           # Main video processing pipeline
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Upload interface
├── static/
│   └── asset/
│       └── Pratt_Institute_Logo.svg
├── uploads/              # Temporary video storage
└── job_outputs/          # Processing results (cleaned up after email)
```

## Important Notes

- The system is designed for screen capture tutorials with detailed narration
- Processing time scales with video length (3 min video ≈ 5 min processing)
- Uses Redis for Celery broker/backend - ensure Redis is running
- Email delivery requires SMTP configuration
- OpenAI API usage will incur costs based on video length and content
- Frame extraction is optimized for screen recordings, not general video content