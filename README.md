# Video Tutorial Processor

A Flask-based web application that automatically converts screen-capture tutorial videos into structured markdown documentation using AI-powered transcription, OCR, and content analysis.

## Features

- **Automated Video Processing**: Upload tutorial videos and receive markdown documentation via email
- **AI-Powered Analysis**: Uses OpenAI GPT-4o and Whisper for transcription and content extraction
- **Screen-Optimized Frame Extraction**: Multiple detection methods optimized for screen recordings
- **Intelligent OCR**: Extracts text from video frames using vision language models
- **Async Processing**: Celery-based task queue for handling long-running video processing jobs
- **Email Delivery**: Automatic email notifications with generated documentation attachments

## How It Works

1. **Upload**: Users upload video files through a web interface
2. **Audio Transcription**: Extract and transcribe audio using Whisper API
3. **Frame Analysis**: Extract key frames using scene detection, motion analysis, and periodic sampling
4. **OCR Processing**: Extract text content from frames using OpenAI vision models
5. **Segment Alignment**: Match transcript segments with relevant visual frames
6. **Tutorial Generation**: Create structured step-by-step instructions using GPT-4o
7. **Delivery**: Email the final markdown documentation to the user

## Requirements

- Python 3.8+
- Redis server
- FFmpeg
- OpenAI API key
- SMTP server access

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ngaited/PrattVideoTutorialProcessor.git
   cd PrattVideoTutorialProcessor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file with the following variables:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   WHISPER_BASE_URL=http://your-whisper-service:8009/
   CELERY_BROKER_URL=redis://localhost:6379/0
   CELERY_RESULT_BACKEND=redis://localhost:6379/0
   FLASK_SECRET_KEY=your_secret_key
   SENDER_EMAIL=your_email@domain.com
   SMTP_HOST=smtp.your-provider.com
   SMTP_PORT=587
   SMTP_USER=your_smtp_username
   SMTP_PASSWORD=your_smtp_password
   ```

4. **Start Redis server**:
   ```bash
   redis-server
   ```

## Usage

1. **Start the Celery worker** (in one terminal):
   ```bash
   celery -A celery_worker worker --loglevel=info
   ```

2. **Start the Flask application** (in another terminal):
   ```bash
   python app.py
   ```

3. **Access the web interface**:
   Open your browser and go to `http://localhost:5000`

4. **Upload a video**:
   - Select a screen-capture tutorial video
   - Enter your email address
   - Click "Submit Job"

5. **Wait for processing**:
   - Processing time varies with video length (3 min video ≈ 5 min processing)
   - You'll receive an email with the generated markdown documentation

## Best Practices for Video Input

- Use clear, detailed narration that describes each step
- Screen recordings work better than camera footage
- Ensure good audio quality for accurate transcription
- Keep videos focused on specific tutorial topics
- Avoid background noise and interruptions

## Project Structure

```
PrattVideoTutorialProcessor/
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
└── job_outputs/          # Processing results (auto-cleaned)
```

## API Dependencies

This project integrates with several external services:

- **OpenAI API**: For GPT-4o text generation and vision processing
- **Whisper API**: For audio transcription (requires separate service)
- **Redis**: For task queue management
- **SMTP Server**: For email delivery

## Processing Pipeline

The application follows a sophisticated multi-stage pipeline:

1. **Audio Extraction**: Uses FFmpeg to extract audio optimized for speech recognition
2. **Frame Extraction**: Combines multiple detection methods:
   - Scene change detection
   - Motion-based detection
   - Periodic sampling
   - Histogram analysis
3. **Frame Deduplication**: Removes similar frames using perceptual hashing
4. **OCR Processing**: Extracts text from frames using OpenAI vision models
5. **Segment Alignment**: Matches transcript segments with relevant frames
6. **Tutorial Generation**: Creates structured instructions using GPT-4o
7. **Finalization**: Merges all steps into coherent markdown documentation

## Configuration

### Transcription Parameters

The system uses optimized parameters for video tutorial transcription:
- Temperature: 0.0 (for clarity)
- Language: English
- Returns timestamps for segment alignment
- Optimized for technical content

### Frame Extraction Settings

- Adaptive sampling based on video duration
- Multiple detection thresholds for screen content
- Minimum/maximum frame limits to ensure quality
- Optimized for screen recordings vs. general video

## Error Handling

- Comprehensive error handling throughout the pipeline
- Email notifications for both success and failure cases
- Automatic cleanup of temporary files
- Detailed logging for debugging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the logs for detailed error information
- Verify all API keys and environment variables are correctly set

## Acknowledgments

- Built for Pratt Institute
- Uses OpenAI's GPT-4o and Whisper models
- Leverages FFmpeg for video processing
- Powered by Flask and Celery for web interface and task processing