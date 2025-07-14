# celery_worker.py
import os
import shutil
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from celery import Celery
from dotenv import load_dotenv

# Import your main processing class
from processor import HybridVideoProcessor

# Load environment variables from .env file
load_dotenv()

# Configure Celery
celery = Celery(
    __name__,
    broker=os.environ.get("CELERY_BROKER_URL"),
    backend=os.environ.get("CELERY_RESULT_BACKEND")
)
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

def send_email_with_attachment(recipient_email: str, subject: str, body: str, file_content: str, filename: str):
    """Sends an email with a markdown file attachment."""
    try:
        msg = MIMEMultipart()
        msg['From'] = os.environ['SENDER_EMAIL']
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Attach the markdown file
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(file_content.encode('utf-8'))
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
        msg.attach(part)

        # Connect to SMTP server and send
        with smtplib.SMTP(os.environ['SMTP_HOST'], int(os.environ['SMTP_PORT'])) as server:
            server.starttls()
            server.login(os.environ['SMTP_USER'], os.environ['SMTP_PASSWORD'])
            server.send_message(msg)
        print(f"Successfully sent email to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email to {recipient_email}: {e}")


@celery.task(name="process_video_task")
def process_video_task(video_path: str, user_email: str):
    """
    Celery task to process the video and email the result.
    This function runs on the worker, not the web server.
    """
    print(f"Starting video processing task for {video_path} for user {user_email}")
    
    # Each job gets its own unique output directory
    job_id = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("job_outputs", job_id)

    try:
        # Initialize your processor
        processor = HybridVideoProcessor(
            openai_api_key=os.environ['OPENAI_API_KEY'],
            whisper_base_url=os.environ['WHISPER_BASE_URL']
        )
        
        # Run the main processing pipeline
        final_tutorial = processor.process_video(video_path, output_dir)
        
        # Send success email with the markdown file
        subject = "Your Video Tutorial Documentation is Ready!"
        body = "Hello,\n\nAttached is the markdown documentation generated from your video.\n\nThank you for using our service!"
        send_email_with_attachment(user_email, subject, body, final_tutorial, "tutorial.md")

    except Exception as e:
        print(f"An error occurred during video processing for {video_path}: {e}")
        # Send failure email
        subject = "Error Processing Your Video Tutorial"
        body = f"Hello,\n\nWe encountered an error while processing your video:\n\n{str(e)}\n\nPlease check the video file and try again."
        send_email_with_attachment(user_email, subject, body, f"Error: {e}", "error.txt")

    finally:
        # Clean up the uploaded video file and the job's output directory
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"Cleaned up video file: {video_path}")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Cleaned up output directory: {output_dir}")