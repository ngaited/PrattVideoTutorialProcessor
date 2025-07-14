# app.py
import os
import uuid
from flask import Flask, request, render_template, flash, redirect, url_for
from dotenv import load_dotenv

# Import the celery task from the worker file
from celery_worker import process_video_task

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a-very-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('No video file part', 'error')
        return redirect(request.url)
    
    file = request.files['video']
    email = request.form.get('email')

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(request.url)

    if not email:
        flash('No email address provided', 'error')
        return redirect(request.url)

    if file:
        # Generate a unique filename to avoid conflicts
        filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        # This is the magic part:
        # Instead of processing here, we send it to the Celery worker.
        # .delay() is the non-blocking call to queue the task.
        process_video_task.delay(video_path, email)
        
        flash('Success! Your video is being processed. You will receive an email with the results shortly.', 'success')
        return redirect(url_for('index'))

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7870)
