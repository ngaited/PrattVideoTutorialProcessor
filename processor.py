"""
Written by: Ted Ngai, Executive Director, Pratt Advanced Technologies
Date: 2025/07/14
"""

import os
import json
import re
import subprocess
import requests # This is no longer used by transcribe_audio but may be used elsewhere.
import base64
import time
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
from imagehash import phash
from PIL import Image
import tempfile
import pytesseract
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("/home/nci/Data/VideoDocumentationExtractor/.env")

VIDEO_TUTORIAL_TRANSCRIPTION_PARAMS = {
    "max_new_tokens": 220,  # Standard length
    "num_beams": 1,  # Greedy decoding for speed
    "condition_on_prev_tokens": False,  # Better context for technical content
    "compression_ratio_threshold": 1.8,  # Higher threshold for repetitive/technical content
    "temperature": 0.0,  # Conservative temperature fallback for clarity
    "logprob_threshold": -0.6,  # More lenient for lower quality audio
    "no_speech_threshold": 0.6,  # Lower threshold to catch quiet speech
    "language": "english",  # Specify language (change as needed)
    "task": "transcribe",  # Transcription not translation
    "return_timestamps": True #   Useful for video tutorials to sync with video, other usage is "word"
}

class HybridVideoProcessor:
    def __init__(self, 
                whisper_base_url: str = "http://172.16.32.24:8009/",
                ocr_vlm_base_url: str = "https://api.openai.com/v1", # "http://172.19.50.61:8002/v1", # Your local VLM endpoint
                ocr_vlm_model: str = "gpt-4.1-mini", # "phi-4-multimodal", # The model name you are running
                openai_api_key: str = None):
        
        # NOTE: OCR client setup is from a previous version. This part uses Tesseract locally now.
        self.ocr_vlm_client = OpenAI(
            api_key=openai_api_key,
            base_url=ocr_vlm_base_url
        )
        self.ocr_vlm_model = ocr_vlm_model
        
        # GPT-4o client for final analysis, using the official OpenAI API
        self.vlm_client = OpenAI(api_key=openai_api_key)
        
        # Whisper endpoint
        self.whisper_base_url = whisper_base_url
    
    
    def transcribe_audio(self, video_path: str, custom_params: Dict = None) -> Dict[str, Any]:
        """
        Extracts audio from video and transcribes it using the Whisper API
        with parameters optimized for video tutorials.
        """
        print("🎵 Extracting audio for transcription...")
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
            audio_path = temp_audio_file.name
        
        try:
            # Extract audio from video with settings optimized for speech recognition
            subprocess.run([
                "ffmpeg", "-i", video_path,
                "-vn",  # No video
                "-acodec", "libmp3lame",  # Use MP3 codec
                "-ar", "16000",  # 16kHz sample rate (optimal for Whisper)
                "-ac", "1",  # Mono audio (sufficient for speech)
                "-q:a", "2",  # Good quality for MP3
                "-af", "highpass=f=80,lowpass=f=8000",  # Filter out very low/high frequencies
                audio_path,
                "-y"
            ], check=True, capture_output=True)

            print("Transcribing audio with Whisper API (optimized for video tutorials)...")
            
            # Use custom parameters if provided, otherwise use video tutorial defaults
            params = custom_params if custom_params is not None else VIDEO_TUTORIAL_TRANSCRIPTION_PARAMS
            
            print(f"Using transcription parameters: {json.dumps(params, indent=2)}")
            print(f"Submitting audio to Whisper API at {self.whisper_base_url}transcribe...")
            
            # Send to your Whisper API with parameters
            with open(audio_path, "rb") as audio_file:
                files = {"file": audio_file}
                data = {"params": json.dumps(params)}
                
                initial_response = requests.post(
                    f"{self.whisper_base_url}transcribe",
                    files=files,
                    data=data
                )
            
            if initial_response.status_code != 200:
                raise Exception(f"Whisper API submission error: {initial_response.text}")
            
            initial_data = initial_response.json()
            task_id = initial_data.get("task_id")
            if not task_id:
                raise Exception(f"Whisper API did not return a task_id: {initial_data}")
            
            print(f"Transcription job submitted. Task ID: {task_id}. Status: {initial_data.get('status')}")
            
            # Poll for the result
            start_time = time.time()
            max_poll_time = getattr(self, 'max_poll_time', 1800)  # 30 minutes default
            poll_interval = getattr(self, 'poll_interval', 10)  # 10 seconds default
            
            while True:
                if time.time() - start_time > max_poll_time:
                    raise Exception(f"Whisper API polling timed out for task {task_id} after {max_poll_time} seconds.")
                
                print(f"Checking status for task {task_id}...")
                status_response = requests.get(f"{self.whisper_base_url}status/{task_id}")
                
                if status_response.status_code != 200:
                    print(f"Warning: Could not get status for task {task_id}. Retrying. Error: {status_response.text}")
                    time.sleep(poll_interval)
                    continue

                status_data = status_response.json()
                current_status = status_data.get("status")
                
                print(f"Task {task_id} current status: {current_status}")

                if current_status == "success":
                    print(f"Transcription for task {task_id} completed successfully.")
                    result = status_data["result"]
                    
                    # Handle both timestamped and non-timestamped results
                    if isinstance(result, dict) and "chunks" in result:
                        # The API returns "chunks". We rename it to "segments" for consistency
                        # with the rest of our pipeline (e.g., align_segments).
                        return {
                            "text": result.get("text", ""),
                            "segments": result.get("chunks", []), # Renamed "chunks" to "segments"
                            "timestamps": True
                        }
                    elif isinstance(result, str):
                        # Plain text result
                        return {"text": result, "segments": [], "timestamps": False}
                    else:
                        # Fallback
                        return {"text": str(result), "segments": [], "timestamps": False}
                        
                elif current_status == "error":
                    error_message = status_data.get("error", "Unknown transcription error.")
                    raise Exception(f"Whisper API task {task_id} failed: {error_message}")
                elif current_status in ["queued", "processing"]:
                    time.sleep(poll_interval)
                else:
                    raise Exception(f"Whisper API task {task_id} returned unexpected status: {current_status}")

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg audio extraction failed!")
            print(f"STDERR: {e.stderr.decode('utf-8')}")
            raise
        finally:
            # Ensure the temporary file is always cleaned up
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    def extract_key_frames(self, video_path: str, output_dir: str) -> Dict[str, float]:
        """
        Extracts key frames optimized for screen capture videos with multiple detection methods.
        """
        print("🎬 Extracting key frames (optimized for screen capture)...")
        os.makedirs(output_dir, exist_ok=True)
        
        frame_to_timestamp_map = {}
        all_timestamps = []

        # Get video duration first
        try:
            duration_probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
            duration_result = subprocess.run(duration_probe_cmd, capture_output=True, text=True, check=True)
            duration = float(duration_result.stdout)
            print(f"   -> Video duration: {duration:.2f} seconds")
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"   -> Warning: Could not get video duration. Error: {e}")
            duration = 180  # Default to 3 minutes if we can't get duration

        # --- Method 1: Scene detection with lower threshold for screen content ---
        print("   -> Method 1: Scene detection (optimized for screen capture)...")
        scene_timestamps = []
        
        # Try multiple scene detection thresholds
        for threshold in [0.05, 0.1, 0.15]:  # Much lower thresholds for screen content
            ffprobe_command = [
                "ffprobe",
                "-v", "quiet",
                "-show_frames",
                "-of", "json",
                "-f", "lavfi",
                f"movie={video_path},select='gt(scene,{threshold})'"
            ]
            
            try:
                result = subprocess.run(ffprobe_command, capture_output=True, text=True, check=True)
                scene_data = json.loads(result.stdout)
                timestamps = [float(frame['pkt_pts_time']) for frame in scene_data.get('frames', [])]
                scene_timestamps.extend(timestamps)
                print(f"      -> Threshold {threshold}: Found {len(timestamps)} scene changes")
                
                # If we get a good number of scenes, break
                if len(set(scene_timestamps)) >= max(10, duration / 10):
                    break
                    
            except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                print(f"      -> Warning: Scene detection failed for threshold {threshold}. Error: {e}")
                continue

        # --- Method 2: Content-aware sampling (more frequent for screen recordings) ---
        print("   -> Method 2: Content-aware sampling...")
        
        # For screen recordings, sample more frequently
        if duration <= 60:  # 1 minute or less
            interval = 2.0  # Every 3 seconds
        elif duration <= 300:  # 5 minutes or less
            interval = 4.0  # Every 5 seconds
        elif duration <= 600:  # 10 minutes or less
            interval = 8.0  # Every 8 seconds
        else:
            interval = 10.0  # Every 10 seconds for longer videos
        
        periodic_timestamps = list(np.arange(0, duration, interval))
        print(f"   -> Adding {len(periodic_timestamps)} periodic frames (every {interval}s)")

        # --- Method 3: Motion-based detection for screen content ---
        print("   -> Method 3: Motion-based detection...")
        motion_timestamps = []
        
        # Use select filter to detect frames with significant changes
        motion_command = [
            "ffprobe",
            "-v", "quiet",
            "-show_frames",
            "-of", "json",
            "-f", "lavfi",
            f"movie={video_path},select='gte(t,0)*lt(t,{duration})*gt(abs(PREV_INPTS-INPTS),0.1)'"
        ]
        
        try:
            result = subprocess.run(motion_command, capture_output=True, text=True, check=True)
            motion_data = json.loads(result.stdout)
            motion_timestamps = [float(frame['pkt_pts_time']) for frame in motion_data.get('frames', [])]
            print(f"   -> Found {len(motion_timestamps)} motion-based changes")
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"   -> Warning: Motion detection failed. Error: {e}")

        # --- Method 4: Histogram-based detection for screen content ---
        print("   -> Method 4: Histogram-based detection...")
        histogram_timestamps = []
        
        # Sample frames and compare histograms
        sample_interval = max(1.0, duration / 100)  # Sample up to 100 points
        sample_times = np.arange(0, duration, sample_interval)
        
        try:
            # Extract histogram data for comparison
            hist_command = [
                "ffprobe",
                "-v", "quiet",
                "-show_frames",
                "-of", "json",
                "-f", "lavfi",
                f"movie={video_path},select='not(mod(n\\,{max(1, int(sample_interval * 30))}))',signalstats"
            ]
            
            result = subprocess.run(hist_command, capture_output=True, text=True, check=True)
            hist_data = json.loads(result.stdout)
            
            # Simple approach: if we have frame data, add some timestamps
            if hist_data.get('frames'):
                step = max(1, len(hist_data['frames']) // 20)  # Take every nth frame
                histogram_timestamps = [
                    float(frame['pkt_pts_time']) 
                    for i, frame in enumerate(hist_data['frames'][::step]) 
                    if 'pkt_pts_time' in frame
                ]
                print(f"   -> Found {len(histogram_timestamps)} histogram-based changes")
                
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"   -> Warning: Histogram detection failed. Error: {e}")

        # --- Combine all methods and remove duplicates ---
        all_timestamps = scene_timestamps + periodic_timestamps + motion_timestamps + histogram_timestamps
        
        # Remove duplicates and sort (with tolerance for near-duplicate timestamps)
        unique_timestamps = []
        tolerance = 0.5  # 0.5 second tolerance
        
        for ts in sorted(all_timestamps):
            if not any(abs(ts - existing) < tolerance for existing in unique_timestamps):
                unique_timestamps.append(ts)
        
        # Ensure we have a reasonable number of frames
        min_frames = max(20, int(duration / 8))  # At least 20 frames or 1 every 8 seconds
        max_frames = min(200, int(duration * 2))  # At most 200 frames or 2 per second
        
        if len(unique_timestamps) < min_frames:
            print(f"   -> Too few frames ({len(unique_timestamps)}), adding more periodic frames...")
            additional_interval = duration / min_frames
            additional_timestamps = list(np.arange(0, duration, additional_interval))
            
            for ts in additional_timestamps:
                if not any(abs(ts - existing) < tolerance for existing in unique_timestamps):
                    unique_timestamps.append(ts)
            
            unique_timestamps.sort()
        
        elif len(unique_timestamps) > max_frames:
            print(f"   -> Too many frames ({len(unique_timestamps)}), sampling down to {max_frames}...")
            # Keep evenly distributed frames
            indices = np.linspace(0, len(unique_timestamps) - 1, max_frames, dtype=int)
            unique_timestamps = [unique_timestamps[i] for i in indices]

        print(f"   -> Total unique timestamps to extract: {len(unique_timestamps)}")

        # --- Extract frames at each timestamp ---
        print(f"   -> Extracting {len(unique_timestamps)} frames...")
        
        for i, ts in enumerate(unique_timestamps):
            # Format timestamp for filename
            seconds = int(ts)
            milliseconds = int((ts - seconds) * 100)
            output_filename = f"frame_{seconds:05d}_{milliseconds:02d}.jpg"
            output_path = os.path.join(output_dir, output_filename)

            ffmpeg_extract_command = [
                "ffmpeg",
                "-ss", str(ts),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                "-y",  # Overwrite existing files
                output_path,
            ]
            
            try:
                subprocess.run(ffmpeg_extract_command, check=True, capture_output=True)
                if os.path.exists(output_path):
                    frame_to_timestamp_map[output_path] = ts
                    if (i + 1) % 10 == 0:  # Progress indicator
                        print(f"      -> Extracted {i + 1}/{len(unique_timestamps)} frames...")
            except subprocess.CalledProcessError as e:
                print(f"      -> Warning: Could not extract frame at {ts:.2f}s")
                continue

        print(f"✅ Extracted {len(frame_to_timestamp_map)} frames total")
        return frame_to_timestamp_map
    
    def deduplicate_frames(self, frame_paths: List[str], threshold: int = 5) -> List[str]:
        """Remove near-identical frames using perceptual hashing (O(N) sequential comparison)."""
        print("🔍 Deduplicating frames...")
        
        if not frame_paths:
            return []

        unique_frames = []
        last_kept_hash = None
        
        for frame_path in frame_paths:
            try:
                img = Image.open(frame_path)
                current_hash = phash(img)
                
                # Only compare to the last unique frame kept
                if last_kept_hash is None or abs(current_hash - last_kept_hash) > threshold:
                    unique_frames.append(frame_path)
                    last_kept_hash = current_hash
                else:
                    # This frame is a duplicate of the previous one, remove it
                    os.remove(frame_path)
                    
            except Exception as e:
                print(f"Error processing {frame_path}: {e}")
                continue
        
        print(f"Kept {len(unique_frames)} unique frames from {len(frame_paths)} total")
        return unique_frames
    
    def ocr_frames(self, frame_paths: List[str]) -> Dict[str, str]:
        """
        OCR each frame using a Vision Language Model (VLM) via an OpenAI-compatible API.
        """
        print(f"📝 Running OCR on {len(frame_paths)} frames using VLM ({self.ocr_vlm_model})...")
        ocr_results = {}
        
        for i, frame_path in enumerate(frame_paths):
            print(f"  -> OCR on frame {i+1}/{len(frame_paths)}: {os.path.basename(frame_path)}")
            try:
                # 1. Encode the image to Base64
                base64_image = self.encode_image(frame_path)
                
                # 2. Create the prompt for the VLM
                # This prompt is designed to get raw text output, similar to Tesseract.
                prompt_text = "Summarize the content of the image. If there are text present, perform OCR and exract all text content accurately. "

                # 3. Call the VLM API
                response  = self.ocr_vlm_client.responses.create(
                    model=self.ocr_vlm_model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt_text},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                                }
                            ]
                        }
                    ]
                )

                # 4. Store the result
                extracted_text = response .output_text
                ocr_results[frame_path] = extracted_text.strip()

            except Exception as e:
                print(f"   -> VLM OCR error for {frame_path}: {e}")
                ocr_results[frame_path] = "" # Maintain compatibility on error
        
        return ocr_results
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def align_segments(self, transcript: Dict, frame_to_timestamp_map: Dict[str, float], ocr_results: Dict[str, str]) -> List[Dict]:
        """Aligns transcript segments with frames, their timestamps, and their specific OCR text."""
        print("🔗 Aligning segments...")
        
        segments = []
        # Your client correctly gets the 'chunks' from the API response now.
        transcript_segments = transcript.get('segments', [])

        if not transcript_segments:
            print("Warning: No transcript segments found to align.")
            return []

        sorted_frame_paths = sorted(frame_to_timestamp_map.keys(), key=lambda p: frame_to_timestamp_map[p])

        for ts_segment in transcript_segments:
            ### FIX IS HERE ###
            # The API returns timestamps as a list: [start, end]
            # Check if the 'timestamp' key exists and is a list with 2 elements.
            if 'timestamp' not in ts_segment or not isinstance(ts_segment['timestamp'], list) or len(ts_segment['timestamp']) != 2:
                continue # Skip this segment if the timestamp data is malformed

            start_time, end_time = ts_segment['timestamp']
            # The API can sometimes return None for timestamps, so we must handle that.
            if start_time is None or end_time is None:
                continue
            
            segment_frames_data = []
            for path in sorted_frame_paths:
                timestamp = frame_to_timestamp_map.get(path)
                if timestamp is not None and start_time <= timestamp <= end_time:
                    segment_frames_data.append({
                        "path": path,
                        "timestamp": timestamp,
                        "ocr": ocr_results.get(path, "") # Get specific OCR for this frame
                    })
            
            segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'transcript': ts_segment['text'],
                # Store the detailed frame data, limiting to 3 for the VLM prompt
                'frames_data': segment_frames_data[:3] 
            })
        
        return segments
    
    def process_segment_with_vlm(self, segment: Dict) -> Dict:
        """Processes a single segment with GPT-4o and o4-mini, providing structured frame and OCR data."""
        
        # Build the text part of the prompt
        prompt_text = f"""
    Transcript ({segment['start_time']:.1f}s → {segment['end_time']:.1f}s):
    "{segment['transcript']}"
    """
        
        # Add structured information for each frame
        frame_details = []
        for frame_data in segment.get('frames_data', []):
            frame_details.append(
                f"\n--- Frame at {frame_data['timestamp']:.1f}s ---\n"
                f"OCR from this frame:\n{frame_data['ocr'] or 'No text detected.'}"
            )
        
        prompt_text += "\n".join(frame_details)
        
        prompt_text += """

    Tasks:
    1. Extract the main action(s) being demonstrated.
    2. Write as a clear instructional step, max 100 words, imperative voice.
    3. If the OCR text or image shows an exact command, include it verbatim in backticks.
    4. Use the images and their corresponding OCR to clarify any ambiguous references.

    Return JSON format:
    {"step": "<instruction text>", "code": ["<command1>", ...], "notes": "<additional context>"}
    """
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert technical writer creating step-by-step tutorials from screen recordings. You will be given a transcript, and a series of images with their corresponding OCR text."
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}]
            }
        ]
        
        # Add frame images to the user message
        for frame_data in segment.get('frames_data', []):
            frame_path = frame_data['path']
            if os.path.exists(frame_path):
                img_base64 = self.encode_image(frame_path)
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
        
        try:
            response = self.vlm_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"VLM processing error: {e}")
            return {"step": segment['transcript'], "code": [], "notes": f"Error processing: {e}"}
    
    def finalize_tutorial(self, step_results: List[Dict]) -> str:
        """Merge and clean up all steps into final tutorial"""
        print("📝 Finalizing tutorial...")
        
        prompt = f"""
You are a senior technical editor. Here are extracted steps from a video tutorial:

{json.dumps(step_results, indent=2)}

Tasks:
1. Merge into a coherent, numbered tutorial
2. Remove duplicates and redundant steps  
3. Add section headers when topics change
4. Ensure logical flow and clarity
5. Format code blocks properly

Return a well-structured tutorial in markdown format.
"""
        
        try:
            response = self.vlm_client.chat.completions.create(
                model="o4-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Finalization error: {e}")
            tutorial = "# Tutorial\n\n"
            for i, step in enumerate(step_results, 1):
                tutorial += f"## Step {i}\n\n{step.get('step', '')}\n\n"
                if step.get('code'):
                    tutorial += "```\n" + "\n".join(step['code']) + "\n```\n\n"
            return tutorial
    
    def process_video(self, video_path: str, output_dir: str = "output") -> str:
        """Main pipeline execution"""
        print(f"🚀 Starting hybrid video processing for: {video_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, "frames")
        
        try:
            transcript = self.transcribe_audio(video_path)
            with open(os.path.join(output_dir, "transcript.json"), "w") as f: json.dump(transcript, f, indent=2)

            # 1. Get the map of all frames to their timestamps
            all_frames_map = self.extract_key_frames(video_path, frames_dir)
            
            # 2. Deduplicate using the list of frame paths (the keys of the map)
            unique_frame_paths = self.deduplicate_frames(list(all_frames_map.keys()))
            
            # 3. Create a new map containing only the unique frames and their timestamps
            unique_frames_map = {path: all_frames_map[path] for path in unique_frame_paths if path in all_frames_map}
            
            # 4. Run OCR on the unique frames
            ocr_results = self.ocr_frames(unique_frame_paths)
            
            # 5. Align segments using the filtered map of unique frames
            segments = self.align_segments(transcript, unique_frames_map, ocr_results)
            with open(os.path.join(output_dir, "segments.json"), "w") as f: json.dump(segments, f, indent=2)
            
            print("🤖 Processing segments with GPT-4o...")
            step_results = []
            for i, segment in enumerate(segments):
                print(f"Processing segment {i+1}/{len(segments)} ({segment['start_time']:.1f}s - {segment['end_time']:.1f}s)")
                result = self.process_segment_with_vlm(segment)
                step_results.append(result)
            with open(os.path.join(output_dir, "steps.json"), "w") as f: json.dump(step_results, f, indent=2)
            
            final_tutorial = self.finalize_tutorial(step_results)
            
            output_file = os.path.join(output_dir, "tutorial.md")
            with open(output_file, "w", encoding="utf-8") as f: f.write(final_tutorial)
            
            print(f"✅ Tutorial saved to: {output_file}")
            return final_tutorial
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            raise

if __name__ == "__main__":
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
    
    
    processor = HybridVideoProcessor(
        openai_api_key=api_key,
        whisper_base_url=""
    )
    
    video_path = "your_tutorial_video.mp4" # <--- IMPORTANT: SET YOUR VIDEO PATH HERE
    output_dir = "output" # <--- IMPORTANT: SET YOUR OUTPUT DIRECTORY HERE
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}. Please update the path in the script.")

    tutorial = processor.process_video(video_path, output_dir)
    print("\n" + "="*50)
    print("Generated Tutorial:")
    print("="*50)
    print(tutorial)