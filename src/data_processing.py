import os
import logging
from pathlib import Path
from moviepy.editor import VideoFileClip
import whisper

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load Whisper model once at the start
logging.info("Loading Whisper model...")
whisper_model = whisper.load_model("base", device="cpu")  # Force Whisper to use CPU
logging.info("Whisper model loaded.")

def video_to_audio(video_path: str, audio_path: str):
    """Extracts audio from a video file and saves it as an MP3."""
    try:
        logging.info(f"Extracting audio from {video_path} to {audio_path}")
        video = VideoFileClip(video_path)

        if video.audio is None:
            logging.error(f"No audio track found in {video_path}")
            return  # Exit function gracefully
        
        video.audio.write_audiofile(audio_path, codec='mp3')
        logging.info(f"Audio extracted and saved to {audio_path}")
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")

def audio_to_text(audio_path: str) -> str:
    """Transcribes audio using Whisper and returns text."""
    try:
        logging.info(f"Transcribing audio from {audio_path}")
        result = whisper_model.transcribe(audio_path)
        transcript = result['text'].strip()

        if not transcript:
            logging.warning("No speech detected in the audio.")
            return "No speech detected."
        
        logging.info("Transcription completed")
        return transcript
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        return "Transcription failed due to an error."

def save_transcript(transcript: str, transcript_path: str):
    """Saves transcribed text to a file."""
    try:
        logging.info(f"Saving transcript to {transcript_path}")
        with open(transcript_path, 'w', encoding="utf-8") as f:
            f.write(transcript)
        logging.info(f"Transcript saved to {transcript_path}")
    except Exception as e:
        logging.error(f"Error saving transcript: {e}")

if __name__ == "__main__":
    # Example usage
    video_path = "data/raw_videos/example.mkv"  # Update this to match your video file name
    audio_path = "data/audio_files/example.mp3"
    transcript_path = "data/transcripts/example.txt"

    # Ensure directories exist
    Path(os.path.dirname(audio_path)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(transcript_path)).mkdir(parents=True, exist_ok=True)

    # Process video to audio
    video_to_audio(video_path, audio_path)

    # Transcribe audio to text
    transcript = audio_to_text(audio_path)

    # Save transcript
    save_transcript(transcript, transcript_path)