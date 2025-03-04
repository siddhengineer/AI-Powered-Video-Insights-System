import os
import logging
import pickle
from pathlib import Path
from moviepy.editor import VideoFileClip
import whisper
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load Whisper model once at the start
logging.info("Loading Whisper model...")
whisper_model = whisper.load_model("base", device="cpu")  # Using CPU for stability
logging.info("Whisper model loaded.")

def video_to_audio(video_path: str, audio_path: str):
    """Extracts audio from a video file and saves it as an MP3."""
    try:
        logging.info(f"Extracting audio from {video_path} to {audio_path}")
        video = VideoFileClip(video_path)

        if video.audio is None:
            logging.error(f"No audio track found in {video_path}")
            return None  # Exit function gracefully
        
        video.audio.write_audiofile(audio_path, codec='mp3')
        logging.info(f"Audio extracted and saved to {audio_path}")
        return audio_path
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")
        return None

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

def split_text_into_chunks(text: str, chunk_size=50):
    """Splits text into chunks of approximately `chunk_size` words while preserving sentence boundaries."""
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split at sentence boundaries
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        word_count = sum(len(chunk.split()) for chunk in current_chunk)
        
        if word_count >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logging.info(f"Generated {len(chunks)} text chunks.")  # Debugging

    return chunks

def save_transcript_and_chunks(transcript: str, transcript_path: str, chunks_path: str):
    """Saves transcribed text and its processed chunks."""
    try:
        logging.info(f"Saving transcript to {transcript_path}")
        with open(transcript_path, 'w', encoding="utf-8") as f:
            f.write(transcript)
        logging.info(f"Transcript saved to {transcript_path}")

        # Split into chunks and save
        text_chunks = split_text_into_chunks(transcript)
        with open(chunks_path, 'wb') as f:
            pickle.dump(text_chunks, f)
        logging.info(f"Processed text chunks saved to {chunks_path}")
    except Exception as e:
        logging.error(f"Error saving transcript or chunks: {e}")

if __name__ == "__main__":
    # Example usage
    video_path = "data/raw_videos/example.mkv"
    audio_path = "data/audio_files/example.mp3"
    transcript_path = "data/transcripts/example.txt"
    chunks_path = "data/processed_chunks.pkl"

    # Ensure directories exist
    Path(os.path.dirname(audio_path)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(transcript_path)).mkdir(parents=True, exist_ok=True)

    # Process video to audio
    extracted_audio = video_to_audio(video_path, audio_path)
    
    if extracted_audio:
        # Transcribe audio to text
        transcript = audio_to_text(audio_path)

        # Save transcript and chunks
        save_transcript_and_chunks(transcript, transcript_path, chunks_path)
