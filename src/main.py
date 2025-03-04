import logging
import subprocess
import sys
from pathlib import Path

# Add the src directory to the PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_script(script_name):
    """Run a Python script."""
    logging.info(f"Running {script_name}...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Error running {script_name}: {result.stderr}")
    else:
        logging.info(f"Finished running {script_name}")

if __name__ == "__main__":
    # Run the data processing script
    run_script("src/data_processing.py")

    # Run the embedding store script
    run_script("src/embedding_store.py")

    # Run the search engine script
    run_script("src/search_engine.py")

    # Start the FastAPI application
    logging.info("Starting FastAPI application...")
    subprocess.run(["uvicorn", "src.chatbot:app", "--host", "0.0.0.0", "--port", "8000"])