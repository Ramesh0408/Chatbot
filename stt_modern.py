import os
import json
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import pipeline
from datetime import datetime


# Load Whisper model once (important)
print("Loading speech-to-text model...")

stt_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base"
)

print("Speech-to-text model loaded.")


# ==========================================================
# FUNCTION 1: Record from microphone and transcribe
# Used by Streamlit
# ==========================================================

def record_and_transcribe(
    seconds=5,
    output_folder="outputs",
    log_file="outputs/speech_log.json"
):

    os.makedirs(output_folder, exist_ok=True)

    fs = 16000

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"recording_{timestamp}.wav"

    filepath = os.path.join(output_folder, filename)

    print("Recording... Speak now")

    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)

    sd.wait()

    write(filepath, fs, audio)

    print("Recording complete")

    result = stt_pipeline(filepath)

    text = result["text"]

    entry = {

        "timestamp": timestamp,

        "audio_file": filepath,

        "recognized_text": text

    }

    save_log(entry, log_file)

    return entry


# ==========================================================
# FUNCTION 2: Transcribe uploaded audio file
# Used by FastAPI
# ==========================================================

def transcribe_audio_file(
    filepath,
    log_file="outputs/speech_log.json"
):

    if not os.path.exists(filepath):

        return {"error": "Audio file not found"}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = stt_pipeline(filepath)

    text = result["text"]

    entry = {

        "timestamp": timestamp,

        "audio_file": filepath,

        "recognized_text": text

    }

    save_log(entry, log_file)

    return entry


# ==========================================================
# FUNCTION 3: Save JSON log safely
# ==========================================================

def save_log(entry, log_file):

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if os.path.exists(log_file):

        try:

            with open(log_file, "r", encoding="utf-8") as f:

                data = json.load(f)

        except:

            data = []

    else:

        data = []

    data.append(entry)

    with open(log_file, "w", encoding="utf-8") as f:

        json.dump(data, f, indent=4)


# ==========================================================
# Standalone testing
# ==========================================================

if __name__ == "__main__":

    result = record_and_transcribe()

    print("\nResult:")
    print(result)