import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer
from datetime import datetime
import json
import os

tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")


def text_to_speech(text, output_folder="outputs", log_file="outputs/tts_log.json"):

    os.makedirs(output_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = f"tts_{timestamp}.wav"
    audio_path = os.path.join(output_folder, audio_file)

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    sf.write(audio_path, output.squeeze().numpy(), 16000)

    entry = {
        "timestamp": timestamp,
        "text": text,
        "audio_file": audio_path
    }

    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
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

    print("Saved:", audio_path)

    # FIX: return dictionary instead of string
    return entry