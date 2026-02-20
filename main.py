from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil
import os

from stt_modern import transcribe_audio_file
from tts_modern import text_to_speech
from text_extraction import extract_text_from_image
from llm_module import generate_response

app = FastAPI(title="Multimodal Intelligent AI")

os.makedirs("outputs", exist_ok=True)


@app.get("/")
def home():
    return {"message": "Multimodal Intelligent API running"}


# TEXT INPUT → LLM → optional TTS
@app.post("/chat")
async def chat(
    text: str = Form(...),
    output_type: str = Form("text")  # text / audio / both
):

    response = generate_response(text)

    result = {
        "input": text,
        "response": response
    }

    if output_type in ["audio", "both"]:
        audio_path = text_to_speech(response)
        result["audio_file"] = audio_path

    return result


# VOICE INPUT → STT → LLM → optional TTS
@app.post("/voice")
async def voice_chat(
    file: UploadFile = File(...),
    output_type: str = Form("text")
):

    filepath = os.path.join("outputs", file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    stt_result = transcribe_audio_file(filepath)

    user_text = stt_result["recognized_text"]

    response = generate_response(user_text)

    result = {
        "input_voice_text": user_text,
        "response": response
    }

    if output_type in ["audio", "both"]:
        audio_path = text_to_speech(response)
        result["audio_file"] = audio_path

    return result


# IMAGE INPUT → OCR → LLM → optional TTS
@app.post("/image")
async def image_chat(
    file: UploadFile = File(...),
    output_type: str = Form("text")
):

    filepath = os.path.join("outputs", file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_result = extract_text_from_image(filepath)

    image_text = image_result["image_description"] + " " + image_result["extracted_text"]

    response = generate_response(image_text)

    result = {
        "image_analysis": image_result,
        "response": response
    }

    if output_type in ["audio", "both"]:
        audio_path = text_to_speech(response)
        result["audio_file"] = audio_path

    return result


@app.get("/audio/{filename}")
def get_audio(filename: str):

    path = os.path.join("outputs", filename)

    if os.path.exists(path):
        return FileResponse(path, media_type="audio/wav")

    return {"error": "File not found"}