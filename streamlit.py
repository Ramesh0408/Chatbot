import streamlit as st
import os
import tempfile

# Import your modules
from stt_modern import transcribe_audio_file
from tts_modern import text_to_speech
from text_extraction import extract_text_from_image
from llm_module import process_with_llm


# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)


st.set_page_config(
    page_title="Multimodal AI System",
    page_icon="🤖",
    layout="centered"
)

st.title("Multimodal AI System")


# =====================================================
# Select Input Mode
# =====================================================

input_mode = st.selectbox(
    "Select Input Mode",
    ["Text", "Audio", "Image"]
)


# =====================================================
# Select Output Mode
# =====================================================

output_mode = st.selectbox(
    "Select Output Mode",
    ["Text"]
)


# =====================================================
# TEXT INPUT MODE
# =====================================================

if input_mode == "Text":

    user_input = st.text_area("Enter your text")

    if st.button("Process Text"):

        if user_input.strip() == "":
            st.warning("Please enter text")
        else:

            # LLM processing
            llm_result = process_with_llm(user_input)

            response_text = llm_result["response"]

            st.subheader("Response Text:")
            st.write(response_text)

            # If Audio output selected
            if output_mode == "Audio":

                audio_file = text_to_speech(response_text)

                st.subheader("Response Audio:")
                st.audio(audio_file)


# =====================================================
# AUDIO INPUT MODE
# =====================================================

elif input_mode == "Audio":

    audio_option = st.radio(
        "Choose Audio Input Method",
        ["Upload Audio", "Record Audio"]
    )

    audio_path = None


    # Upload Audio
    if audio_option == "Upload Audio":

        uploaded_file = st.file_uploader(
            "Upload audio file",
            type=["wav", "mp3", "m4a"]
        )

        if uploaded_file is not None:

            temp_path = os.path.join("outputs", uploaded_file.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            audio_path = temp_path


    # Record Audio
    elif audio_option == "Record Audio":

        recorded_audio = st.audio_input("Record your voice")

        if recorded_audio is not None:

            temp_path = os.path.join("outputs", "recorded_audio.wav")

            with open(temp_path, "wb") as f:
                f.write(recorded_audio.read())

            audio_path = temp_path


    if st.button("Process Audio"):

        if audio_path is None:
            st.warning("Please upload or record audio")
        else:

            # STT
            stt_result = transcribe_audio_file(audio_path)

            user_text = stt_result["recognized_text"]

            st.subheader("Recognized Text:")
            st.write(user_text)

            # LLM
            llm_result = process_with_llm(user_text)

            response_text = llm_result["response"]

            st.subheader("Response Text:")
            st.write(response_text)

            # TTS if selected
            if output_mode == "Audio":

                audio_file = text_to_speech(response_text)

                st.subheader("Response Audio:")
                st.audio(audio_file)


# =====================================================
# IMAGE INPUT MODE
# =====================================================

elif input_mode == "Image":

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg", "webp"]
    )

    if st.button("Process Image"):

        if uploaded_image is None:
            st.warning("Please upload image")
        else:

            image_path = os.path.join("outputs", uploaded_image.name)

            with open(image_path, "wb") as f:
                f.write(uploaded_image.read())


            # Extract text from image
            extraction_result = extract_text_from_image(image_path)

            extracted_text = extraction_result.get("ocr_text", "")

            st.subheader("Extracted Text:")
            st.write(extracted_text)


            # LLM response
            llm_result = process_with_llm(extracted_text)

            response_text = llm_result["response"]

            st.subheader("Response Text:")
            st.write(response_text)

