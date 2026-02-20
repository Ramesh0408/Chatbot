import os
import json
import requests
from datetime import datetime
from PIL import Image
import pytesseract
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


# IMPORTANT: Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Load BLIP model once (performance optimization)
print("Loading image captioning model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

print("Image captioning model loaded successfully.")


def extract_text_from_image(
    image_input,
    output_folder="outputs",
    log_file="outputs/image_log.json"
):
    """
    Extracts:
    - Image caption (BLIP)
    - OCR text (Tesseract)
    - Saves JSON log
    - Returns structured result
    """

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load image
    try:
        if isinstance(image_input, str):

            # URL image
            if image_input.startswith("http"):
                image = Image.open(
                    requests.get(image_input, stream=True).raw
                ).convert("RGB")

                image_name = image_input.split("/")[-1]

            # Local image
            else:
                image = Image.open(image_input).convert("RGB")
                image_name = os.path.basename(image_input)

        else:
            # Streamlit uploaded file
            image = Image.open(image_input).convert("RGB")
            image_name = f"uploaded_{timestamp}.jpg"

    except Exception as e:
        return {
            "error": f"Failed to load image: {str(e)}"
        }


    # Generate image caption
    try:
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(**inputs)

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

    except Exception as e:
        caption = f"Caption generation failed: {str(e)}"


    # OCR text extraction
    try:
        extracted_text = pytesseract.image_to_string(image).strip()

    except Exception as e:
        extracted_text = f"OCR failed: {str(e)}"


    # Prepare result
    result = {

        "timestamp": timestamp,

        "image_name": image_name,

        "image_description": caption,

        "extracted_text": extracted_text

    }


    # Save JSON log safely
    try:

        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:

            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)

        else:
            data = []

    except:
        data = []


    data.append(result)


    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


    print(f"Processed image: {image_name}")
    print(f"Saved log: {log_file}")

    return result


# Standalone test mode
if __name__ == "__main__":

    image_path = input("Enter image path: ").strip().strip('"').strip("'")

    result = extract_text_from_image(image_path)

    print("\nResult:")
    print(json.dumps(result, indent=4, ensure_ascii=False))