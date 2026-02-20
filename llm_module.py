from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
from datetime import datetime

# Load model once (important)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium"
)

# Chat history storage
chat_history_ids = None


def process_with_llm(user_input, mode="text", log_file="outputs/llm_log.json"):
    global chat_history_ids

    os.makedirs("outputs", exist_ok=True)

    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt"
    )

    with torch.no_grad():
        chat_history_ids = model.generate(
            new_input_ids if chat_history_ids is None
            else torch.cat([chat_history_ids, new_input_ids], dim=-1),

            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    response = tokenizer.decode(
        chat_history_ids[:, new_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    entry = {
        "timestamp": timestamp,
        "user_input": user_input,
        "response": response,
        "mode": mode
    }

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

    return entry