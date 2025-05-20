from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_path = "./flan-t5-finetuned-vocab"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(
    "cuda" if torch.cuda.is_available() else "cpu")
model.eval()


def clean_response(text):
    text = text.strip()

    if text.startswith("To "):
        text = "to" + text[2:]

    text = re.sub(r"\.\.\.*\s*$", ".", text)

    if not text.endswith((".", "!", "?")):
        text += "."

    return text


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    inputs = tokenizer(prompt, return_tensors="pt",
                       padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": clean_response(decoded)})


if __name__ == "__main__":
    app.run(debug=True)
