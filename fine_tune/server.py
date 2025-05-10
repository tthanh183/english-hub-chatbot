from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# Load model và tokenizer
model_path = "./final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load few-shot examples từ file
EXAMPLE_FILE = "examples.jsonl"
with open(EXAMPLE_FILE, "r", encoding="utf-8") as f:
    FEW_SHOT_EXAMPLES = [json.loads(line) for line in f]

app = Flask(__name__)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    raw_prompt = data.get("prompt", "").strip()
    if not raw_prompt:
        return jsonify({"error": "Missing prompt"}), 400

    # Tạo few-shot prompt động (lấy 2 ví dụ đầu tiên)
    few_shot_prompt = ""
    for example in FEW_SHOT_EXAMPLES[:2]:  # Hoặc bạn có thể chọn theo từ khóa
        few_shot_prompt += f"Instruction: {example['instruction']}\nResponse: {example['response']}\n\n"

    # Ghép câu hỏi của người dùng
    final_prompt = few_shot_prompt + f"Instruction: {raw_prompt}\nResponse:"

    # Tokenize và generate
    inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=False  # deterministic
    )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Cắt phần phản hồi sau "Response:"
    if "Response:" in full_output:
        response = full_output.split("Response:")[-1].strip()
        if not response:
            response = "Sorry, I don't have an example for that word yet."
    else:
        response = "Sorry, I couldn't understand the request."

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
