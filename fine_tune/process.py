import csv
import requests
import time
import os

API_KEY = ""
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"


def get_definition_and_example(word):
    prompt = (
        f'Give a short, simple English definition and a natural example sentence for the word "{word}". '
        f'Format:\nDefinition: ...\nExample: ...'
    )
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    try:
        response = requests.post(API_URL, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Parse definition and example
        definition, example = "", ""
        for line in text.splitlines():
            if line.lower().startswith("definition:"):
                definition = line.split(":", 1)[1].strip()
            elif line.lower().startswith("example:"):
                example = line.split(":", 1)[1].strip()
        return definition, example
    except Exception as e:
        print(f"Error for word '{word}': {e}")
        if "429" in str(e):
            return None, None
        return "", ""


def main():
    input_path = "words.txt"
    output_path = "output.csv"

    # Đọc các từ đã làm rồi (nếu có)
    done_words = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_words.add(row["word"])

    # Đọc danh sách từ
    with open(input_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    # Nếu file chưa có header, ghi header
    if not os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["word", "definition", "example"])
            writer.writeheader()

    for word in words:
        if word in done_words:
            print(f"Skip: {word}")
            continue
        definition, example = get_definition_and_example(word)
        if definition is None and example is None:
            print("Quota exceeded, please run again tomorrow.")
            break
        with open(output_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["word", "definition", "example"])
            writer.writerow(
                {"word": word, "definition": definition, "example": example})
        print(f"{word}: {definition} | {example}")
        time.sleep(1)


if __name__ == "__main__":
    main()
