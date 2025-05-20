# !pip install transformers datasets accelerate - q

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import json
import pandas as pd
import shutil
shutil.rmtree("/kaggle/working/results", ignore_errors=True)


df = pd.read_csv("/kaggle/input/output/output.csv")  # Đổi path nếu cần

definition_templates = [
    "What does {word} mean?",
    "Can you define {word}?",
    "Tell me the meaning of {word}.",
    "What's the meaning of {word}?",
    "Explain the word {word}."
]
example_templates = [
    "Can you give me an example of {word}?",
    "Show me a sentence using {word}.",
    "How is {word} used in a sentence?",
    "Give an example sentence with {word}.",
    "Use {word} in a sentence."
]

data = []
for _, row in df.iterrows():
    word = str(row["word"])
    definition = str(row["definition"]) if pd.notnull(
        row["definition"]) else ""
    example = str(row["example"]) if pd.notnull(row["example"]) else ""
    for temp in definition_templates:
        instruction = temp.format(word=word)
        response = f"{word.capitalize()} means {definition}."
        data.append({"instruction": instruction, "response": response})
    for temp in example_templates:
        instruction = temp.format(word=word)
        response = example
        data.append({"instruction": instruction, "response": response})

with open("vocab_data.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")


torch.cuda.empty_cache()
dataset = load_dataset('json', data_files={'train': 'vocab_data.jsonl'})

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))


def preprocess_function(examples):
    input_ids_list = []
    labels_list = []
    max_length = 128
    for prompt, response in zip(examples['instruction'], examples['response']):
        prompt_ids = tokenizer(
            prompt, truncation=True, max_length=64, add_special_tokens=False)['input_ids']
        response_ids = tokenizer(
            response, truncation=True, max_length=64, add_special_tokens=False)['input_ids']
        input_ids = prompt_ids + [tokenizer.eos_token_id] + \
            response_ids + [tokenizer.eos_token_id]
        labels = [-100] * (len(prompt_ids) + 1) + \
            response_ids + [tokenizer.eos_token_id]
        input_ids = input_ids[:max_length] + \
            [tokenizer.pad_token_id] * (max_length - len(input_ids))
        labels = labels[:max_length] + [-100] * (max_length - len(labels))
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    return {
        'input_ids': input_ids_list,
        'labels': labels_list,
        'attention_mask': [[1 if id != tokenizer.pad_token_id else 0 for id in ids] for ids in input_ids_list]
    }


train_dataset = dataset['train'].map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

training_args = TrainingArguments(
    output_dir="./final_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    learning_rate=5e-5,
    fp16=True,
    save_strategy="no",
    logging_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")



model = AutoModelForCausalLM.from_pretrained("./final_model")
tokenizer = AutoTokenizer.from_pretrained("./final_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def ask_model(prompt):
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100,
                             do_sample=False, temperature=0.7)
    print(tokenizer.decode(
        outputs[0], skip_special_tokens=True).replace(formatted, ""))


ask_model("What is the meaning of football")

