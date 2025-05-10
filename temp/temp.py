import sqlite3
import ollama

# Kết nối tới cơ sở dữ liệu SQLite (hoặc tạo mới nếu chưa tồn tại)
conn = sqlite3.connect('vocabulary.db')
cursor = conn.cursor()

# Tạo bảng từ vựng nếu chưa tồn tại
cursor.execute('''
CREATE TABLE IF NOT EXISTS vocabulary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL,
    definition TEXT NOT NULL,
    example TEXT NOT NULL
);
''')
conn.commit()

# Thêm một số từ vựng vào cơ sở dữ liệu (chỉ thêm một lần)


def insert_sample_data():
    sample_words = [
        ('serendipity', 'the occurrence of events by chance in a happy or beneficial way',
         'It was pure serendipity that they met in the park.'),
        ('ephemeral', 'lasting for a very short time',
         'The beauty of the sunset was ephemeral, disappearing in minutes.'),
        ('melancholy', 'a deep, persistent sadness',
         'There was a feeling of melancholy in the air as the summer ended.')
    ]
    cursor.executemany('''
    INSERT INTO vocabulary (word, definition, example)
    VALUES (?, ?, ?);
    ''', sample_words)
    conn.commit()

# Hàm truy vấn từ vựng theo từ khóa


def search_word(word):
    cursor.execute('''
    SELECT * FROM vocabulary WHERE word LIKE ?;
    ''', ('%' + word + '%',))
    result = cursor.fetchall()
    return result

# Hàm trả lời câu hỏi từ dữ liệu


def answer_question(query):
    results = search_word(query)
    if results:
        answer = ""
        for result in results:
            word, definition, example = result[1], result[2], result[3]
            answer += f"Word: {word}\nDefinition: {definition}\nExample: {example}\n\n"
        return answer
    else:
        return "Sorry, I couldn't find information on that word."

# Chèn dữ liệu mẫu vào (chỉ cần làm một lần)
# insert_sample_data()


# Tương tác với người dùng
while True:
    input_query = input('Ask me a word (type "exit" to quit): ')
    if input_query.lower() == "exit":
        break

    response = answer_question(input_query)
    print(response)

    # Gọi Ollama để tạo câu trả lời chi tiết hơn nếu cần
    instruction_prompt = f'''
    You are a helpful chatbot that help user learning new vocabulary. Please answer the question about vocabulary only.
    If the question is not related to vocabulary, respond with "I can only help with vocabulary-related questions."
    If the question is related to vocabulary, provide a detailed answer.
    Your task is to provide a detailed answer based on the context provided.
    The context is a database of vocabulary words, their definitions, and example sentences.
    Use only the following context to answer the question:
    {response}
    '''

    stream = ollama.chat(
        model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
        messages=[{'role': 'system', 'content': instruction_prompt},
                  {'role': 'user', 'content': input_query}],
        stream=True
    )

    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print("\n" + "="*50)  # Separator for clarity
