import re
from flask import Flask, request, jsonify, send_from_directory
import main

app = Flask(__name__)

# Загружаем данные и модель
directions, facts_map = main.load_data("Направления.xlsx")
gen_pipeline, tokenizer = main.load_model()

def clean_question(raw: str) -> str:
    text = raw.replace('\r', '').strip()
    # Разбиваем на строки и отбрасываем помехи
    good_lines = [
        line.strip()
        for line in text.split('\n')
        if line and not re.match(r'^\s*(Проверка|Вариант|[-–—])', line, re.IGNORECASE)
    ]
    # Ищем в каждой строке первый вопросительный знак
    for line in good_lines:
        m = re.search(r'(.+\?)', line)
        if m:
            return m.group(1).strip()
    return good_lines[0] if good_lines else raw

@app.route("/")
def index():
    return send_from_directory(".", "index_v3.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    raw_q = data.get("question", "")
    if not raw_q.strip():
        return jsonify({"error": "Не указан вопрос"}), 400

    question = clean_question(raw_q)
    direction = main.extract_direction(question, directions)
    if not direction:
        return jsonify({"error": "Направление не найдено"}), 404

    answer = main.generate_response(
        gen_pipeline, tokenizer,
        direction, facts_map[direction],
        question
    )
    return jsonify({
        "question": question,
        "direction": direction,
        "answer": answer
    })

if __name__ == "__main__":
    app.run(debug=True)