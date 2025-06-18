from flask import Flask, request, jsonify, send_from_directory
import main
import pandas as pd

app = Flask(__name__)

# Загружаем модель и данные
model_embed = main.load_model()
questions_list, answers_list = main.load_qa_pairs("train_data.jsonl")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True)
    score = data.get("score", 0)
    subject = data.get("subject", "informatics")
    
    df = pd.read_excel("Направления с предпочтениями.xlsx")
    
    min_score_col = "Минимальный балл ЕГЭ(информатика)" if subject == "informatics" else "Минимальный балл ЕГЭ(физика)"
    
    available_directions = df[df[min_score_col] <= score].copy()
    
    available_directions["accessibility"] = score - available_directions["Проходной балл ЕГЭ(Бюджет)"]
    
    available_directions = available_directions.sort_values("accessibility", ascending=False)
    
    # Формируем ответ
    if not available_directions.empty:
        directions_list = []
        for _, row in available_directions.iterrows():
            directions_list.append({
                "name": row["Направление"],
                "min_score": row[min_score_col],
                "avg_score": row["Проходной балл ЕГЭ(Бюджет)"],
                "accessibility": row["accessibility"],
                "budget_places": row["Бюджетных мест"]
            })
        
        return jsonify({
            "status": "success",
            "score": score,
            "subject": subject,
            "available_directions": directions_list
        })
    else:
        return jsonify({
            "status": "no_directions",
            "message": "К сожалению, с вашим баллом доступных направлений не найдено.",
            "min_possible_score": df[min_score_col].min()
        })

@app.route("/")
def index():
    return send_from_directory(".", "index_v3.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    raw_q = data.get("question", "").strip()

    if not raw_q:
        return jsonify({
            "question": raw_q,
            "answer": "Пожалуйста, задайте вопрос.",
            "history": main.dialog_history[-3:],
            "close_chat": False
        })

    # Если это благодарность или прощание — обрабатываем сразу
    if main.is_thank_you(raw_q):
        answer = main.get_thank_you_response()
        return jsonify({
            "question": raw_q,
            "answer": answer,
            "history": main.dialog_history[-3:],
            "close_chat": False
        })

    if main.is_goodbye(raw_q):
        answer = main.get_goodbye_response()
        main.dialog_history.clear()
        return jsonify({
            "question": raw_q,
            "answer": answer,
            "history": [],
            "close_chat": True
        })

    if main.is_greeting(raw_q):

        words = raw_q.split()
        if len(words) <= 2:
            answer = main.get_greeting_response()
            return jsonify({
                "question": raw_q,
                "answer": answer,
                "history": main.dialog_history[-3:],
                "close_chat": False
            })

        for greet in main.GREETINGS_LIST:
            if raw_q.lower().startswith(greet):
                raw_q = raw_q[len(greet):].strip(",.?! ").strip()
                break

    # Ищем лучший ответ по эмбеддингам с порогом
    answer, similarity = main.get_best_answer_with_threshold(model_embed, raw_q, questions_list, answers_list)

    if similarity < main.SIMILARITY_THRESHOLD:
        answer = "Извините, я не понял вопрос. Пожалуйста, уточните."

    return jsonify({
        "question": raw_q,
        "answer": answer,
        "history": main.dialog_history[-3:],
        "close_chat": False
    })


if __name__ == "__main__":
    app.run(debug=True)
