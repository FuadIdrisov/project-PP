from sentence_transformers import SentenceTransformer, util
import json
import random

dialog_history = []

# Порог сходства для ответа
SIMILARITY_THRESHOLD = 0.5

GREETINGS_LIST = ["привет", "здравствуй", "добрый день", "хай", "hello", "hi"]

def load_model():
    return SentenceTransformer("trained_embed_model")

def load_qa_pairs(jsonl_path):
    questions, answers = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if len(obj["texts"]) == 2:
                questions.append(obj["texts"][0])
                answers.append(obj["texts"][1])
    return questions, answers

def get_best_answer_with_threshold(model, question, questions_list, answers_list):
    question_embedding = model.encode(question, convert_to_tensor=True)
    db_embeddings = model.encode(questions_list, convert_to_tensor=True)

    similarities = util.cos_sim(question_embedding, db_embeddings)[0]
    top_idx = similarities.argmax().item()
    top_sim = similarities[top_idx].item()

    best_answer = answers_list[top_idx]
    dialog_history.append((question, best_answer))
    return best_answer, top_sim

def is_greeting(text: str) -> bool:
    return any(g in text.lower() for g in GREETINGS_LIST)

def is_thank_you(text: str) -> bool:
    thank_you_phrases = [
        "спасибо", "спасибо большое", "огромное спасибо", "спс", "благодарю",
        "спасибо тебе", "благодарствую", "спасибо за помощь", "thanks", "thank you",
        "спасибо огромное", "ты помог", "ты молодец", "спасибо за ответ", "спасибо, бот",
        "пасиб", "пасибки", "благодарю за всё", "респект", "благодарочка"
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in thank_you_phrases)

def is_goodbye(text: str) -> bool:
    goodbye_phrases = [
        "пока", "до свидания", "увидимся", "бай", "bye",
        "до встречи", "прощай", "всего доброго", "счастливо", "чао",
        "мне пора", "я пошёл", "ещё увидимся", "увидимся позже", "я ухожу",
        "выключаюсь", "до завтра", "всё, пока", "бот, пока", "счастливо оставаться"
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in goodbye_phrases)

def get_greeting_response():
    return "Привет! Чем могу помочь?"

def get_thank_you_response():
    if not dialog_history:
        return random.choice([
            "Я ещё ничего не сделал, но спасибо!",
            "Спасибо, но я пока не помог!",
            "Благодарю, хотя я ещё не участвовал в диалоге!",
            "Спасибо за вежливость, но я жду вашего вопроса!",
            "Приятно слышать благодарность, но давайте начнём диалог!",
        ])
    else:
        return random.choice([
            "Пожалуйста! Обращайтесь ещё!",
            "Всегда рад помочь!",
            "Не стоит благодарности!",
            "Рад был помочь!",
        ])

def get_goodbye_response():
    if not dialog_history:
        return random.choice([
            "Прощайте!",
            "До свидания!",
            "Всего доброго!",
            "Пока!",
        ])
    else:
        return random.choice([
            "До свидания! Удачи!",
            "Был рад помочь! До новых встреч!",
            "До встречи!",
        ])
