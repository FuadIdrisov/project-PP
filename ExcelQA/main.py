import pandas as pd
from difflib import get_close_matches
from transformers import pipeline, AutoTokenizer

# --- Конфигурация ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 120
TEMPERATURE = 0.4

# --- Загрузка и кэширование данных ---
def load_data(file_path: str):
    """Читает Excel и возвращает dict: направление → факты."""
    df = pd.read_excel(file_path, sheet_name="Лист1", engine="openpyxl")
    df = df.fillna("не указано")
    # Преобразуем в словарь: { "Направление": { "баллы": ..., "предметы": ..., ... } }
    facts_map = {}
    for _, row in df.iterrows():
        facts_map[row["Направление"]] = {
            "баллы": row["Проходной балл ЕГЭ"],
            "предметы": row["Любимые предметы"],
            "профессии": row["Будущие профессии"],
            "специализация": row["Специализация"]
        }
    return list(facts_map.keys()), facts_map

# --- Загрузка модели один раз при старте ---
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    gen = pipeline(
        "text-generation",
        model=MODEL_NAME,
        tokenizer=tokenizer,
        device="cpu",
        torch_dtype="auto"
    )
    return gen, tokenizer

# --- Поиск направления с fuzzy-match ---
def extract_direction(question: str, directions: list[str]) -> str | None:
    q = question.lower()
    # Можно добавить предобработку q: исправление опечаток и т.п.
    match = get_close_matches(q, directions, n=1, cutoff=0.4)
    return match[0] if match else None

# --- Формирование короткой подсказки и вызов модели ---
def generate_response(
    gen_pipeline,
    tokenizer,
    direction: str,
    facts: dict,
    question: str
) -> str:
    # Собираем короткую строку с данными
    facts_str = (
        f"Проходной балл: {facts['баллы']}. "
        f"Предметы: {facts['предметы']}. "
        f"Профессии: {facts['профессии']}. "
        f"Специализация: {facts['специализация']}."
    )
    prompt = (
        f"Ты — консультант приёмной комиссии, которые отвечает на вопросы по поступления в ИРИТ-РТФ.\n"
        f"Направление: {direction}\n"
        f"{facts_str}\n"
        f"Вопрос: {question}\n"
        f"Ответь чётко и кратко, 2–3 предложения. "
        f"Не упоминай, что ты — модель ИИ."
    )

    # Генерируем текст, передавая только prompt и параметры
    out = gen_pipeline(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False
    )
    # Берём сгенерированный текст и чуть-чуть его «почистим»
    answer = out[0]["generated_text"].strip()
    # Если нужно, можно здесь вызвать функцию clean_response(answer)
    return answer

# --- Пример использования в консоли ---
if __name__ == "__main__":
    directions, facts_map = load_data("Направления.xlsx")
    gen_pipeline, tokenizer = load_model()

    while True:
        q = input("🎓 Ваш вопрос (или «выход»): ").strip()
        if q.lower() == "выход":
            break
        dir_found = extract_direction(q, directions)
        if not dir_found:
            print("❌ Направление не найдено.")
            continue
        answer = generate_response(
            gen_pipeline,
            tokenizer,
            dir_found,
            facts_map[dir_found],
            q
        )
        print("💡 Ответ:", answer)