from flask import Flask, request, jsonify, render_template
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Инициализация модели
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# Явно устанавливаем pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    try:
        df = pd.read_excel(file, engine='openpyxl')
        data = df.where(pd.notnull(df), None).to_dict(orient="records")
        return jsonify({"data": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json.get('data', [])[:10]
        question = request.json.get('question', '')[:150]
        
        prompt = f"Анализ данных:\n{data}\n\nВопрос: {question}\nОтвет:"
        
        # Токенизация с явным указанием attention_mask
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Декодируем, пропуская спецтокены
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = answer.split("Ответ:")[-1].strip()
        
        return jsonify({"answer": answer})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
