from flask import Flask, render_template, request, jsonify
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

app = Flask(__name__)

# Load model and tokenizer
MODEL_PATH = "model_checkpoints/best_model"  # Update with your model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("Loading default model...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    data = request.json
    formal_text = data.get('text', '')
    
    if not formal_text:
        return jsonify({"error": "No text provided"}), 400
    
    # Prepare input
    input_text = f"convert to informal: {formal_text}"
    input_encoding = tokenizer(
        input_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_encoding.input_ids,
            attention_mask=input_encoding.attention_mask,
            max_length=128,
            num_beams=4,
            early_stopping=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # Decode output
    informal_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({"informal_text": informal_text})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)    
    print("Web demo ready! Open http://127.0.0.1:5000/ in your browser.")
    app.run(debug=True)
