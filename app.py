from flask import Flask, send_from_directory, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__, static_url_path='', static_folder='')

# Load the TinyBERT model and tokenizer
model_name = 'huawei-noah/TinyBERT_General_4L_312D'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Serve the index.html file
@app.route('/')
def index():
    return send_from_directory('', 'index.html')

# Define a route for handling POST requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if text:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)

        # Get the prediction (assuming a classification task)
        predictions = torch.argmax(outputs.logits, dim=1).item()

        return jsonify({'prediction': predictions})
    else:
        return jsonify({'error': 'No text provided'}), 400

# Serve static files like CSS
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('', filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
