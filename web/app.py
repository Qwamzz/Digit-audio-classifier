from flask import Flask, request, jsonify
import torch
import librosa
import numpy as np
import os
from werkzeug.utils import secure_filename

from ..src.audio_utils import load_model, predict_digit

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
MODEL_PATH = '../models/audio_cnn.pt'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Load audio
        waveform, sr = librosa.load(filepath, sr=8000)
        digit = predict_digit(model, waveform, sr)
        os.remove(filepath)
        return jsonify({'digit': int(digit)})
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
