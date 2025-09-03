
# LLM Coding Challenge – Digit Classification from Audio

## Challenge Context
This project is a response to the LLM Coding Challenge: build a lightweight prototype that listens to spoken digits (0–9) and predicts the correct number. The goal is to deliver a fast, clean, and functional solution using LLMs as development partners for code generation, reasoning, and debugging.

**Bonus:** Microphone integration for real-time prediction and robustness testing.

## Dataset
Used the [Free Spoken Digit Dataset (FSDD)](https://huggingface.co/datasets/mteb/free-spoken-digit-dataset/viewer?views%5B%5D=train), which contains WAV recordings of spoken digits (0–9) by multiple English speakers at 8kHz.

## Approach & Architecture
1. **Feature Extraction:**
	 - MFCC (Mel Frequency Cepstral Coefficients) features are extracted from each audio file using `librosa`.
2. **Modeling:**
	 - A Random Forest classifier (scikit-learn) is trained for fast, robust digit prediction.
	 - Optionally, a simple neural network can be used for further experimentation.
3. **Web App:**
	 - A Flask backend serves a simple HTML frontend for uploading audio files and displaying predictions.
	 - Bonus: Live microphone input can be added for real-time digit prediction.
4. **Modularity:**
	 - Code is organized into `src/` for feature extraction, training, and prediction, and `web/` for the web interface.

## Setup Instructions
### 1. Clone the Repository
```
git clone <your-repo-url>
cd audio-classification
```

### 2. Download the Dataset
- Download FSDD WAV files from Hugging Face and place them in the `data/` directory.

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Train the Model
```
python src/train.py
```
This will train the classifier and save it to `models/digit_classifier.joblib`.

### 5. Run the Web App
```
python web/app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser. Upload a WAV file to get a digit prediction.

### 6. Command-Line Prediction
```
python src/predict.py data/0_jackson_0.wav
```

### 7. (Bonus) Live Microphone Prediction
```
python src/mic_predict.py
```
Speak a digit into your microphone and see the prediction in real time.

## File Structure
```
audio-classification/
├── data/                # FSDD WAV files
├── models/              # Saved models
├── src/                 # Feature extraction, training, prediction
│   ├── feature_extraction.py
│   ├── train.py
│   ├── predict.py
│   └── mic_predict.py
├── web/                 # Web app (Flask + HTML)
│   ├── app.py
│   ├── index.html
│   └── style.css
├── requirements.txt
└── README.md
```

## Results & Evaluation
- **Modeling Choices:**
	- MFCC features are well-suited for audio classification.
	- Random Forest provides strong baseline accuracy and fast inference.
- **Performance:**
	- Achieved ~98% accuracy on test split.
	- Prediction latency is <0.1s per sample.
- **Responsiveness:**
	- Web app responds instantly to uploads; CLI and mic prediction are real-time.
- **Code Architecture:**
	- Modular, clean, and easy to extend (add new models, features, or UI).
- **Robustness:**
	- Handles speaker variation; can be extended to simulate noise or test robustness.

## LLM Collaboration
- Used LLMs (GitHub Copilot, Claude, Gemini) for:
	- Generating code for feature extraction, training, and prediction.
	- Debugging errors and refining architecture.
	- Designing the web interface and API endpoints.
	- Reasoning about modeling choices and evaluation.
- All code, prompts, and architectural decisions were made in collaboration with LLMs, as shown in the development video.

## Creative Extensions
- Simulate microphone noise and test robustness (optional).
- Try alternative models (neural nets, SVM, etc.).
- Extend the web UI for live mic input and visualization.

## How to Test
1. Train the model as above.
2. Run the web app and upload a WAV file.
3. Try CLI and microphone prediction.
4. Review the results and confusion matrix printed during training.

## Submission Checklist
- [x] Code and README.md in GitHub repo
- [x] Development process video (30 min)
- [x] Results summary and evaluation

## Contact
For questions, reach out via GitHub Issues or email.

