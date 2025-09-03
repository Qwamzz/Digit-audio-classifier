import torch
import librosa
import numpy as np
from models.model import AudioCNN

def preprocess_audio(waveform, sample_rate, target_sample_rate=8000, target_length=8000):
    """Preprocess audio by resampling and padding/truncating"""
    # Resample if necessary
    if sample_rate != target_sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
    
    # Pad or truncate to target length
    if len(waveform) > target_length:
        waveform = waveform[:target_length]
    else:
        waveform = np.pad(waveform, (0, target_length - len(waveform)))
    
    # Normalize
    waveform = waveform / np.max(np.abs(waveform))
    return waveform

def extract_features(waveform, sample_rate=8000, n_mels=64, n_fft=1024, hop_length=512):
    """Extract mel spectrogram features from audio"""
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def predict_digit(model, waveform, sample_rate):
    """Process and predict digit from audio input"""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Preprocess audio
        waveform = preprocess_audio(waveform, sample_rate)
        
        # Extract features
        features = extract_features(waveform)
        
        # Convert to tensor
        features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)
        
        # Get prediction
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        
        return predicted.item()

def load_model(model_path):
    """Load trained model from path"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model
