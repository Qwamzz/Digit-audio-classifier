import sounddevice as sd
import numpy as np
import time
from audio_utils import load_model, predict_digit

def record_audio(duration=1, sample_rate=8000):
    """Record audio from microphone"""
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio.flatten()

def main():
    # Load trained model
    model = load_model('models/audio_cnn.pt')
    print("Model loaded successfully!")
    
    try:
        while True:
            input("\nPress Enter to record a 1-second audio clip (or Ctrl+C to exit)...")
            
            # Record audio
            start_time = time.time()
            waveform = record_audio()
            
            # Make prediction
            prediction = predict_digit(model, waveform, 8000)
            end_time = time.time()
            
            # Print results
            print(f"\nPredicted digit: {prediction}")
            print(f"Processing time: {(end_time - start_time):.3f} seconds")
            
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
