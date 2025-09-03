import torch
from torch.utils.data import Dataset
from audio_utils import preprocess_audio, extract_features

class SpokenDigitDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        waveform = item['audio']['array']
        sample_rate = item['audio']['sampling_rate']
        
        # Preprocess audio
        waveform = preprocess_audio(waveform, sample_rate)
        
        # Extract features
        features = extract_features(waveform)
        
        # Convert to tensor
        features = torch.FloatTensor(features).unsqueeze(0)
        label = torch.tensor(int(item['label']))
        
        return features, label
