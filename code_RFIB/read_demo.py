import pickle
import numpy as np

# Path to the demo file
demo_path = 'assets/demo.pkl'

# Load the demo data
with open(demo_path, 'rb') as f:
    demo_data = pickle.load(f)

# Loop over entries and print details
for idx, entry in enumerate(demo_data):
    speaker_id, speaker_embedding, features = entry
    mel_padded, f0_quantized, num_frames, utt_id = features
    print(f"Entry {idx}:")
    print(f"  Speaker ID: {speaker_id}")
    print(f"  Utterance ID: {utt_id}")
    print(f"  Speaker embedding shape: {np.array(speaker_embedding).shape}")
    print(f"  Mel spectrogram shape: {mel_padded.shape}")
    print(f"  F0 one-hot shape: {f0_quantized.shape}")
    print(f"  Number of valid frames: {num_frames}\n")
