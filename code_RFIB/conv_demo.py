import os
import pickle
import torch
import soundfile as sf
from synthesis import build_model, wavegen

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Create output directory if it doesn't exist
results_dir = 'demo_wavs'
os.makedirs(results_dir, exist_ok=True)

# Build and load the vocoder model
model = build_model().to(device)
checkpoint = torch.load("assets/checkpoint_step001000000_ema.pth", map_location=torch.device(device))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Load demo.pkl
with open('assets/demo.pkl', 'rb') as f:
    demo_data = pickle.load(f)

# Process each demo entry
for entry in demo_data:
    # Entry format: [speaker_id, speaker_embedding, (mel_spectrogram, f0_quantized, num_frames, utterance_id)]
    speaker_id = entry[0]
    utterance_id = entry[2][3]
    # mel_spec is stored as a 2D NumPy array of shape (T, n_mels)
    mel_spec = entry[2][0]

    name = f"{speaker_id}_{utterance_id}"
    print(f"Generating waveform for: {name}")

    # Pass the mel-spectrogram directly to wavegen.
    # wavegen expects a NumPy array of shape (T, n_mels) so it can do: torch.FloatTensor(c.T).unsqueeze(0)
    waveform = wavegen(model, c=mel_spec)

    # Save the generated waveform as a .wav file
    output_wav_path = os.path.join(results_dir, f"{name}.wav")
    sf.write(output_wav_path, waveform, samplerate=16000)
    print(f"Saved {output_wav_path}")

print("All files generated!")
