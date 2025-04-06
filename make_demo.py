import os
import pickle
import numpy as np

# Path to extracted spectrograms & F0s
spmel_dir = 'assets/spmel'
f0_dir = 'assets/raptf0'

# Define your speakers
speakers = ['your_source_speaker', 'your_target_speaker']  # Change these to match folder names
import os
import pickle
import numpy as np

# Path to extracted spectrograms & F0s
spmel_dir = 'assets/spmel'
f0_dir = 'assets/raptf0'

# Define your speakers
speakers = ['your_source_speaker', 'your_target_speaker']  # Change these to match folder names
speakers = ['p558', 'p547']

metadata = []
for speaker in speakers:
    spkid = np.zeros((82,), dtype=np.float32)  # Assuming 82 speakers max
    # Set a hardcoded speaker embedding
    if speaker == 'p558':
        spkid[1] = 1.0
    else:
        spkid[7] = 1.0

    speaker_folder = os.path.join(spmel_dir, speaker)
    files = sorted(os.listdir(speaker_folder))
    
    if not files:
        print(f"Error: No files found for speaker {speaker}")
        continue

    # List available utterances for the speaker.
    print(f"\nSpeaker: {speaker}")
    print("Available utterances:")
    for idx, sample_file in enumerate(files):
        print(f"  [{idx}]: {sample_file}")

    # Prompt the user to select one or more utterance indices.
    selection = input(f"Enter the indices of utterances to choose for speaker {speaker} (comma separated): ")
    try:
        selected_indices = [int(x.strip()) for x in selection.split(",") if x.strip() != ""]
    except ValueError:
        print("Invalid input. Skipping speaker.")
        continue

    for idx in selected_indices:
        if idx < 0 or idx >= len(files):
            print(f"Index {idx} is out of range for speaker {speaker}. Skipping this index.")
            continue
        sample_file = files[idx]
        spect_path = os.path.join(speaker_folder, sample_file)
        f0_path = os.path.join(f0_dir, speaker, sample_file)
        if os.path.exists(spect_path) and os.path.exists(f0_path):
            spect = np.load(spect_path)
            f0 = np.load(f0_path)
            metadata.append([speaker, spkid, (spect, f0, len(f0), sample_file)])
        else:
            print(f"Warning: Missing spect or f0 file for {sample_file} in speaker {speaker}")

# Save the demo metadata
with open('assets/demo.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("✅ `demo.pkl` created successfully!")
