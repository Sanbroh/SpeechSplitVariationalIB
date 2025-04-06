import os
import pickle

# Define the directory where your speaker folders are stored.
# For example, if your raw audio is stored in assets/wavs/
wav_dir = 'assets/wavs'

# List all speaker directories (assuming each speaker is a subdirectory).
speakers = [d for d in os.listdir(wav_dir) if os.path.isdir(os.path.join(wav_dir, d))]
print("Found speakers:")
for speaker in speakers:
    print(f" - {speaker}")

# Create the speaker-to-gender mapping dictionary.
spk2gen = {}

print("\nEnter gender for each speaker (M or F):")
for speaker in speakers:
    # Get user input for gender
    gender = input(f"Gender for speaker '{speaker}': ").strip().upper()
    # Validate input; if invalid, default to 'M'
    if gender not in ['M', 'F']:
        print(f"Invalid input for {speaker}. Defaulting to 'M'.")
        gender = 'M'
    spk2gen[speaker] = gender

# Define output path for the pickle file.
output_path = os.path.join('assets', 'spk2gen.pkl')

# Save the dictionary to the pickle file.
with open(output_path, 'wb') as f:
    pickle.dump(spk2gen, f)

print(f"\nSpeaker-to-gender mapping saved to {output_path}.")
