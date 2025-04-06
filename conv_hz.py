import os
import librosa
import soundfile as sf

def resample_wav(input_path, target_sr=16000):
    """Resamples a WAV file to the target sample rate (16kHz) and overwrites it."""
    audio, sr = librosa.load(input_path, sr=None)  # Load with original sample rate
    if sr != target_sr:
        print(f"Converting {input_path} from {sr} Hz to {target_sr} Hz")
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sf.write(input_path, audio_resampled, target_sr)
    else:
        print(f"{input_path} is already {target_sr} Hz, skipping.")

def convert_all_wavs_in_folder(root_folder):
    """Finds and converts all WAV files in the given folder (including subfolders) to 16kHz."""
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)
                resample_wav(file_path)

# Change this to your WAV folder (e.g., 'assets/wavs')
root_folder = 'assets/wavs'
convert_all_wavs_in_folder(root_folder)

print("All WAV files have been converted to 16kHz!")
