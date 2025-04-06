import os
import soundfile as sf

rootDir = 'assets/wavs'
# Dictionary to store folder names that contain one or more non-16000 Hz files,
# along with a mapping of file names to their sampling rates.
folders_with_non16000 = {}

for subdir, _, files in os.walk(rootDir):
    # Temporary dict to hold non-16000 Hz files for this folder.
    non_16000_files = {}
    
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(subdir, file)
            data, fs = sf.read(file_path)
            print(f"{file}: Sampling Rate = {fs} Hz")
            # Check if sampling rate is not 16000 Hz.
            if fs != 16000:
                non_16000_files[file] = fs
    
    # If there is at least one file in the current folder with a non-16000 Hz rate,
    # store the folder and its non-conforming files.
    if non_16000_files:
        folders_with_non16000[subdir] = non_16000_files

print("\nFolders with files that do not have 16000 Hz:")
for folder, files_dict in folders_with_non16000.items():
    print(f"\nFolder: {folder}")
    for filename, rate in files_dict.items():
        print(f"  {filename}: {rate} Hz")
        break
