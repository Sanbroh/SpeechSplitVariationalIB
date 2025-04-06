import pickle

# Define the path to your spk2gen.pkl file
pkl_path = 'assets/spk2gen.pkl'

# Open and load the pickle file
with open(pkl_path, 'rb') as f:
    spk2gen = pickle.load(f)

# Print the loaded dictionary
print("Speaker to Gender Mapping:")
count = 0
count_f = 0
for speaker_id, gender in spk2gen.items():
    print(f"Speaker: {speaker_id}, Gender: {gender}")
    count += 1
    if gender == 'F':
    	count_f += 1
print("TOTAL:", count)
print("M:", count-count_f)
print("F:", count_f)
