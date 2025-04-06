import pickle
import numpy as np

# Define the path to the pickle file with loss data.
loss_data_path = 'run/samples/loss_data.pkl'  # Replace with the correct path

# Define your sample step (the interval at which losses were recorded)
sample_step = 1000  # Adjust as needed

# Load the loss data.
with open(loss_data_path, 'rb') as f:
    loss_data = pickle.load(f)

# Extract iteration numbers and loss lists.
iter_list = np.array(loss_data['iter_list'])
loss_g_list = np.array(loss_data['loss_g_list'])
loss_p_list = np.array(loss_data['loss_p_list'])

# Filter: only keep entries where iteration is a multiple of sample_step and less than 100,000.
mask = (iter_list % sample_step == 0) & (iter_list < 100000)
iter_list_filtered = iter_list[mask]
loss_g_filtered = loss_g_list[mask]
loss_p_filtered = loss_p_list[mask]

# Find the index of the lowest loss for each model among the filtered entries.
min_idx_g = np.argmin(loss_g_filtered)
min_idx_p = np.argmin(loss_p_filtered)

# Get the corresponding iteration numbers and loss values.
best_iter_g = iter_list_filtered[min_idx_g]
best_loss_g = loss_g_filtered[min_idx_g]
best_iter_p = iter_list_filtered[min_idx_p]
best_loss_p = loss_p_filtered[min_idx_p]

print("Lowest Generator (G) loss: {:.8f} at iteration: {}".format(best_loss_g, best_iter_g))
print("Lowest Pitch Converter (P) loss: {:.8f} at iteration: {}".format(best_loss_p, best_iter_p))
