import os
import pickle

def find_min_loss(folder_name):
    # Construct the path to the pickle file
    file_path = os.path.join("run", folder_name, "validation_losses.pkl")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None
    
    # Load the pickle file, which should contain a dictionary
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    # Ensure the dictionary has the expected keys
    if not ("iterations" in data and "loss_G" in data):
        print(f"Error: Expected keys 'iterations' and 'loss_G' not found in {file_path}. Found keys: {list(data.keys())}")
        return None
    
    iterations = data["iterations"]
    losses = data["loss_G"]
    
    # Check that both lists are of equal length
    if len(iterations) != len(losses):
        print(f"Error: Number of iterations ({len(iterations)}) and losses ({len(losses)}) do not match in {file_path}.")
        return None
    
    # Find the index of the minimum loss value
    min_index = losses.index(min(losses))
    min_iteration = iterations[min_index]
    min_loss = losses[min_index]
    
    return min_iteration, min_loss

def main():
    # List the folder names you want to process (update this list as needed)
    folders = ["samples_800k_IB123_NEW", "samples_800k_IB123_NEW_B0.1", "samples_800k_IB123_NEW_B0.5", "samples_800k_IB123_NEW_B2", "samples_800k_IB123_NEW_B10", "samples_800k_IB123_NEW_B100", "samples_800k_IB123_NEW_B50"]
    
    # Process each folder and output the iteration with the lowest loss
    for folder in folders:
        result = find_min_loss(folder)
        if result is not None:
            min_iter, min_loss = result
            print(f"In folder '{folder}', the lowest loss is {min_loss} at iteration {min_iter}.")

if __name__ == "__main__":
    main()

