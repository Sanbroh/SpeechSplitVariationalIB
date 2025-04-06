import os
import pickle
import matplotlib.pyplot as plt

def load_data_from_pkl(file_path):
    """Load and return data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_data(aggregated_data, title, xlabel, ylabel, output_path):
    """Plot aggregated data curves and save the graph."""
    plt.figure(figsize=(10, 6))
    for label, (x, y) in aggregated_data.items():
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(output_path, dpi=150)
    plt.show()

def main():
    # Define the folder containing the experiment subfolders
    run_folder = 'run'
    if not os.path.exists(run_folder):
        print(f"Folder '{run_folder}' does not exist in the current directory.")
        return

    # Ask the user for a list of subfolder names (comma separated)
    folder_names = ["samples_800k_d_STELLA", "samples_800k_IB123_NEW", "samples_800k_IB123_NEW_B0.1", "samples_800k_IB123_NEW_B0.5", "samples_800k_IB123_NEW_B2", "samples_800k_IB123_NEW_B10", "samples_800k_IB123_NEW_B100"]

    # Ask the user for a keyword to filter which pickle files to load
    keyword = input("Enter the keyword to filter pkl files (e.g., 'loss_data' or 'validation'): ")

    # Dictionary to hold aggregated data per folder
    # Each entry will be: {folder_name: (x_values, y_values)}
    aggregated_data = {}

    # Loop through each provided folder
    for folder_name in folder_names:
        folder_path = os.path.join(run_folder, folder_name)
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist. Skipping.")
            continue

        # Find all .pkl files in the folder that include the keyword in their name
        pkl_files = [f for f in os.listdir(folder_path)
                     if f.endswith('.pkl') and keyword in f]
        if not pkl_files:
            print(f"No pkl files containing keyword '{keyword}' found in folder '{folder_name}'.")
            continue

        all_x = []
        all_y = []
        for pkl_file in pkl_files:
            file_path = os.path.join(folder_path, pkl_file)
            data = load_data_from_pkl(file_path)
            # Determine which keys to use based on the data structure
            if 'iter_list' in data and 'loss_g_list' in data:
                x = data['iter_list']
                y = data['loss_g_list']
            elif 'iterations' in data and 'loss_G' in data:
                x = data['iterations']
                y = data['loss_G']
            else:
                print(f"Data structure in file '{pkl_file}' not recognized. Skipping.")
                continue

            all_x.append(x)
            all_y.append(y)

        if not all_x:
            continue

        # Aggregate data for the folder:
        # If more than one file was found, average the y values (assuming all x values are identical)
        aggregated_x = all_x[0]
        if len(all_y) == 1:
            aggregated_y = all_y[0]
        else:
            # Use the smallest length among curves to avoid index errors
            min_length = min(len(y) for y in all_y)
            aggregated_x = aggregated_x[:min_length]
            # Average y values at each index
            aggregated_y = [sum(y[i] for y in all_y) / len(all_y) for i in range(min_length)]
        
        aggregated_data[folder_name] = (aggregated_x, aggregated_y)
    
    if not aggregated_data:
        print("No data was aggregated. Please check your folder names and keyword.")
        return

    # Plot aggregated loss curves
    title = f"Aggregated Loss Curve ({keyword})"
    xlabel = "Iteration"
    ylabel = "Loss"
    output_path = f"aggregated_loss_plot_{keyword}.png"
    plot_data(aggregated_data, title, xlabel, ylabel, output_path)
    print(f"Aggregated plot saved to {output_path}")

if __name__ == '__main__':
    main()

