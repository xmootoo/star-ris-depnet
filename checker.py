from scipy.io import loadmat
import numpy as np

# Load the .mat file
dataset_15_normal = loadmat(r"C:\Users\xmoot\Desktop\VSCode\Mehrazin's Code\Datasets\Star\15\ga_normal_wc_TEST.mat")
dataset_15_star = loadmat(r"C:\Users\xmoot\Desktop\VSCode\Mehrazin's Code\Datasets\Star\15\ga_star_wc_TEST.mat")
dataset_17_normal = loadmat(r"C:\Users\xmoot\Desktop\VSCode\Mehrazin's Code\Datasets\Star\17\ga_normal_wc_TEST.mat")
dataset_17_star = loadmat(r"C:\Users\xmoot\Desktop\VSCode\Mehrazin's Code\Datasets\Star\17\ga_star_wc_TEST.mat")

# List of keys you're interested in
keys_of_interest = ['x_ga', 'compute_time', 'rate_ga', 'constraint_vio']

files  = [dataset_15_normal, dataset_15_star, dataset_17_normal, dataset_17_star]

for file in files:
    print("\n")
    for key in keys_of_interest:
        if key in file:
            
            # Compute and print the mean value
            mean_value = np.mean(file[key])
            print(f"Mean {key}: {mean_value}")
        else:
            print(f"{key} not found in the .mat file.")
