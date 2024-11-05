import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA

# Functions
def compute_sam(a, b):
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.matmul(a, b.T) / np.matmul(a_norm, b_norm.T), a_min=-1.0, a_max=1.0))

# Main path settings
base_dir = r"C:\Users\USUARIO\Documents\UIS_Cacao\Base_Datos_Cacao\ALL_VIS_special_1"
out_dir = "built_datasets"
os.makedirs(out_dir, exist_ok=True)

# Variables for processing
efficiency_range = [500, 850]  # nanometers
entrega1_white_scaling = 21.0
conveyor_belt_samples = 200
angle_error = 0.25
max_num_samples = 1000

cocoa_batch_sizes = [100, 200, 300, 500, 1000]
plot_num_samples = 500

# Load wavelengths and set efficiency threshold
wavelength_data = loadmat(os.path.join(base_dir, 'wavelengths_VIS.mat'))
wavelengths = next(v for k, v in wavelength_data.items() if not k.startswith('__')).squeeze()
efficiency_threshold = (efficiency_range[0] <= wavelengths) & (wavelengths <= efficiency_range[1])
wavelengths = wavelengths[efficiency_threshold]

# Dictionary of paths to data files (adapted from your setup)
full_cocoa_paths = {
    # Add your dataset paths here similar to your original script
}

# Process each batch size
for cocoa_batch_size in cocoa_batch_sizes:
    print(f"Processing batch size: {cocoa_batch_size}")

    cocoa_bean_dataset = []
    label_dataset = []

    cocoa_bean_batch_mean_dataset = []
    label_batch_mean_dataset = []

    for subset_name, lot_filenames in full_cocoa_paths.items():
        print(f"Processing {subset_name} subset")

        for label, filenames in lot_filenames.items():
            # Load the data for each lot
            lot_data = loadmat(os.path.join(base_dir, filenames['L']))
            lot = next(v for k, v in lot_data.items() if not k.startswith('__'))
            lot = lot[:, efficiency_threshold]

            # (Load and process white and black references similar to above)

            # Spectral Angle Mapper and other preprocessing steps
            selected_indices = np.random.choice(lot.shape[0], min(lot.shape[0], max_num_samples), replace=False)
            selected_cocoa = lot[selected_indices, :]

            # Compute reflectance and log transform
            selected_cocoa_reflectance = np.log(1 / selected_cocoa)
            cocoa_bean_dataset.append(selected_cocoa_reflectance)
            label_dataset.append(np.ones(selected_cocoa_reflectance.shape[0]) * label)

            # Batch mean calculations
            batch_means = [selected_cocoa_reflectance[np.random.choice(selected_cocoa_reflectance.shape[0], cocoa_batch_size, replace=False), :].mean(axis=0)
                           for _ in range(cocoa_batch_samples)]
            cocoa_bean_batch_mean_dataset.append(np.array(batch_means))
            label_batch_mean_dataset.append(np.ones(len(batch_means)) * label)

    # PCA Analysis
    full_cocoa_bean_dataset = np.concatenate(cocoa_bean_dataset, axis=0)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(full_cocoa_bean_dataset)

    # Plotting PCA results
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(np.unique(label_dataset)):
        indices = [idx for idx, val in enumerate(label_dataset) if val == label]
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=f'Label {label}')
    plt.title(f'PCA of Cocoa Data (Batch Size: {cocoa_batch_size})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Adjust further for your specific datasets, file paths, and processing logic.
