import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat

from sklearn.decomposition import PCA
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


# functions

def compute_sam(a, b):
    assert a.ndim == 2, "a must have two dimensions, if you only have one, please add an new dimension in the first place"
    assert b.ndim == 2, "b must have two dimensions, if you only have one, please add an new dimension in the first place"

    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.matmul(a, b.T) / np.matmul(a_norm, b_norm.T), a_min=-1.0, a_max=1.0))


# set main paths

base_dir = r"C:\Users\USUARIO\Documents\UIS_Cacao\Base_Datos_Cacao\ALL_VIS_special_1"
out_dir = os.path.join("built_datasets")
os.makedirs(out_dir, exist_ok=True)

# set variables

efficiency_range = [500, 850]  # nanometers
entrega1_white_scaling = 21.0  # white / this
conveyor_belt_samples = 200  # for sam metric
angle_error = 0.25  # angle error between conveyor belt and cocoa signatures
max_num_samples = 1000  # selected samples from lot with higher sam

#cocoa_batch_size = 1000  # guillotine methodology
cocoa_batch_samples = 1000  # number of batch samples

plot_num_samples = 500
debug = False
debug_pca = True

# set path to cocoa dataset

full_cocoa_paths = {
    'train': {0: {"L": "L1F60H096R290324C070524VISTRAIFULL.mat",
                  "B": "blanco.mat",
                  "N": "negro.mat",
                  "E": "Entrega 1"},
              1: {"L": "L2F66H144R310324C070524VISTRAIFULL.mat",
                  "B": "blanco.mat",
                  "N": "negro.mat",
                  "E": "Entrega 1"},
              2: {"L": "L2F73H144E270624C240724VISTRAIFULL.mat",
                  "B": "B2F73H144E270624C240724VISTRAIFULL.mat",
                  "N": "N2F73H144E270624C240724VISTRAIFULL.mat",
                  "E": "Entrega 2"},
              3: {"L": "L3F84H192R020424C090524VISTRAIFULL.mat",
                  "B": "blanco.mat",
                  "N": "negro.mat",
                  "E": "Entrega 1"},
              4: {"L": "L1F85H110E270624C240724VISTRAIFULL.mat",
                  "B": "B1F85H110E270624C240724VISTRAIFULL.mat",
                  "N": "N1F85H110E270624C240724VISTRAIFULL.mat",
                  "E": "Entrega 2"},
              5: {"L": "L4F92H264R130424C090524VISTRAIFULL.mat",
                  "B": "blanco.mat",
                  "N": "negro.mat",
                  "E": "Entrega 1"},
              6: {"L": "L3F94H216E270624C240724VISTRAIFULL.mat",
                  "B": "B3F94H216E270624C240724VISTRAIFULL.mat",
                  "N": "N3F94H216E270624C240724VISTRAIFULL.mat",
                  "E": "Entrega 2"},

              7: {"L": "L4F96H252E270624C240724VISTRAIFULL.mat",
                  "B": "B4F96H252E270624C240724VISTRAIFULL.mat",
                  "N": "N4F96H252E270624C240724VISTRAIFULL.mat",
                  "E": "Entrega 2"},

              }
}

# load wavelengths

wavelengths = next(
    v for k, v in loadmat(os.path.join(base_dir, 'wavelengths_VIS.mat')).items() if not k.startswith('__')).squeeze()

# set threshold between 400 and 900 nm

efficiency_threshold = (efficiency_range[0] <= wavelengths) & (wavelengths <= efficiency_range[1])
wavelengths = wavelengths[efficiency_threshold]

# load and build dataset

cocoa_batch_sizes = [100, 200, 300, 500, 1000]
all_distances = []

for cocoa_batch_size in cocoa_batch_sizes:
    print(f"Processing with batch size: {cocoa_batch_size}")

    cocoa_bean_dataset = []
    label_dataset = []

    cocoa_bean_batch_mean_dataset = []
    label_batch_mean_dataset = []


    for subset_name, lot_filenames in full_cocoa_paths.items():
        print(f"Processing {subset_name} subset")

        cocoa_bean_dataset = []
        label_dataset = []

        cocoa_bean_batch_mean_dataset = []
        label_batch_mean_dataset = []

        for label, lot_filename in lot_filenames.items():
            print(f"Processing {lot_filename['E']} - {lot_filename['L']}")

            white = next(
                v for k, v in loadmat(os.path.join(base_dir, lot_filename['B'])).items() if not k.startswith('__'))
            black = next(
                v for k, v in loadmat(os.path.join(base_dir, lot_filename['N'])).items() if not k.startswith('__'))
            lot = next(
                v for k, v in loadmat(os.path.join(base_dir, lot_filename['L'])).items() if not k.startswith('__'))[1:]

            # apply efficiency threshold

            white = white[:, efficiency_threshold.squeeze()]
            black = black[:, efficiency_threshold]
            lot = lot[:, efficiency_threshold]
            lot = np.delete(lot, 8719, axis=0) if lot_filename == 'L2F66H144R310324C070524VISTESTFULL.mat' else lot

            white = white.mean(axis=0)[None, ...]
            black = black.mean(axis=0)[None, ...]
            if white.max() < 50000.0:
                white = white * entrega1_white_scaling

            # get conveyor belt signatures

            if 'S' in lot_filename['E']:
                conveyor_belt = np.zeros_like(lot)
                cc_distances = compute_sam(lot, conveyor_belt)
                lot_distances = cc_distances.min(axis=-1)
                selected_cocoa = lot
            else:
                conveyor_belt = lot[:conveyor_belt_samples, :]
                cc_distances = compute_sam(lot, conveyor_belt)
                lot_distances = cc_distances.min(axis=-1)
                sorted_indices = np.argsort(lot_distances)[::-1]  # from higher sam to lower
                selected_indices = np.sort(sorted_indices[:max_num_samples])
                selected_cocoa = lot[selected_indices, :]

            selected_cocoa_reflectance = (selected_cocoa - black) / (white - black)
            selected_cocoa_reflectance = selected_cocoa_reflectance / selected_cocoa_reflectance.max(axis=-1, keepdims=True)
            selected_cocoa_reflectance = np.log(1 / (np.clip( selected_cocoa_reflectance, 1e-10, None)))
            selected_cocoa_reflectance = selected_cocoa_reflectance / selected_cocoa_reflectance.max(axis=-1, keepdims=True)
            cocoa_bean_dataset.append(selected_cocoa_reflectance)
            label_dataset.append(np.ones(selected_cocoa_reflectance.shape[0], dtype=int) * label)

            # shuffle and batch mean
            cocoa_bean_batch_mean_aux = []
            for i in range(cocoa_batch_samples):
                random_indices = np.random.choice(selected_cocoa_reflectance.shape[0], cocoa_batch_size, replace=False)
                cocoa_bean_batch_mean_aux.append(selected_cocoa_reflectance[random_indices].mean(axis=0))

            cocoa_bean_batch_mean_aux = np.stack(cocoa_bean_batch_mean_aux, axis=0)
            cocoa_bean_batch_mean_dataset.append(cocoa_bean_batch_mean_aux)
            label_batch_mean_dataset.append(np.ones(cocoa_bean_batch_mean_aux.shape[0], dtype=int) * label)

        # compute mean and std of dataset and plot

        entrega_numbers = [1, 1, 2, 1, 2, 1, 2, 2, 'mix', 'mix']
        ferm_levels = [60, 66, 73, 84, 85, 92, 94, 96, 'camila 84', 'mix']
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'orange', 'purple', 'brown']
        markers = ['o', 'o', 's', 'P', 'P', 'X', 'X', 'X', '^', '^']
        line_styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'dashed', 'dashed']



        # compute PCA with X_mean using sklearn

        full_cocoa_bean_dataset = np.concatenate(cocoa_bean_dataset, axis=0)
        full_label_dataset = np.concatenate(label_dataset, axis=0)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(full_cocoa_bean_dataset)
        explained_variance = pca.explained_variance_ratio_


        # plot cocoa bean batch mean dataset

        plt.figure(figsize=(8, 6))
        for i in range(len(cocoa_bean_batch_mean_dataset)):
            X_class = cocoa_bean_batch_mean_dataset[i]
            mean = X_class.mean(axis=0)
            std = X_class.std(axis=0)
            plt.plot(wavelengths, mean, color=colors[i], linestyle=line_styles[i],
                    label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')
            plt.fill_between(wavelengths, mean - std, mean + std, alpha=0.2, color=colors[i], linewidth=0.0)

        from sklearn.metrics import pairwise_distances
        # Calculate PCA
        full_cocoa_bean_batch_mean_dataset = np.concatenate(cocoa_bean_batch_mean_dataset, axis=0)
        full_label_batch_mean_dataset = np.concatenate(label_batch_mean_dataset, axis=0)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(np.concatenate(cocoa_bean_dataset, axis=0))
        explained_variance = pca.explained_variance_ratio_

        # Plot PCA
        plt.figure(figsize=(10, 8))
        group_means = []
        for i in range(len(cocoa_bean_batch_mean_dataset)):
            X_class = X_pca[full_label_batch_mean_dataset.squeeze() == i]
            group_mean = X_class.mean(axis=0)
            group_means.append(group_mean)
            plt.scatter(X_class[:, 0], X_class[:, 1], color=colors[i], alpha=0.5, marker=markers[i], label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')

        group_means = np.array(group_means)
        distances = pairwise_distances(group_means)

        # Plot distances and store them
        for i in range(len(group_means)):
            for j in range(i + 1, len(group_means)):
                plt.plot([group_means[i][0], group_means[j][0]], [group_means[i][1], group_means[j][1]], 'k--', lw=0.5)
                mid_point = (group_means[i] + group_means[j]) / 2
                distance = distances[i, j]
                plt.text(mid_point[0], mid_point[1], f'{distance:.2f}', ha='center', va='center')
                all_distances.append(distance)

        # Calculate and display the total distance
        total_distance = sum(all_distances)
        print(f'Total Distance: {total_distance:.2f}')

        plt.xlabel(f'Component 1: {explained_variance[0] * 100:.2f}%')
        plt.ylabel(f'Component 2: {explained_variance[1] * 100:.2f}%')
        plt.title('Cocoa mean PCA with distances')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()








