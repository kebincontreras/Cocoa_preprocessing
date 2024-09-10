import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import k_means

# Definición de directorios base y subdirectorios para organizar datos y resultados
base_dir = "/home/enmartz/Jobs/cacao/Base_Datos_Cacao/02_07_2024/Optical_lab_spectral/VIS"
results_dir = os.path.join("samples/results_old")

# Asegurar la creación de los directorios si no existen
os.makedirs(results_dir, exist_ok=True)

eff_percentage = 0.2
angle_error = 0.2

num_samples_per_cocoa_bean = 1
epsilon_bound = 10

lot_size = 50
num_lot_reps = 1000

num_samples_train = 143
num_samples_test = 68

full_cocoa_paths = {'train': {1: {"L": "L1F85H110E270624C240724VISTRAIFULL.mat",
                                  "B": "B1F85H110E270624C240724VISTRAIFULL.mat",
                                  "N": "N1F85H110E270624C240724VISTRAIFULL.mat"},
                              2: {"L": "L2F73H144E270624C240724VISTRAIFULL.mat",
                                  "B": "B2F73H144E270624C240724VISTRAIFULL.mat",
                                  "N": "N2F73H144E270624C240724VISTRAIFULL.mat"},
                              3: {"L": "L3F94H216E270624C240724VISTRAIFULL.mat",
                                  "B": "B3F94H216E270624C240724VISTRAIFULL.mat",
                                  "N": "N3F94H216E270624C240724VISTRAIFULL.mat"},
                              4: {"L": "L4F96H252E270624C240724VISTRAIFULL.mat",
                                  "B": "B4F96H252E270624C240724VISTRAIFULL.mat",
                                  "N": "N4F96H252E270624C240724VISTRAIFULL.mat"}},
                    'test': {1: {"L": "L1F85H110E270624C250724VISTESTFULL.mat",
                                 "B": "B1F85H110E270624C250724VISTESTFULL.mat",
                                 "N": "N1F85H110E270624C250724VISTESTFULL.mat"},
                             2: {"L": "L2F73H144E270624C250724VISTESTFULL.mat",
                                 "B": "B2F73H144E270624C250724VISTESTFULL.mat",
                                 "N": "N2F73H144E270624C250724VISTESTFULL.mat"},
                             3: {"L": "L3F94H216E270624C250724VISTESTFULL.mat",
                                 "B": "B3F94H216E270624C250724VISTESTFULL.mat",
                                 "N": "N3F94H216E270624C250724VISTESTFULL.mat"},
                             4: {"L": "L4F96H252E270624C250724VISTESTFULL.mat",
                                 "B": "B4F96H252E270624C250724VISTESTFULL.mat",
                                 "N": "N4F96H252E270624C250724VISTESTFULL.mat"}},
                    }

# # belt
#
# BANDA = loadmat(os.path.join(banda_dir, "BANDATRANSPORTADORAC090524.mat"))['BANDA'][:, efficiency_indices]
# wavelengths = BANDA[0, :]
# # conveyor_belt = (BANDA[1:] - black_ref) / (white_ref - black_ref)
# conveyor_belt = BANDA[1:]
# conveyor_cluster_centers, _, _ = k_means(conveyor_belt, n_clusters=5, n_init='auto', random_state=0)

white_ref = loadmat(os.path.join(base_dir, full_cocoa_paths['train'][1]['B']))['BLANCO'].mean(axis=0)
eff_indices = white_ref >= white_ref.min() + eff_percentage * (white_ref.max() - white_ref.min())
num_bands = eff_indices.sum()

wavelengths = loadmat(os.path.join(base_dir, 'wavelengths_VIS.mat'))['wavelengths'].squeeze()
wavelengths = wavelengths[eff_indices]


# Append new data to dataset
def append_to_dataset(dataset, new_data):
    current_shape = dataset.shape
    new_shape = (current_shape[0] + new_data.shape[0], current_shape[1])
    dataset.resize(new_shape)
    dataset[current_shape[0]:] = new_data


for subset_name, cocoa_filenames in full_cocoa_paths.items():
    print(f"Processing {subset_name} subset")
    with h5py.File(os.path.join(results_dir, f'{subset_name}_real_cocoa_hdsp_oneCenter_lots.h5'), 'w') as d:
        dataset = d.create_dataset('spec', shape=(0, lot_size, num_bands), maxshape=(None, lot_size, num_bands),
                                   chunks=(256, lot_size, num_bands), dtype=np.float32)
        labelset = d.create_dataset('label', (0, 1), maxshape=(None, 1), chunks=(256, 1), dtype=np.uint8)

        for label, cocoa_filename in cocoa_filenames.items():

            print(f"Processing {cocoa_filename}")
            white = loadmat(os.path.join(base_dir, cocoa_filename['B']))['BLANCO'].mean(axis=0)[eff_indices]
            black = loadmat(os.path.join(base_dir, cocoa_filename['N']))['NEGRO'].mean(axis=0)[eff_indices]
            cocoa = loadmat(os.path.join(base_dir, cocoa_filename['L']))['CAPTURA_SPN'][:, eff_indices]
            cocoa = (cocoa - black) / (white - black)

            # sam

            scores = np.arccos(np.matmul(cocoa, conveyor_cluster_centers.T) / np.matmul(
                np.linalg.norm(cocoa, axis=-1, keepdims=True),
                np.linalg.norm(conveyor_cluster_centers, axis=-1, keepdims=True).T))

            distance_Bands = np.min(scores, axis=-1)
            sam_mask = distance_Bands > angle_error

            # localization

            indices = np.where(sam_mask)[0]
            cocoa_beans = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

            print('The number of cocoa beans is:', len(cocoa_beans))

            # build dataset for each cocoa bean

            cocoa_bean_list = []
            for c_idx, cocoa_bean_indices in enumerate(cocoa_beans):
                if len(cocoa_bean_indices) >= num_samples_per_cocoa_bean + epsilon_bound:
                    center_index = len(cocoa_bean_indices) // 2
                    center_dev = num_samples_per_cocoa_bean // 2
                    selected_indices = cocoa_bean_indices[center_index - center_dev:center_index + center_dev + (
                        1 if num_samples_per_cocoa_bean % 2 == 1 else 0)]

                    cocoa_bean_samples = cocoa_lot[selected_indices]
                    cocoa_bean_list.append(cocoa_bean_samples)
                else:
                    # print('Invalid cocoa bean range', c_idx, 'This cocoa bean will be skipped')
                    pass

            print('The number of valid cocoa beans is:', len(cocoa_bean_list))

            # append to dataset

            cocoa_final_list = np.concatenate(cocoa_bean_list, axis=0)
            cocoa_final_list = (cocoa_final_list - black_ref) / (white_ref - black_ref)

            # append to dataset

            if num_samples_train > 0 or num_samples_test > 0:
                num_samples = num_samples_train if subset_name == 'train' else num_samples_test
            else:
                num_samples = cocoa_final_list.shape[0]

            final_indices = np.linspace(0, cocoa_final_list.shape[0] - 1, num_samples, dtype=np.uint8)
            cocoa_final_list = cocoa_final_list[final_indices]

            # generate lots

            cocoa_lot_final_list = []

            for i in range(num_lot_reps):
                rand_indices = np.random.permutation(cocoa_final_list.shape[0])
                cocoa_lot_final_list.append(cocoa_final_list[rand_indices[:lot_size]].flatten())

            cocoa_lot_final_list = np.stack(cocoa_lot_final_list, axis=0)

            # append_to_dataset(dataset, cocoa_final_list[final_indices])
            append_to_dataset(dataset, cocoa_lot_final_list)
            append_to_dataset(labelset, np.ones((num_lot_reps, 1), dtype=np.uint8) * label)

            print('The final number of samples is:', num_lot_reps)
