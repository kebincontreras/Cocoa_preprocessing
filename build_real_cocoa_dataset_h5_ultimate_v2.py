import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import k_means

# Definición de directorios base y subdirectorios para organizar datos y resultados
base_dir = "/home/enmartz/Jobs/cacao/HDSP-dataset/FLAME"
banda_dir = os.path.join(base_dir, "Anexos")
bw_dir = os.path.join(base_dir, "bw_ref")
lote_dir = os.path.join(base_dir, "Optical_lab_spectral")
results_dir = os.path.join("results")

# Asegurar la creación de los directorios si no existen
os.makedirs(results_dir, exist_ok=True)

eff_percentage = 0.2
angle_error = 0.2

num_samples_per_cocoa_bean = 1
epsilon_bound = 10

num_samples_train = 143
num_samples_test = 68

full_cocoa_paths = {'train': {0: "L1F60R290324C070524TRAINFULL.mat",
                              1: "L2F66R310324C070524TRAINFULL.mat",
                              2: "L3F84R020424C090524TRAINFULL.mat",
                              3: "L4F92R130424C090524TRAINFULL.mat",
                              4: "L5F96RDDMMAAC090524TRAINFULL.mat"},
                    'test': {0: "L1F60R290324C070524TESTFULL.mat",
                             1: "L2F66R310324C070524TESTFULL.mat",
                             2: "L3F84R020424C090524TESTFULL.mat",
                             3: "L4F92R130424C090524TESTFULL.mat",
                             4: "L5F96RDDMMAAC090524TESTFULL.mat"}
                    }

# black and white refs

white_ref = np.loadtxt(os.path.join(bw_dir, 'BLANCO_ESCALADO_K.csv'), delimiter=',')

efficiency_indices = white_ref >= white_ref.min() + eff_percentage * (white_ref.max() - white_ref.min())

white_ref = white_ref[efficiency_indices]
black_ref = np.loadtxt(os.path.join(bw_dir, 'NEGRO_DEEPL_KEBIN.csv'), delimiter=',')[efficiency_indices]

# Cargar datos desde archivos .mat
BANDA = loadmat(os.path.join(banda_dir, "BANDATRANSPORTADORAC090524.mat"))['BANDA'][:, efficiency_indices]
wavelengths = BANDA[0, :]
# conveyor_belt = (BANDA[1:] - black_ref) / (white_ref - black_ref)
conveyor_belt = BANDA[1:]
conveyor_cluster_centers, _, _ = k_means(conveyor_belt, n_clusters=5, n_init='auto', random_state=0)


# Append new data to dataset
def append_to_dataset(dataset, new_data):
    current_shape = dataset.shape
    new_shape = (current_shape[0] + new_data.shape[0], current_shape[1])
    dataset.resize(new_shape)
    dataset[current_shape[0]:] = new_data


def moving_average(a, n=3):
    ret = np.cumsum(a, axis=1, dtype=float)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:] / n


for subset_name, cocoa_filenames in full_cocoa_paths.items():
    print(f"Processing {subset_name} subset")
    with h5py.File(os.path.join(results_dir, f'{subset_name}_real_cocoa_hdsp_oneCenter.h5'), 'w') as d:
        dataset = d.create_dataset('spec', shape=(0, len(white_ref)), maxshape=(None, len(white_ref)),
                                   chunks=(256, len(white_ref)), dtype=np.float32)
        labelset = d.create_dataset('label', (0, 1), maxshape=(None, 1), chunks=(256, 1), dtype=np.uint8)

        for label, cocoa_filename in cocoa_filenames.items():
            print(f"Processing {cocoa_filename}")
            COCOA = loadmat(os.path.join(lote_dir, cocoa_filename))['LCACAO'][:, efficiency_indices]
            wavelengths = COCOA[0]
            # cocoa_lot = (COCOA[1:] - black_ref) / (white_ref - black_ref)
            cocoa_lot = COCOA[1:]

            cocoa_lot = np.delete(cocoa_lot, 8719,
                                  axis=0) if cocoa_filename == 'L2F66R310324C070524TESTFULL.mat' else cocoa_lot

            # sam

            scores = np.arccos(np.matmul(cocoa_lot, conveyor_cluster_centers.T) / np.matmul(
                np.linalg.norm(cocoa_lot, axis=-1, keepdims=True),
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

            append_to_dataset(dataset, cocoa_final_list[final_indices])
            append_to_dataset(labelset, np.ones((num_samples, 1), dtype=np.uint8) * label)

            print('The final number of samples is:', num_samples)
