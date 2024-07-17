import os
import h5py
import numpy as np
from einops import repeat

import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import k_means

# Definición de directorios base y subdirectorios para organizar datos y resultados
base_dir = "/home/enmartz/Jobs/cacao/HDSP-dataset"
banda_dir = os.path.join(base_dir, "Anexos")
lote_dir = os.path.join(base_dir, "Optical_lab_spectral")
results_dir = os.path.join("Results")

angle_error = 0.3

# Asegurar la creación de los directorios si no existen
os.makedirs(results_dir, exist_ok=True)

# Cargar datos desde archivos .mat
BANDA = loadmat(os.path.join(banda_dir, "BANDATRANSPORTADORAC090524.mat"))['BANDA'][:, 48:]
wavelengths = BANDA[0, :]
conveyor_belt = BANDA[1:]
conveyor_cluster_centers, _, _ = k_means(conveyor_belt, n_clusters=5, n_init='auto', random_state=0)

full_cocoa_paths = {'train': {0: "L1F60R290324C070524TRAINFULL.mat"},
                    'test': {0: "L1F60R290324C070524TESTFULL.mat"}}

# full_cocoa_paths = {'train': {1: "L2F66R310324C070524TRAINFULL.mat"},
#                     'test': {1: "L2F66R310324C070524TESTFULL.mat", }}

# full_cocoa_paths = {'train': {2: "L3F84R020424C090524TRAINFULL.mat"},
#                     'test': {2: "L3F84R020424C090524TESTFULL.mat"}}

# full_cocoa_paths = {'train': {3: "L4F92R130424C090524TRAINFULL.mat"},
#                     'test': {3: "L4F92R130424C090524TESTFULL.mat"}}

# full_cocoa_paths = {'train': {4: "L5F96RDDMMAAC090524TRAINFULL.mat"},
#                     'test': {4: "L5F96RDDMMAAC090524TESTFULL.mat"}}


# Append new data to dataset
def append_to_dataset(dataset, new_data):
    current_shape = dataset.shape
    new_shape = (current_shape[0] + new_data.shape[0], current_shape[1])
    dataset.resize(new_shape)
    dataset[current_shape[0]:] = new_data


for subset_name, cocoa_filenames in full_cocoa_paths.items():
    print(f"Processing {subset_name} subset")
    with h5py.File(os.path.join(results_dir, f'{subset_name}_real_cocoa_hdsp_sam03_ultra_small.h5'), 'w') as d:
        dataset = d.create_dataset('spec', shape=(0, 2000), maxshape=(None, 2000), chunks=(256, 2000),
                                   dtype=np.float32)
        labelset = d.create_dataset('label', (0, 1), maxshape=(None, 1), chunks=(256, 1), dtype=np.uint8)

        for label, cocoa_filename in cocoa_filenames.items():
            print(f"Processing {cocoa_filename}")
            COCOA = loadmat(os.path.join(lote_dir, cocoa_filename))['LCACAO'][:, 48:]
            cocoa_lot = COCOA[1:]

            cocoa_lot = np.delete(cocoa_lot, 8719,
                                  axis=0) if cocoa_filename == 'L2F66R310324C070524TESTFULL.mat' else cocoa_lot

            # sam

            scores = np.arccos(np.matmul(cocoa_lot, conveyor_cluster_centers.T) / np.matmul(
                np.linalg.norm(cocoa_lot, axis=-1, keepdims=True),
                np.linalg.norm(conveyor_cluster_centers, axis=-1, keepdims=True).T))

            distance_Bands = np.min(scores, axis=-1)
            sam_mask = distance_Bands > angle_error

            # localize the positions where the boolean vector sam_mask changes from False to True

            sam_mask_diff = np.diff(np.concatenate(([False], sam_mask)))

            counter = 0
            for index, value in enumerate(sam_mask_diff):
                if value:
                    counter += 1

                if counter == 2:
                    sam_mask_diff[index] = False
                    counter = 0

            print('The number of cocoa beans is:', np.sum(sam_mask_diff))

            cocoa_bean_indices, = np.where(sam_mask_diff)

            # build dataset for each cocoa bean

            num_samples_per_cocoa_bean = 20

            cocoa_bean_list = []
            cocoa_bean_rep_list = []
            for cocoa_index in cocoa_bean_indices:
                if sam_mask[cocoa_index:cocoa_index + num_samples_per_cocoa_bean].all():
                    cocoa_bean_samples = cocoa_lot[cocoa_index:cocoa_index + num_samples_per_cocoa_bean]
                    cocoa_bean_rep, labels, _ = k_means(cocoa_bean_samples, n_clusters=3, # 1
                                                        n_init='auto', random_state=0)

                    # sam metric

                    # cocoa_scores = np.arccos(np.matmul(cocoa_bean_samples, cocoa_bean_rep.T) / np.matmul(
                    #     np.linalg.norm(cocoa_bean_samples, axis=-1, keepdims=True),
                    #     np.linalg.norm(cocoa_bean_rep, axis=-1, keepdims=True).T))
                    #
                    # min_distance_index, = np.argmin(cocoa_scores, axis=0)

                    cocoa_bean_list.append(cocoa_bean_samples)
                    cocoa_bean_rep_list.append(cocoa_bean_rep)
                    # cocoa_bean_rep_list.append(cocoa_bean_samples[min_distance_index])

                else:
                    print('Invalid cocoa bean range', cocoa_index, cocoa_index + num_samples_per_cocoa_bean,
                          'This cocoa bean will be skipped')

            # append to dataset

            print('Taipo')

            # append_to_dataset(dataset, selected_samples)
            # append_to_dataset(labelset, labels)
