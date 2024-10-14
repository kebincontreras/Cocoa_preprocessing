import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from sklearn.cluster import k_means

# Definición de directorios base y subdirectorios para organizar datos y resultados
base_dir = "/home/enmartz/Jobs/cacao/HDSP-dataset/FLAME"
banda_dir = os.path.join(base_dir, "Anexos")
bw_dir = os.path.join(base_dir, "bw_ref")
lote_dir = os.path.join(base_dir, "Optical_lab_spectral")
results_dir = os.path.join("../../results")

# Asegurar la creación de los directorios si no existen
os.makedirs(results_dir, exist_ok=True)

angle_error = 0.4

# black and white refs

white_ref = np.loadtxt(os.path.join(bw_dir, 'BLANCO_ESCALADO_K.csv'), delimiter=',')[48:]
black_ref = np.loadtxt(os.path.join(bw_dir, 'NEGRO_DEEPL_KEBIN.csv'), delimiter=',')[48:]

# Cargar datos desde archivos .mat
BANDA = loadmat(os.path.join(banda_dir, "BANDATRANSPORTADORAC090524.mat"))['BANDA'][:, 48:]
wavelengths = BANDA[0, :]
conveyor_belt = (BANDA[1:] - black_ref) / (white_ref - black_ref)
conveyor_cluster_centers, _, _ = k_means(conveyor_belt, n_clusters=5, n_init='auto', random_state=0)

# cocoa lots

full_cocoa_paths = {'train': {0: "L1F60R290324C070524TRAINFULL.mat",
                              1: "L2F66R310324C070524TRAINFULL.mat",
                              2: "L3F84R020424C090524TRAINFULL.mat",
                              3: "L4F92R130424C090524TRAINFULL.mat",
                              4: "L5F96RDDMMAAC090524TRAINFULL.mat"},
                    'test': {0: "L1F60R290324C070524TESTFULL.mat",
                             1: "L2F66R310324C070524TESTFULL.mat",
                             2: "L3F84R020424C090524TESTFULL.mat",
                             3: "L4F92R130424C090524TESTFULL.mat",
                             4: "L5F96RDDMMAAC090524TESTFULL.mat"}}

for subset_name, cocoa_filenames in full_cocoa_paths.items():
    print(f"Processing {subset_name} subset")
    with h5py.File(os.path.join(results_dir, f'{subset_name}_cocoa_hdsp_sam04_ultra_small.h5'), 'w') as d:
        dataset = d.create_dataset('spec', shape=(0, 2000), maxshape=(None, 2000), chunks=(256, 2000),
                                   dtype=np.float32)
        labelset = d.create_dataset('label', (0, 1), maxshape=(None, 1), chunks=(256, 1), dtype=np.uint8)


        # Append new data to dataset
        def append_to_dataset(dataset, new_data):
            current_shape = dataset.shape
            new_shape = (current_shape[0] + new_data.shape[0], current_shape[1])
            dataset.resize(new_shape)
            dataset[current_shape[0]:] = new_data


        for label, cocoa_filename in cocoa_filenames.items():
            print(f"Processing {cocoa_filename}")
            COCOA = loadmat(os.path.join(lote_dir, cocoa_filename))['LCACAO'][:, 48:]
            cocoa_lot = (COCOA[1:] - black_ref) / (white_ref - black_ref)

            cocoa_lot = np.delete(cocoa_lot, 8719,
                                  axis=0) if cocoa_filename == 'L2F66R310324C070524TESTFULL.mat' else cocoa_lot

            # sam

            scores = np.arccos(np.matmul(cocoa_lot, conveyor_cluster_centers.T) / np.matmul(
                np.linalg.norm(cocoa_lot, axis=-1, keepdims=True),
                np.linalg.norm(conveyor_cluster_centers, axis=-1, keepdims=True).T))

            distance_Bands = np.min(scores, axis=-1)
            sam_mask = distance_Bands > angle_error

            # selected bands

            selected_samples = cocoa_lot[sam_mask]
            labels = np.ones((len(selected_samples), 1), dtype=np.uint8) * label

            # append to dataset

            # num_samples = 5000 if subset_name == 'train' else 1000  # small
            num_samples = 4750 if subset_name == 'train' else 250  # ultra-small

            append_to_dataset(dataset, selected_samples[:num_samples])
            append_to_dataset(labelset, labels[:num_samples])
