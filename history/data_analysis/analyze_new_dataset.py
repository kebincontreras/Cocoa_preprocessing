import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import k_means

# Definición de directorios base y subdirectorios para organizar datos y resultados
base_dir = "/home/enmartz/Jobs/cacao/Base_Datos_Cacao/ALL_VIS"
band_dir = os.path.join(base_dir, "BANDATRANSPORTADORAC090524.mat")
results_dir = os.path.join("samples/results_old")

# Asegurar la creación de los directorios si no existen
os.makedirs(results_dir, exist_ok=True)

eff_percentage = 0.2
angle_error = 0.2

num_samples_per_cocoa_bean = 1
epsilon_bound = 5

full_cocoa_paths = {'train': {0: {"L": "L1F60H096R290324C070524VISTRAIFULL.mat",
                                  "B": "blanco.mat",
                                  "N": "negro.mat"},
                              1: {"L": "L2F66H144R310324C070524VISTRAIFULL.mat",
                                  "B": "blanco.mat",
                                  "N": "negro.mat"},
                              2: {"L": "L7F73H144E270624C240724VISTRAIFULL.mat",
                                  "B": "B7F73H144E270624C240724VISTRAIFULL.mat",
                                  "N": "N7F73H144E270624C240724VISTRAIFULL.mat"},
                              3: {"L": "L3F84H192R020424C090524VISTRAIFULL.mat",
                                  "B": "blanco.mat",
                                  "N": "negro.mat"},
                              4: {"L": "L6F85H110E270624C240724VISTRAIFULL.mat",
                                  "B": "B6F85H110E270624C240724VISTRAIFULL.mat",
                                  "N": "N6F85H110E270624C240724VISTRAIFULL.mat"},
                              5: {"L": "L4F92H264R130424C090524VISTRAIFULL.mat",
                                  "B": "blanco.mat",
                                  "N": "negro.mat"},
                              6: {"L": "L8F94H216E270624C240724VISTRAIFULL.mat",
                                  "B": "B8F94H216E270624C240724VISTRAIFULL.mat",
                                  "N": "N8F94H216E270624C240724VISTRAIFULL.mat"},
                              # 7: {"L": "L5F96HXXXRDDMMAAC090524VISTRAIFULL.mat",
                              #     "B": "blanco.mat",
                              #     "N": "negro.mat"},
                              7: {"L": "L9F96H252E270624C240724VISTRAIFULL.mat",
                                  "B": "B9F96H252E270624C240724VISTRAIFULL.mat",
                                  "N": "N9F96H252E270624C240724VISTRAIFULL.mat"}},
                    'test': {0: {"L": "L1F60H096R290324C070524VISTESTFULL.mat",
                                 "B": "blanco.mat",
                                 "N": "negro.mat"},
                             1: {"L": "L2F66H144R310324C070524VISTESTFULL.mat",
                                 "B": "blanco.mat",
                                 "N": "negro.mat"},
                             2: {"L": "L7F73H144E270624C250724VISTESTFULL.mat",
                                 "B": "B7F73H144E270624C250724VISTESTFULL.mat",
                                 "N": "N7F73H144E270624C250724VISTESTFULL.mat"},
                             3: {"L": "L3F84H192R020424C090524VISTESTFULL.mat",
                                 "B": "blanco.mat",
                                 "N": "negro.mat"},
                             4: {"L": "L6F85H110E270624C250724VISTESTFULL.mat",
                                 "B": "B6F85H110E270624C250724VISTESTFULL.mat",
                                 "N": "N6F85H110E270624C250724VISTESTFULL.mat"},
                             5: {"L": "L4F92H264R130424C090524VISTESTFULL.mat",
                                 "B": "blanco.mat",
                                 "N": "negro.mat"},
                             6: {"L": "L8F94H216E270624C250724VISTESTFULL.mat",
                                 "B": "B8F94H216E270624C250724VISTESTFULL.mat",
                                 "N": "N8F94H216E270624C250724VISTESTFULL.mat"},
                             # 7: {"L": "L5F96HXXXRDDMMAAC090524VISTESTFULL.mat",
                             #     "B": "blanco.mat",
                             #     "N": "negro.mat"},
                             7: {"L": "L9F96H252E270624C250724VISTESTFULL.mat",
                                 "B": "B9F96H252E270624C250724VISTESTFULL.mat",
                                 "N": "N9F96H252E270624C250724VISTESTFULL.mat"}},
                    }

# black and white refs + efficiency indices

white_ref = loadmat(os.path.join(base_dir, full_cocoa_paths['train'][0]['B']))['spectral_data'].mean(axis=0)
eff_indices = white_ref >= white_ref.min() + eff_percentage * (white_ref.max() - white_ref.min())
num_bands = eff_indices.sum()

wavelengths = loadmat(os.path.join(base_dir, 'wavelengths_VIS.mat'))['wavelengths'].squeeze()
wavelengths = wavelengths[eff_indices]

# # belt

black_ref = loadmat(os.path.join(base_dir, full_cocoa_paths['train'][0]['N']))['spectral_data'].mean(axis=0)

white_ref = white_ref[eff_indices]
black_ref = black_ref[eff_indices]

BANDA = loadmat(band_dir)['BANDA'][:, eff_indices]
conveyor_belt = (BANDA[1:] - black_ref[None, :]) / (white_ref[None, :] - black_ref[None, :])
conveyor_cluster_centers, _, _ = k_means(conveyor_belt, n_clusters=5, n_init='auto', random_state=0)


# Append new data to dataset

def append_to_dataset(dataset, new_data):
    current_shape = dataset.shape
    new_shape = (current_shape[0] + new_data.shape[0], current_shape[1])
    dataset.resize(new_shape)
    dataset[current_shape[0]:] = new_data


for subset_name, cocoa_filenames in full_cocoa_paths.items():
    print(f"Processing {subset_name} subset")

    cocoa_sam_list = []
    label_sam_list = []

    for label, cocoa_filename in cocoa_filenames.items():

        print(f"Processing {cocoa_filename}")
        try:
            white = loadmat(os.path.join(base_dir, cocoa_filename['B']))['BLANCO'].mean(axis=0)[eff_indices]
        except:
            white = loadmat(os.path.join(base_dir, cocoa_filename['B']))['spectral_data'].mean(axis=0)[eff_indices]

        try:
            black = loadmat(os.path.join(base_dir, cocoa_filename['N']))['NEGRO'].mean(axis=0)[eff_indices]
        except:
            black = loadmat(os.path.join(base_dir, cocoa_filename['N']))['spectral_data'].mean(axis=0)[eff_indices]

        try:
            cocoa = loadmat(os.path.join(base_dir, cocoa_filename['L']))['CAPTURA_SPN'][:, eff_indices]
        except:
            cocoa = loadmat(os.path.join(base_dir, cocoa_filename['L']))['LCACAO'][:, eff_indices]

        cocoa = np.delete(cocoa, 8719, axis=0) if cocoa_filename == 'L2F66R310324C070524TESTFULL.mat' else cocoa

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

                cocoa_bean_samples = cocoa[selected_indices]
                cocoa_bean_list.append(cocoa_bean_samples)
            else:
                # print('Invalid cocoa bean range', c_idx, 'This cocoa bean will be skipped')
                pass

        print('The number of valid cocoa beans is:', len(cocoa_bean_list))

        # append to dataset

        cocoa = (cocoa - black) / (white - black)
        cocoa_final_list = np.concatenate(cocoa_bean_list, axis=0)
        # cocoa_final_list = (cocoa_final_list - black) / (white - black)

        # cocoa_final_list = cocoa

        # final sam list

        zeros = 1e-3 * np.ones((1, cocoa_final_list.shape[-1]))
        sam_scores = np.arccos(np.matmul(cocoa_final_list, zeros.T) / np.matmul(
            np.linalg.norm(cocoa_final_list, axis=-1, keepdims=True),
            np.linalg.norm(zeros, axis=-1, keepdims=True).T))

        cocoa_sam_list.append(sam_scores)

    # plot cocoa_sam_list with labels in colors

    ferm_levels = [60, 66, 73, 84, 85, 92, 94, 96, 96]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple']

    plt.figure(figsize=(8, 8))

    for i, cocoa_sam in enumerate(cocoa_sam_list):
        plt.plot(cocoa_sam.squeeze(), color=colors[i], alpha=0.5)
        plt.scatter(cocoa_sam.min(), cocoa_sam.min(), color=colors[i], label=f'ferm level {ferm_levels[i]}')

    plt.xlabel('Wavelengths')
    plt.ylabel('SAM')

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{results_dir}/cocoa_sam_{subset_name}.svg')
    plt.show()

    break
