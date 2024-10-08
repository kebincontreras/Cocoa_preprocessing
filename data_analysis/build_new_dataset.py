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
epsilon_bound = 10

lot_size = 50
np.random.seed(0)
num_lot_reps = 1000

num_samples_train = 143
num_samples_test = 68

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

# Cargar datos desde archivos .mat
wavelengths = BANDA[0, :]

# Append new data to dataset

def append_to_dataset(dataset, new_data):
    current_shape = dataset.shape
    new_shape = (current_shape[0] + new_data.shape[0], current_shape[1], current_shape[2])
    dataset.resize(new_shape)
    dataset[current_shape[0]:] = new_data


def append_to_label_dataset(dataset, new_data):
    current_shape = dataset.shape
    new_shape = (current_shape[0] + new_data.shape[0], current_shape[1])
    dataset.resize(new_shape)
    dataset[current_shape[0]:] = new_data


for subset_name, cocoa_filenames in full_cocoa_paths.items():
    print(f"Processing {subset_name} subset")

    cocoa_sam_list = []
    label_sam_list = []

    black_list = []
    white_list = []

    with h5py.File(os.path.join(results_dir, f'tester_{subset_name}_real_cocoa_hdsp_oneCenter_squarelots_BAW.h5'),
                   'w') as d:
        wavelengths = d.create_dataset('wavelengths', data=wavelengths)
        dataset = d.create_dataset('spec', shape=(0, lot_size, len(white_ref)),
                                   maxshape=(None, lot_size, len(white_ref)),
                                   chunks=(256, lot_size, len(white_ref)), dtype=np.float32)
        labelset = d.create_dataset('label', (0, 1), maxshape=(None, 1), chunks=(256, 1), dtype=np.uint8)

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

            black_list.append(black)
            white_list.append(white)

            continue

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

            cocoa_final_list = np.concatenate(cocoa_bean_list, axis=0)
            cocoa_final_list = (cocoa_final_list - black) / (white - black)

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
                cocoa_lot_final_list.append(cocoa_final_list[rand_indices[:lot_size]])

            cocoa_lot_final_list = np.stack(cocoa_lot_final_list, axis=0)

            # append_to_dataset(dataset, cocoa_final_list[final_indices])
            append_to_dataset(dataset, cocoa_lot_final_list)
            append_to_label_dataset(labelset, np.ones((num_lot_reps, 1), dtype=np.uint8) * label)

            print('The final number of samples is:', num_lot_reps)

            # final sam list

            zeros = 1e-3 * np.ones((1, cocoa_final_list.shape[-1]))
            sam_scores = np.arccos(np.matmul(cocoa_final_list, zeros.T) / np.matmul(
                np.linalg.norm(cocoa_final_list, axis=-1, keepdims=True),
                np.linalg.norm(zeros, axis=-1, keepdims=True).T))

            # zeros = 1e-3 * np.ones((1, cocoa_lot_final_list.shape[-1]))
            # sam_scores = np.arccos(np.matmul(cocoa_lot_final_list, zeros.T) / np.matmul(
            #     np.linalg.norm(cocoa_lot_final_list, axis=-1, keepdims=True),
            #     np.linalg.norm(zeros, axis=-1, keepdims=True).T))

            # zeros = 1e-3 * np.ones((1, cocoa_lot_final_list.mean(axis=1).shape[-1]))
            # sam_scores = np.arccos(np.matmul(cocoa_lot_final_list.mean(axis=1), zeros.T) / np.matmul(
            #     np.linalg.norm(cocoa_lot_final_list.mean(axis=1), axis=-1, keepdims=True),
            #     np.linalg.norm(zeros, axis=-1, keepdims=True).T))

            cocoa_sam_list.append(sam_scores)

        # plot cocoa_sam_list with labels in colors

        entrega_numbers = [1, 1, 2, 1, 2, 1, 2, 2]
        ferm_levels = [60, 66, 73, 84, 85, 92, 94, 96]
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'orange']

        wavelengths = BANDA[0, :]
        black_list = np.stack(black_list, axis=0)
        white_list = np.stack(white_list, axis=0)

        alpha = 0.5
        scale = 1
        plt.figure(figsize=(scale * 10, scale * 5))

        for i, black in enumerate(black_list):
            if entrega_numbers[i] == 1:
                plt.plot(black.squeeze(), color=colors[i], alpha=alpha)
                plt.scatter(black.min(), black.min(), color=colors[i],
                            label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')

        plt.title('Black Ref - E1')
        plt.xlabel('Wavelength')
        plt.ylabel('Intensity')

        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{results_dir}/black_{subset_name}.svg')
        plt.close()

        plt.figure(figsize=(scale * 10, scale * 5))

        for i, white in enumerate(white_list):
            # if entrega_numbers[i] == 1:
            plt.plot(white.squeeze(), color=colors[i], alpha=alpha)
            plt.scatter(wavelengths.min(), white.min(), color=colors[i],
                        label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')

        plt.title('White Ref')
        plt.xlabel('Wavelength')
        plt.ylabel('Intensity')

        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{results_dir}/white_{subset_name}.svg')
        plt.close()

        entrega_numbers = [1, 1, 2, 1, 2, 1, 2, 2]
        ferm_levels = [60, 66, 73, 84, 85, 92, 94, 96]
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'orange']

        plt.figure(figsize=(scale * 10, scale * 5))

        for i, cocoa_sam in enumerate(cocoa_sam_list):
            x_index = np.linspace(0, cocoa_sam.shape[0] - 1, cocoa_sam.shape[0])
            plt.plot(x_index, cocoa_sam.squeeze(), color=colors[i], alpha=alpha)
            plt.scatter(cocoa_sam.min(), cocoa_sam.min(), color=colors[i],
                        label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')

        plt.xlabel('Sample Index')
        plt.ylabel('SAM')

        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{results_dir}/cocoa_sam_{subset_name}.svg')
        # plt.show()
        plt.close()

        scale = 1.5
        plt.figure(figsize=(scale * 10, scale * 5))

        for i, cocoa_sam in enumerate(cocoa_sam_list):
            plt.subplot(2, 4, i + 1)
            plt.plot(cocoa_sam.squeeze(), color=colors[i], alpha=alpha)
            plt.scatter(cocoa_sam.min(), cocoa_sam.min(), color=colors[i],
                        label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')
            plt.xlabel('Sample Index')
            plt.ylabel('SAM')
            # plt.ylim([0.5, 0.64])
            plt.ylim([0.25, 2.5])

            plt.grid()
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'{results_dir}/all_cocoa_sam_{subset_name}.svg')
        # plt.show()
        plt.close()

        # fila india

        plt.figure(figsize=(scale * 10, scale * 5))

        x_count = 0
        x_index = np.linspace(0, cocoa_sam.shape[0] - 1, cocoa_sam.shape[0])
        for i, cocoa_sam in enumerate(cocoa_sam_list):
            plt.plot(x_index, cocoa_sam.squeeze(), color=colors[i], alpha=alpha)
            plt.scatter(cocoa_sam.min(), cocoa_sam.min(), color=colors[i],
                        label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')
            x_index += cocoa_sam.shape[0]

        plt.xlabel('Sample Index')
        plt.ylabel('SAM')

        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{results_dir}/india_cocoa_sam_{subset_name}.svg')
        # plt.show()
        plt.close()

        scale = 1.5
        plt.figure(figsize=(scale * 10, scale * 5))

        for i, cocoa_sam in enumerate(cocoa_sam_list):
            plt.subplot(2, 4, i + 1)
            plt.plot(cocoa_sam.squeeze(), color=colors[i], alpha=alpha)
            plt.scatter(cocoa_sam.min(), cocoa_sam.min(), color=colors[i],
                        label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')
            plt.xlabel('Sample Index')
            plt.ylabel('SAM')
            # plt.ylim([0.5, 0.64])
            plt.ylim([0.25, 2.5])

            plt.grid()
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'{results_dir}/india_all_cocoa_sam_{subset_name}.svg')
        # plt.show()
        plt.close()

        break
