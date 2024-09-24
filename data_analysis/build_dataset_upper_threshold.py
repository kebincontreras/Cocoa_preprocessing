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
lote_dir = os.path.join(base_dir, "data")
results_dir = os.path.join("results")

# Asegurar la creación de los directorios si no existen
os.makedirs(results_dir, exist_ok=True)

eff_percentage = 0.2
angle_error = 0.2
upper_angle_error = 0.8

num_samples_per_cocoa_bean = 1
epsilon_bound = 10

lot_size = 50
np.random.seed(0)
num_lot_reps = 1000

num_samples_train = 143
num_samples_test = 68

full_cocoa_paths = {'train': {0: "L1F60H096R290324C070524VISTRAIFULL.mat",
                              1: "L2F66H144R310324C070524VISTRAIFULL.mat",
                              2: "L2F73H144E270624C240724VISTRAIFULL.mat",
                              3: "L3F84H192R020424C090524VISTRAIFULL.mat",
                              4: "L1F85H110E270624C240724VISTRAIFULL.mat",
                              5: "L4F92H264R130424C090524VISTRAIFULL.mat",
                              6: "L3F94H216E270624C240724VISTRAIFULL.mat",
                              7: "L4F96H252E270624C240724VISTRAIFULL.mat",
                              },
                    'test': {0: "L1F60H096R290324C070524VISTESTFULL.mat",
                             1: "L2F66H144R310324C070524VISTESTFULL.mat",
                             2: "L2F73H144E270624C250724VISTESTFULL.mat",
                             3: "L3F84H192R020424C090524VISTESTFULL.mat",
                             4: "L1F85H110E270624C250724VISTESTFULL.mat",
                             5: "L4F92H264R130424C090524VISTESTFULL.mat",
                             6: "L3F94H216E270624C250724VISTESTFULL.mat",
                             7: "L4F96H252E270624C250724VISTESTFULL.mat",
                             }
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
    new_shape = (current_shape[0] + new_data.shape[0], current_shape[1], current_shape[2])
    dataset.resize(new_shape)
    dataset[current_shape[0]:] = new_data


def append_to_label_dataset(dataset, new_data):
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

    cocoa_sam_list = []

    with h5py.File(os.path.join(results_dir, f'tester_{subset_name}_real_cocoa_hdsp_oneCenter_squarelots.h5'),
                   'w') as d:
        wavelengths = d.create_dataset('wavelengths', data=wavelengths)
        dataset = d.create_dataset('spec', shape=(0, lot_size, len(white_ref)),
                                   maxshape=(None, lot_size, len(white_ref)),
                                   chunks=(256, lot_size, len(white_ref)), dtype=np.float32)
        labelset = d.create_dataset('label', (0, 1), maxshape=(None, 1), chunks=(256, 1), dtype=np.uint8)

        for label, cocoa_filename in cocoa_filenames.items():
            print(f"Processing {cocoa_filename}")
            try:
                COCOA = loadmat(os.path.join(lote_dir, cocoa_filename))['LCACAO'][:, efficiency_indices]
            except:
                COCOA = loadmat(os.path.join(lote_dir, cocoa_filename))['CAPTURA_SPN'][:, efficiency_indices]

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
            # sam_mask = (upper_angle_error > distance_Bands) * (distance_Bands > angle_error)
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

            # CONVEYOR BELT

            # sam_scores = np.arccos(np.matmul(cocoa_final_list, conveyor_cluster_centers.T) / np.matmul(
            #     np.linalg.norm(cocoa_final_list, axis=-1, keepdims=True),
            #     np.linalg.norm(conveyor_cluster_centers, axis=-1, keepdims=True).T))
            # sam_scores = np.min(sam_scores, axis=-1)

            # sam_scores = np.arccos(np.matmul(cocoa_lot_final_list, conveyor_cluster_centers.T) / np.matmul(
            #     np.linalg.norm(cocoa_lot_final_list, axis=-1, keepdims=True),
            #     np.linalg.norm(conveyor_cluster_centers, axis=-1, keepdims=True).T))
            # sam_scores = np.min(sam_scores, axis=-1)

            # sam_scores = np.arccos(np.matmul(cocoa_lot_final_list.mean(axis=1), conveyor_cluster_centers.T) / np.matmul(
            #     np.linalg.norm(cocoa_lot_final_list.mean(axis=1), axis=-1, keepdims=True),
            #     np.linalg.norm(conveyor_cluster_centers, axis=-1, keepdims=True).T))
            # sam_scores = np.min(sam_scores, axis=-1)

            if 'L2F73' in cocoa_filename:
                print('taipo')

            upper_sam_mask = sam_scores < upper_angle_error
            indices = np.where(sam_mask)[0]
            cocoa_beans = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

            sam_scores = np.arccos(np.matmul(cocoa_beans, conveyor_cluster_centers.T) / np.matmul(
                np.linalg.norm(cocoa_beans, axis=-1, keepdims=True),
                np.linalg.norm(conveyor_cluster_centers, axis=-1, keepdims=True).T))
            sam_scores = np.min(sam_scores, axis=-1)


            cocoa_sam_list.append(sam_scores[upper_sam_mask])

            # upper threshold for cocoa_final_list

            cocoa_final_list = cocoa_final_list[sam_scores < upper_angle_error]

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


        # plot cocoa_sam_list with labels in colors

        entrega_numbers = [1, 1, 2, 1, 2, 1, 2, 2]
        ferm_levels = [60, 66, 73, 84, 85, 92, 94, 96]
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'orange']

        alpha = 0.5
        scale = 1
        plt.figure(figsize=(scale * 10, scale * 5))

        for i, cocoa_sam in enumerate(cocoa_sam_list):
            plt.plot(cocoa_sam.squeeze(), color=colors[i], alpha=alpha)
            plt.scatter(cocoa_sam.min(), cocoa_sam.min(), color=colors[i],
                        label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')

        plt.xlabel('Sample Index')
        plt.ylabel('SAM')

        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{results_dir}/cocoa_sam_{subset_name}.svg')
        plt.show()

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
        plt.show()

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
        plt.show()

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
        plt.show()

        break
