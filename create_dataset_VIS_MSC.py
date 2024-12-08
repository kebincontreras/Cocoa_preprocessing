import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat

from sklearn.decomposition import PCA
from datetime import datetime


# functions

def msc(input_data):
    """
        :msc: Scatter Correction technique performed with mean of the sample data as the reference.
        :param input_data: Array of spectral data
        :type input_data: DataFrame
        :returns: data_msc (ndarray): Scatter corrected spectra data
    """
    eps = np.finfo(np.float32).eps
    input_data = np.array(input_data, dtype=np.float64)
    ref = []
    sampleCount = int(len(input_data))

    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i, :] -= input_data[i, :].mean()

    # Get the reference spectrum. If not given, estimate it from the mean
    # Define a new array and populate it with the corrected data
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        for j in range(0, sampleCount, 10):
            ref.append(np.mean(input_data[j:j + 10], axis=0))
            # Run regression
            fit = np.polyfit(ref[i], input_data[i, :], 1, full=True)
            # Apply correction
            data_msc[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]

    return data_msc


def compute_sam(a, b):
    assert a.ndim == 2, "a must have two dimensions, if you only have one, please add an new dimension in the first place"
    assert b.ndim == 2, "b must have two dimensions, if you only have one, please add an new dimension in the first place"

    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.matmul(a, b.T) / np.matmul(a_norm, b_norm.T), a_min=-1.0, a_max=1.0))


def append_to_dataset(dataset, new_data):
    current_shape = dataset.shape
    new_shape = (current_shape[0] + new_data.shape[0], current_shape[1])
    dataset.resize(new_shape)
    dataset[current_shape[0]:] = new_data


# set main paths

base_dir = "/home/enmartz/Jobs/cacao/Base_Datos_Cacao/ALL_VIS"
out_dir = os.path.join("built_datasets")
os.makedirs(out_dir, exist_ok=True)

# set variables

efficiency_range = [500, 850]  # nanometers
entrega1_white_scaling = 21.0  # white / this
conveyor_belt_samples = 200  # for sam metric
angle_error = 0.25  # angle error between conveyor belt and cocoa signatures
max_num_samples = 1000  # selected samples from lot with higher sam

cocoa_batch_size = 50  # guillotine methodology
cocoa_batch_samples = 1000  # number of batch samples

plot_num_samples = 500
debug = False
debug_pca = True

# set variables for cocoa dataset

entrega_numbers = [1, 1, 2, 1, 2, 1, 2, 2]
ferm_levels = [60, 66, 73, 84, 85, 92, 94, 96]
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'orange']
markers = ['o', 'o', 's', 'P', 'P', 'X', 'X', 'X']

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
              # 7: {"L": "L5F96HXXXRDDMMAAC090524VISTRAIFULL.mat",
              #     "B": "blanco.mat",
              #     "N": "negro.mat",
              #     "E": "Entrega 1"},
              7: {"L": "L4F96H252E270624C240724VISTRAIFULL.mat",
                  "B": "B4F96H252E270624C240724VISTRAIFULL.mat",
                  "N": "N4F96H252E270624C240724VISTRAIFULL.mat",
                  "E": "Entrega 2"},
              },
    'test': {0: {"L": "L1F60H096R290324C070524VISTESTFULL.mat",
                 "B": "blanco.mat",
                 "N": "negro.mat",
                 "E": "Entrega 1"},
             1: {"L": "L2F66H144R310324C070524VISTESTFULL.mat",
                 "B": "blanco.mat",
                 "N": "negro.mat",
                 "E": "Entrega 1"},
             2: {"L": "L2F73H144E270624C250724VISTESTFULL.mat",
                 "B": "B2F73H144E270624C250724VISTESTFULL.mat",
                 "N": "N2F73H144E270624C250724VISTESTFULL.mat",
                 "E": "Entrega 2"},
             3: {"L": "L3F84H192R020424C090524VISTESTFULL.mat",
                 "B": "blanco.mat",
                 "N": "negro.mat",
                 "E": "Entrega 1"},
             4: {"L": "L1F85H110E270624C250724VISTESTFULL.mat",
                 "B": "B1F85H110E270624C250724VISTESTFULL.mat",
                 "N": "N1F85H110E270624C250724VISTESTFULL.mat",
                 "E": "Entrega 2"},
             5: {"L": "L4F92H264R130424C090524VISTESTFULL.mat",
                 "B": "blanco.mat",
                 "N": "negro.mat",
                 "E": "Entrega 1"},
             6: {"L": "L3F94H216E270624C250724VISTESTFULL.mat",
                 "B": "B3F94H216E270624C250724VISTESTFULL.mat",
                 "N": "N3F94H216E270624C250724VISTESTFULL.mat",
                 "E": "Entrega 2"},
             # 7: {"L": "L5F96HXXXRDDMMAAC090524VISTESTFULL.mat",
             #     "B": "blanco.mat",
             #     "N": "negro.mat",
             #     "E": "Entrega 1"},
             7: {"L": "L4F96H252E270624C250724VISTESTFULL.mat",
                 "B": "B4F96H252E270624C250724VISTESTFULL.mat",
                 "N": "N4F96H252E270624C250724VISTESTFULL.mat",
                 "E": "Entrega 2"},
             },
}

# load wavelengths

wavelengths = next(
    v for k, v in loadmat(os.path.join(base_dir, 'wavelengths_VIS.mat')).items() if not k.startswith('__')).squeeze()

# set threshold between 400 and 900 nm

efficiency_threshold = (efficiency_range[0] <= wavelengths) & (wavelengths <= efficiency_range[1])
wavelengths = wavelengths[efficiency_threshold]
num_bands = efficiency_threshold.sum()

# load and build dataset

for subset_name, lot_filenames in full_cocoa_paths.items():
    print(f"Processing {subset_name} subset")

    with h5py.File(os.path.join(out_dir, f'{subset_name}_cocoa_beans_norm_reflect_abs.h5'), 'w') as d:
        dataset = d.create_dataset('spec', shape=(0, num_bands), maxshape=(None, num_bands),
                                   chunks=(256, num_bands), dtype=np.float32)
        labelset = d.create_dataset('label', (0, 1), maxshape=(None, 1), chunks=(256, 1), dtype=np.uint8)

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

            # if '1' in lot_filename['E']:
            #     lot = lot + black.mean(axis=0)[None, ...]

            if debug:
                plt.figure(figsize=(8, 8))
                plt.suptitle(lot_filename['E'] + ' - ' + lot_filename['L'])

                plt.subplot(3, 1, 1)
                plt.plot(wavelengths, white[::white.shape[0] // plot_num_samples + 1].T, alpha=0.5)
                plt.title('White')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.subplot(3, 1, 2)
                plt.plot(wavelengths, black[::black.shape[0] // plot_num_samples + 1].T, alpha=0.5)
                plt.title('Black')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.subplot(3, 1, 3)
                plt.plot(wavelengths, lot[::lot.shape[0] // plot_num_samples + 1].T, alpha=0.5)
                plt.title('Lot')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.tight_layout()
                plt.show()

            # process white and black

            white = white.mean(axis=0)[None, ...]
            black = black.mean(axis=0)[None, ...]
            if white.max() < 50000.0:
                white = white * entrega1_white_scaling

            # get conveyor belt signatures

            conveyor_belt = lot[:conveyor_belt_samples, :]
            cc_distances = compute_sam(lot, conveyor_belt)
            lot_distances = cc_distances.min(axis=-1)
            sorted_indices = np.argsort(lot_distances)[::-1]  # from higher sam to lower
            selected_indices = np.sort(sorted_indices[:max_num_samples])
            selected_cocoa = lot[selected_indices, :]

            if debug:
                plt.figure(figsize=(8, 8))
                plt.suptitle(lot_filename['E'] + ' - ' + lot_filename['L'])

                plt.subplot(3, 1, 1)
                plt.plot(wavelengths, conveyor_belt.T, alpha=0.5)
                plt.title('Conveyor Belt')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.subplot(3, 1, 2)
                plt.plot(np.sort(lot_distances))
                plt.axvline(x=lot_distances.shape[0] - max_num_samples, color='r', linestyle='--',
                            label=f'Threshold for {max_num_samples} samples')
                plt.title('Sorted Lot Distances')
                plt.xlabel('Lot Sample')
                plt.ylabel('SAM')
                plt.grid()
                plt.legend()

                plt.subplot(3, 1, 3)
                plt.plot(wavelengths, selected_cocoa[::selected_cocoa.shape[0] // plot_num_samples + 1].T, alpha=0.5)
                plt.title('Selected Cocoa')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.tight_layout()
                plt.show()

            # get cocoa lot with reflectance

            selected_cocoa_reflectance = (selected_cocoa - black) / (white - black)
            selected_cocoa_reflectance = selected_cocoa_reflectance / selected_cocoa_reflectance.max(axis=-1,
                                                                                                     keepdims=True)

            # # absorbance
            # selected_cocoa_reflectance[selected_cocoa_reflectance <= 0.0] = 1e-3
            # selected_cocoa_reflectance = -np.log(selected_cocoa_reflectance)

            # MSC

            selected_cocoa_reflectance = msc(selected_cocoa_reflectance)

            if debug:
                plt.figure(figsize=(8, 8))
                plt.suptitle(lot_filename['E'] + ' - ' + lot_filename['L'])

                plt.subplot(3, 1, 1)
                plt.plot(wavelengths, white[::white.shape[0] // plot_num_samples + 1].T, alpha=0.5)
                plt.title('White')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.subplot(3, 1, 2)
                plt.plot(wavelengths, black[::black.shape[0] // plot_num_samples + 1].T, alpha=0.5)
                plt.title('Black')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.subplot(3, 1, 3)
                plt.plot(wavelengths,
                         selected_cocoa_reflectance[::selected_cocoa_reflectance.shape[0] // plot_num_samples + 1].T,
                         alpha=0.5)
                plt.title('Selected Cocoa Reflectance')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Reflectance')
                plt.grid()

                plt.tight_layout()
                plt.show()

            # append to dataset

            cocoa_bean_dataset.append(selected_cocoa_reflectance)
            label_dataset.append(np.ones(selected_cocoa_reflectance.shape[0], dtype=int) * label)

            # shuffle and batch mean
            cocoa_bean_batch_mean_aux = []
            for i in range(cocoa_batch_samples):
                random_indices = np.random.choice(selected_cocoa_reflectance.shape[0], cocoa_batch_size, replace=False)
                cocoa_bean_batch_mean_aux.append(selected_cocoa_reflectance[random_indices].mean(axis=0))

            cocoa_bean_batch_mean_aux = np.stack(cocoa_bean_batch_mean_aux, axis=0)
            cocoa_bean_batch_mean_dataset.append(cocoa_bean_batch_mean_aux)
            label_bean_batch_aux = np.ones(cocoa_bean_batch_mean_aux.shape[0], dtype=int) * label
            label_batch_mean_dataset.append(label_bean_batch_aux)

            # append to h5 dataset

            append_to_dataset(dataset, cocoa_bean_batch_mean_aux)
            append_to_dataset(labelset, label_bean_batch_aux[..., None])

            # raise ValueError

    # compute mean and std of dataset and plot

    if debug_pca:
        plt.figure(figsize=(6, 5))
        for i in range(len(cocoa_bean_dataset)):
            X_class = cocoa_bean_dataset[i]
            mean = X_class.mean(axis=0)
            std = X_class.std(axis=0)
            plt.plot(wavelengths, mean, color=colors[i], label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')
            plt.fill_between(wavelengths, mean - std, mean + std, alpha=0.2, color=colors[i], linewidth=0.0)

        plt.legend()
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Reflectance')
        plt.title('Mean and std of Cocoa dataset by class')

        plt.grid()
        plt.tight_layout()
        plt.show()

    # compute PCA with X_mean using sklearn

    full_cocoa_bean_dataset = np.concatenate(cocoa_bean_dataset, axis=0)
    full_label_dataset = np.concatenate(label_dataset, axis=0)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    full_cocoa_bean_dataset = scaler.fit_transform(full_cocoa_bean_dataset)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(full_cocoa_bean_dataset)
    # explained_variance = pca.explained_variance_ratio_

    if debug_pca:
        plt.figure(figsize=(6, 5))
        for i in range(len(cocoa_bean_dataset)):
            X_class = X_pca[full_label_dataset.squeeze() == i]
            plt.scatter(X_class[:, 0], X_class[:, 1], color=colors[i], alpha=0.5, marker=markers[i],
                        label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')

        # plt.xlabel('Component 1: {:.2f}%'.format(explained_variance[0] * 100))
        # plt.ylabel('Component 2: {:.2f}%'.format(explained_variance[1] * 100))

        plt.title(f'Cocoa dataset PCA')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    # plot cocoa bean batch mean dataset

    plt.figure(figsize=(6, 5))
    for i in range(len(cocoa_bean_batch_mean_dataset)):
        X_class = cocoa_bean_batch_mean_dataset[i]
        mean = X_class.mean(axis=0)
        std = X_class.std(axis=0)
        plt.plot(wavelengths, mean, color=colors[i], label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')
        plt.fill_between(wavelengths, mean - std, mean + std, alpha=0.2, color=colors[i], linewidth=0.0)

    plt.legend()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Reflectance')
    plt.title('Mean of Cocoa dataset Batch Mean by class')

    plt.grid()
    plt.tight_layout()
    plt.show()

    # compute PCA with X_mean using sklearn

    full_cocoa_bean_batch_mean_dataset = np.concatenate(cocoa_bean_batch_mean_dataset, axis=0)
    full_label_batch_mean_dataset = np.concatenate(label_batch_mean_dataset, axis=0)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    full_cocoa_bean_batch_mean_dataset = scaler.fit_transform(full_cocoa_bean_batch_mean_dataset)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(full_cocoa_bean_batch_mean_dataset)
    # explained_variance = pca.explained_variance_ratio_

    if debug_pca:
        plt.figure(figsize=(6, 5))
        for i in range(len(cocoa_bean_dataset)):
            X_class = X_pca[full_label_batch_mean_dataset.squeeze() == i]
            plt.scatter(X_class[:, 0], X_class[:, 1], color=colors[i], alpha=0.5, marker=markers[i],
                        label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')

        # plt.xlabel('Component 1: {:.2f}%'.format(explained_variance[0] * 100))
        # plt.ylabel('Component 2: {:.2f}%'.format(explained_variance[1] * 100))

        plt.title(f'Cocoa dataset PCA')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
