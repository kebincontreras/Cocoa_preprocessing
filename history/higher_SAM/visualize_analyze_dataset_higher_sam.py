import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat

from sklearn.decomposition import PCA

# Definir los parámetros estéticos
fs = 16
params_f = {
    "ytick.color": "black",
    "xtick.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "text.usetex": False,  # Cambiado a False para evitar el uso de LaTeX
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    'font.size': fs,
    'svg.fonttype': 'none'  # Preserve text as selectable SVG text
}

# Aplicar los parámetros estéticos
plt.rcParams.update(params_f)

# functions

def compute_sam(a, b):
    assert a.ndim == 2, "a must have two dimensions, if you only have one, please add an new dimension in the first place"
    assert b.ndim == 2, "b must have two dimensions, if you only have one, please add an new dimension in the first place"

    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.matmul(a, b.T) / np.matmul(a_norm, b_norm.T), a_min=-1.0, a_max=1.0))


# set main paths

base_dir = "/home/enmartz/Jobs/cacao/Base_Datos_Cacao/ALL_VIS"
out_dir = os.path.join("../../built_datasets")
os.makedirs(out_dir, exist_ok=True)

# set variables

efficiency_range = [500, 900]  # nanometers
entrega1_white_scaling = 21.0  # white / this
conveyor_belt_samples = 200  # for sam metric
angle_error = 0.25  # angle error between conveyor belt and cocoa signatures
max_num_samples = 1000  # selected samples from lot with higher sam

cocoa_batch_size = 50  # guillotine methodology
cocoa_batch_samples = 1000  # number of batch samples

plot_num_samples = 500
debug = False

# set path to cocoa dataset

full_cocoa_paths = {'train': {0: {"L": "L1F60H096R290324C070524VISTRAIFULL.mat",
                                  "B": "blanco.mat",
                                  "N": "negro.mat",
                                  "E": "Entrega 1"},
                              4: {"L": "L6F85H110E270624C240724VISTRAIFULL.mat",
                                  "B": "B6F85H110E270624C240724VISTRAIFULL.mat",
                                  "N": "N6F85H110E270624C240724VISTRAIFULL.mat",
                                  "E": "Entrega 2"},
                              7: {"L": "L9F96H252E270624C240724VISTRAIFULL.mat",
                                  "B": "B9F96H252E270624C240724VISTRAIFULL.mat",
                                  "N": "N9F96H252E270624C240724VISTRAIFULL.mat",
                                  "E": "Entrega 2"},
                              },
                    }

# load wavelengths

wavelengths = next(
    v for k, v in loadmat(os.path.join(base_dir, 'wavelengths_VIS.mat')).items() if not k.startswith('__')).squeeze()

# set threshold between 400 and 900 nm

efficiency_threshold = (efficiency_range[0] <= wavelengths) & (wavelengths <= efficiency_range[1])
wavelengths = wavelengths[efficiency_threshold]

# load and build dataset

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
        lot = np.delete(lot, 8719, axis=0) if lot_filename == 'L2F66R310324C070524TESTFULL.mat' else lot

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
        label_batch_mean_dataset.append(np.ones(cocoa_bean_batch_mean_aux.shape[0], dtype=int) * label)

        plt.figure(figsize=(6, 4))
        plt.suptitle(lot_filename['E'] + ' - ' + lot_filename['L'])

        plt.plot(wavelengths,
                 selected_cocoa_reflectance[::selected_cocoa_reflectance.shape[0] // plot_num_samples + 1].T,
                 alpha=0.5)
        plt.title('Selected Cocoa Reflectance')
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Reflectance')
        plt.grid()

        plt.tight_layout()
        plt.savefig('signatures_' + lot_filename['E'] + ' - ' + lot_filename['L'] + '.svg')
        plt.show()
