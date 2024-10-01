import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat


# functions

def compute_sam(a, b):
    assert a.ndim == 2, "a must have two dimensions, if you only have one, please add an new dimension in the first place"
    assert b.ndim == 2, "b must have two dimensions, if you only have one, please add an new dimension in the first place"

    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.matmul(a, b.T) / np.matmul(a_norm, b_norm.T), a_min=-1.0, a_max=1.0))


# set main paths

base_dir = "/home/enmartz/Jobs/cacao/Base_Datos_Cacao/ALL_VIS"
band_dir = os.path.join(base_dir, "BANDATRANSPORTADORAC090524.mat")
out_dir = os.path.join("built_datasets")
os.makedirs(out_dir, exist_ok=True)

# set variables

efficiency_range = [500, 900]  # nanometers
entrega1_white_scaling = 21.0  # white / this
conveyor_belt_samples = 500  # for sam metric
angle_error = 0.25  # angle error between conveyor belt and cocoa signatures

plot_num_samples = 500
debug = True

# set path to cocoa dataset

full_cocoa_paths = {'train': {0: {"L": "L1F60H096R290324C070524VISTRAIFULL.mat",
                                  "B": "blanco.mat",
                                  "N": "negro.mat",
                                  "E": "Entrega 1"},
                              # 1: {"L": "L2F66H144R310324C070524VISTRAIFULL.mat",
                              #     "B": "blanco.mat",
                              #     "N": "negro.mat",
                              #     "E": "Entrega 1"},
                              2: {"L": "L7F73H144E270624C240724VISTRAIFULL.mat",
                                  "B": "B7F73H144E270624C240724VISTRAIFULL.mat",
                                  "N": "N7F73H144E270624C240724VISTRAIFULL.mat",
                                  "E": "Entrega 2"},
                              3: {"L": "L3F84H192R020424C090524VISTRAIFULL.mat",
                                  "B": "blanco.mat",
                                  "N": "negro.mat",
                                  "E": "Entrega 1"},
                              4: {"L": "L6F85H110E270624C240724VISTRAIFULL.mat",
                                  "B": "B6F85H110E270624C240724VISTRAIFULL.mat",
                                  "N": "N6F85H110E270624C240724VISTRAIFULL.mat",
                                  "E": "Entrega 2"},
                              5: {"L": "L4F92H264R130424C090524VISTRAIFULL.mat",
                                  "B": "blanco.mat",
                                  "N": "negro.mat",
                                  "E": "Entrega 1"},
                              6: {"L": "L8F94H216E270624C240724VISTRAIFULL.mat",
                                  "B": "B8F94H216E270624C240724VISTRAIFULL.mat",
                                  "N": "N8F94H216E270624C240724VISTRAIFULL.mat",
                                  "E": "Entrega 2"},
                              7: {"L": "L5F96HXXXRDDMMAAC090524VISTRAIFULL.mat",
                                  "B": "blanco.mat",
                                  "N": "negro.mat",
                                  "E": "Entrega 1"},
                              8: {"L": "L9F96H252E270624C240724VISTRAIFULL.mat",
                                  "B": "B9F96H252E270624C240724VISTRAIFULL.mat",
                                  "N": "N9F96H252E270624C240724VISTRAIFULL.mat",
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
                             2: {"L": "L7F73H144E270624C250724VISTESTFULL.mat",
                                 "B": "B7F73H144E270624C250724VISTESTFULL.mat",
                                 "N": "N7F73H144E270624C250724VISTESTFULL.mat",
                                 "E": "Entrega 2"},
                             3: {"L": "L3F84H192R020424C090524VISTESTFULL.mat",
                                 "B": "blanco.mat",
                                 "N": "negro.mat",
                                 "E": "Entrega 1"},
                             4: {"L": "L6F85H110E270624C250724VISTESTFULL.mat",
                                 "B": "B6F85H110E270624C250724VISTESTFULL.mat",
                                 "N": "N6F85H110E270624C250724VISTESTFULL.mat",
                                 "E": "Entrega 2"},
                             5: {"L": "L4F92H264R130424C090524VISTESTFULL.mat",
                                 "B": "blanco.mat",
                                 "N": "negro.mat",
                                 "E": "Entrega 1"},
                             6: {"L": "L8F94H216E270624C250724VISTESTFULL.mat",
                                 "B": "B8F94H216E270624C250724VISTESTFULL.mat",
                                 "N": "N8F94H216E270624C250724VISTESTFULL.mat",
                                 "E": "Entrega 2"},
                             7: {"L": "L5F96HXXXRDDMMAAC090524VISTESTFULL.mat",
                                 "B": "blanco.mat",
                                 "N": "negro.mat",
                                 "E": "Entrega 1"},
                             8: {"L": "L9F96H252E270624C250724VISTESTFULL.mat",
                                 "B": "B9F96H252E270624C250724VISTESTFULL.mat",
                                 "N": "N9F96H252E270624C250724VISTESTFULL.mat",
                                  "E": "Entrega 2"}},
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

    for label, lot_filename in lot_filenames.items():
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

        if debug and False:
            plt.figure(figsize=(15, 8))
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
        if white.max() > 50000.0:
            white = white / entrega1_white_scaling

        # get conveyor belt signatures

        conveyor_belt = lot[:conveyor_belt_samples, :]
        cc_distances = compute_sam(lot, conveyor_belt)
        lot_distances = cc_distances.min(axis=-1)

        if debug:
            plt.figure(figsize=(15, 8))
            plt.suptitle(lot_filename['E'] + ' - ' + lot_filename['L'])

            plt.subplot(3, 1, 1)
            plt.plot(wavelengths, conveyor_belt.T, alpha=0.5)
            plt.title('Conveyor Belt')
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Intensity')
            plt.grid()

            plt.subplot(3, 1, 2)
            plt.plot(lot_distances)
            plt.axline((0, angle_error), slope=0, color='r', linestyle='--', label='Threshold')
            plt.title('Lot Distances')
            plt.xlabel('Lot Sample')
            plt.ylabel('SAM')
            plt.grid()
            plt.legend()

            plt.subplot(3, 1, 3)
            plt.plot(np.sort(lot_distances))
            plt.axline((0, angle_error), slope=0, color='r', linestyle='--', label='Threshold')
            plt.title('Sorted Lot Distances')
            plt.xlabel('Lot Sample')
            plt.ylabel('SAM')
            plt.grid()
            plt.legend()

            plt.tight_layout()
            plt.show()

        # get cocoa lot with reflectance

        cocoa_mask = lot_distances > angle_error
        cocoa = lot[cocoa_mask, :]
        cocoa_reflectance = (cocoa - black) / (white - black)

        if debug and False:
            plt.figure(figsize=(15, 8))
            plt.suptitle(lot_filename['E'] + ' - ' + lot_filename['L'])

            plt.subplot(2, 1, 1)
            plt.plot(wavelengths, cocoa[::cocoa.shape[0] // plot_num_samples + 1].T, alpha=0.5)
            plt.title('Selected Cocoa Intensity')
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Intensity')
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot(wavelengths, cocoa_reflectance[::cocoa_reflectance.shape[0] // plot_num_samples + 1].T, alpha=0.5)
            plt.title('Selected Cocoa Reflectance')
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Reflectance')
            plt.grid()

            plt.tight_layout()
            plt.show()

        print('taipo')
