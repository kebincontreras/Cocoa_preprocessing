import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Directorio donde se encuentran los archivos
base_dir = r"C:\Users\USUARIO\Documents\GitHub\Dataset\Lab_hdsp_cocoa_experimento_jorge"

# Cargar los archivos
cacao1_VIS = loadmat(os.path.join(base_dir, 'experimento_cacao_1_fermentation.mat'))
cacao2_VIS = loadmat(os.path.join(base_dir, 'experimento_cacao_2_fermentation.mat'))

# Obtener las longitudes de onda
wavelength = cacao1_VIS['wavelengths'].squeeze()

# Filtrar las longitudes de onda en el rango de 450 a 910 nm
mask = (wavelength >= 450) & (wavelength <= 910)

# Filtrar las longitudes de onda
filtered_wavelength = wavelength[mask]

# Filtrar los datos basados en las longitudes de onda seleccionadas
def filter_data(data, mask):
    return data[:, mask]

# Asegurarse de que el divisor tenga la misma longitud que los datos
blanco_teflon_n = filter_data(cacao1_VIS['my_blanco_teflon_n'], mask)

# Normalizar los datos
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.T).T

# Estandarizar los datos
def standardize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data.T).T

# Aplicar el filtro a los datos, dividir por el blanco_teflon_n filtrado, y luego normalizar y estandarizar
cocoa_VIS_normalized = dict(
    open50_1=dict(data=normalize_data(filter_data(cacao1_VIS['my_cacao_50_abierto_1'], mask) / blanco_teflon_n), label='bad_abierto'),
    closed50_1=dict(data=normalize_data(filter_data(cacao1_VIS['my_cacao_50_cerrado_1'], mask) / blanco_teflon_n), label='bad_cerrado'),
    open73_1=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_73_abierto_1'], mask) / blanco_teflon_n), label='neutral_abierto'),
    closed73_1=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_73_cerrado_1'], mask) / blanco_teflon_n), label='neutral_cerrado'),
    open73_2=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_73_abierto_2'], mask) / blanco_teflon_n), label='good_abierto'),
    closed73_2=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_73_cerrado_2'], mask) / blanco_teflon_n), label='good_cerrado'),
    open95_1=dict(data=normalize_data(filter_data(cacao1_VIS['my_cacao_95_abierto_1'], mask) / blanco_teflon_n), label='bad_abierto'),
    closed95_1=dict(data=normalize_data(filter_data(cacao1_VIS['my_cacao_95_cerrado_1'], mask) / blanco_teflon_n), label='bad_cerrado'),
    open96_1=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_96_abierto_1'], mask) / blanco_teflon_n), label='good_abierto'),
    closed96_1=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_96_cerrado_1'], mask) / blanco_teflon_n), label='good_cerrado')
)

cocoa_VIS_standardized = dict(
    open50_1=dict(data=standardize_data(filter_data(cacao1_VIS['my_cacao_50_abierto_1'], mask) / blanco_teflon_n), label='bad_abierto'),
    closed50_1=dict(data=standardize_data(filter_data(cacao1_VIS['my_cacao_50_cerrado_1'], mask) / blanco_teflon_n), label='bad_cerrado'),
    open73_1=dict(data=standardize_data(filter_data(cacao2_VIS['my_cacao_73_abierto_1'], mask) / blanco_teflon_n), label='neutral_abierto'),
    closed73_1=dict(data=standardize_data(filter_data(cacao2_VIS['my_cacao_73_cerrado_1'], mask) / blanco_teflon_n), label='neutral_cerrado'),
    open73_2=dict(data=standardize_data(filter_data(cacao2_VIS['my_cacao_73_abierto_2'], mask) / blanco_teflon_n), label='good_abierto'),
    closed73_2=dict(data=standardize_data(filter_data(cacao2_VIS['my_cacao_73_cerrado_2'], mask) / blanco_teflon_n), label='good_cerrado'),
    open95_1=dict(data=standardize_data(filter_data(cacao1_VIS['my_cacao_95_abierto_1'], mask) / blanco_teflon_n), label='bad_abierto'),
    closed95_1=dict(data=standardize_data(filter_data(cacao1_VIS['my_cacao_95_cerrado_1'], mask) / blanco_teflon_n), label='bad_cerrado'),
    open96_1=dict(data=standardize_data(filter_data(cacao2_VIS['my_cacao_96_abierto_1'], mask) / blanco_teflon_n), label='good_abierto'),
    closed96_1=dict(data=standardize_data(filter_data(cacao2_VIS['my_cacao_96_cerrado_1'], mask) / blanco_teflon_n), label='good_cerrado')
)

# Crear subplots para normalización y estandarización
fig, axs = plt.subplots(5, 1, figsize=(12, 30))

# Plot Normalización
axs[0].set_title('Normalized VIS Spectral Signatures')
for key, value in cocoa_VIS_normalized.items():
    axs[0].plot(filtered_wavelength, value['data'].squeeze(), label=value['label'])
axs[0].set_xlabel('Wavelength (nm)')
axs[0].set_ylabel('Normalized Reflectance')
axs[0].legend()

# Plot Estandarización
axs[1].set_title('Standardized VIS Spectral Signatures')
for key, value in cocoa_VIS_standardized.items():
    axs[1].plot(filtered_wavelength, value['data'].squeeze(), label=value['label'])
axs[1].set_xlabel('Wavelength (nm)')
axs[1].set_ylabel('Standardized Reflectance')
axs[1].legend()

# Plot con colores diferenciados para abierto y cerrado (Normalizado)
axs[2].set_title('Normalized VIS Spectral Signatures (Open vs. Closed)')
for key, value in cocoa_VIS_normalized.items():
    if 'abierto' in value['label']:
        axs[2].plot(filtered_wavelength, value['data'].squeeze(), label=value['label'], color='blue')
    elif 'cerrado' in value['label']:
        axs[2].plot(filtered_wavelength, value['data'].squeeze(), label=value['label'], color='red')
axs[2].set_xlabel('Wavelength (nm)')
axs[2].set_ylabel('Normalized Reflectance')
axs[2].legend()

# Plot con colores diferenciados para abierto y cerrado (Estandarizado)
axs[3].set_title('Standardized VIS Spectral Signatures (Open vs. Closed)')
for key, value in cocoa_VIS_standardized.items():
    if 'abierto' in value['label']:
        axs[3].plot(filtered_wavelength, value['data'].squeeze(), label=value['label'], color='blue')
    elif 'cerrado' in value['label']:
        axs[3].plot(filtered_wavelength, value['data'].squeeze(), label=value['label'], color='red')
axs[3].set_xlabel('Wavelength (nm)')
axs[3].set_ylabel('Standardized Reflectance')
axs[3].legend()

# Calcular y graficar la media de las firmas para "bad", "neutral" y "good" (Normalizado)
bad_normalized = np.mean([value['data'] for key, value in cocoa_VIS_normalized.items() if 'bad' in value['label']], axis=0)
neutral_normalized = np.mean([value['data'] for key, value in cocoa_VIS_normalized.items() if 'neutral' in value['label']], axis=0)
good_normalized = np.mean([value['data'] for key, value in cocoa_VIS_normalized.items() if 'good' in value['label']], axis=0)

axs[4].set_title('Mean VIS Spectral Signatures (Normalized)')
axs[4].plot(filtered_wavelength, bad_normalized.squeeze(), label='Bad', color='red')
axs[4].plot(filtered_wavelength, neutral_normalized.squeeze(), label='Neutral', color='green')
axs[4].plot(filtered_wavelength, good_normalized.squeeze(), label='Good', color='blue')
axs[4].set_xlabel('Wavelength (nm)')
axs[4].set_ylabel('Mean Normalized Reflectance')
axs[4].legend()

# Calcular y graficar la media de las firmas para "bad", "neutral" y "good" (Estandarizado)
bad_standardized = np.mean([value['data'] for key, value in cocoa_VIS_standardized.items() if 'bad' in value['label']], axis=0)
neutral_standardized = np.mean([value['data'] for key, value in cocoa_VIS_standardized.items() if 'neutral' in value['label']], axis=0)
good_standardized = np.mean([value['data'] for key, value in cocoa_VIS_standardized.items() if 'good' in value['label']], axis=0)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_title('Mean VIS Spectral Signatures (Standardized)')
ax.plot(filtered_wavelength, bad_standardized.squeeze(), label='Bad', color='red')
ax.plot(filtered_wavelength, neutral_standardized.squeeze(), label='Neutral', color='green')
ax.plot(filtered_wavelength, good_standardized.squeeze(), label='Good', color='blue')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Mean Standardized Reflectance')
ax.legend()

plt.tight_layout()
plt.show()
