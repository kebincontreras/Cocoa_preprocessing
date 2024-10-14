import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Directorio donde se encuentran los archivos
base_dir = r"C:\Users\USUARIO\Documents\GitHub\Dataset\Lab_hdsp_cocoa_experimento_jorge"

# Cargar los archivos
cacao_NIR = loadmat(os.path.join(base_dir, 'experimento_cacao_3_fermantation_NIR.mat'))

# Obtener las longitudes de onda
wavelength = cacao_NIR['wavelengths'].squeeze()

# Filtrar las longitudes de onda en el rango de 1100 a 2100 nm
mask = (wavelength >= 1100) & (wavelength <= 2100)

# Filtrar las longitudes de onda
filtered_wavelength = wavelength[mask]

# Filtrar los datos basados en las longitudes de onda seleccionadas
def filter_data(data, mask):
    return data[:, mask]

# Asegurarse de que el divisor tenga la misma longitud que los datos
blanconir = filter_data(cacao_NIR['my_blanconir'], mask)

# Normalizar los datos
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.T).T

# Estandarizar los datos
def standardize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data.T).T

# Aplicar el filtro a los datos, dividir por el blanconir filtrado, y luego normalizar y estandarizar
cocoa_NIR_normalized = dict(
    open73=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_73_abierto'], mask) / blanconir), label='neutral_73_abierto'),
    closed73=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_73_cerrado'], mask) / blanconir), label='neutral_73_cerrado'),
    open73_2=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_73_abierto_2'], mask) / blanconir), label='bad_73_abierto_2'),
    closed73_2=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_73_cerrado_2'], mask) / blanconir), label='bad_73_cerrado_2'),
    open85=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_85_abierto'], mask) / blanconir), label='bad_85_abierto'),
    closed85=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_85_cerrado'], mask) / blanconir), label='bad_85_cerrado'),
    open85_2=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_85_abierto_2'], mask) / blanconir), label='good_85_abierto_2'),
    closed85_2=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_85_cerrado_2'], mask) / blanconir), label='good_85_cerrado_2'),
    open94=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_94_abierto'], mask) / blanconir), label='good_94_abierto'),
    closed94=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_94_cerrado'], mask) / blanconir), label='good_94_cerrado'),
    open94_2=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_94_abierto_2'], mask) / blanconir), label='bad_94_abierto_2'),
    closed94_2=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_94_cerrado_2'], mask) / blanconir), label='bad_94_cerrado_2'),
    open96=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_96_abierto'], mask) / blanconir), label='good_96_abierto'),
    closed96=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_96_cerrado'], mask) / blanconir), label='good_96_cerrado'),
    open96_2=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_96_abierto_2'], mask) / blanconir), label='good_96_abierto_2'),
    closed96_2=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_96_cerrado_2'], mask) / blanconir), label='good_96_cerrado_2')
)

cocoa_NIR_standardized = dict(
    open73=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_73_abierto'], mask) / blanconir), label='neutral_73_abierto'),
    closed73=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_73_cerrado'], mask) / blanconir), label='neutral_73_cerrado'),
    open73_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_73_abierto_2'], mask) / blanconir), label='bad_73_abierto_2'),
    closed73_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_73_cerrado_2'], mask) / blanconir), label='bad_73_cerrado_2'),
    open85=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_85_abierto'], mask) / blanconir), label='bad_85_abierto'),
    closed85=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_85_cerrado'], mask) / blanconir), label='bad_85_cerrado'),
    open85_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_85_abierto_2'], mask) / blanconir), label='good_85_abierto_2'),
    closed85_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_85_cerrado_2'], mask) / blanconir), label='good_cerrado'),
    open94=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_94_abierto'], mask) / blanconir), label='good_94_abierto'),
    closed94=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_94_cerrado'], mask) / blanconir), label='good_94_cerrado'),
    open94_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_94_abierto_2'], mask) / blanconir), label='bad_94_abierto_2'),
    closed94_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_94_cerrado_2'], mask) / blanconir), label='bad_94_cerrado_2'),
    open96=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_96_abierto'], mask) / blanconir), label='good_96_abierto'),
    closed96=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_96_cerrado'], mask) / blanconir), label='good_96_cerrado'),
    open96_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_96_abierto_2'], mask) / blanconir), label='good_96_abierto_2'),
    closed96_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_96_cerrado_2'], mask) / blanconir), label='good_96_cerrado_2')
)

# Crear subplots para normalización y estandarización (Figura 1)
fig1, axs1 = plt.subplots(2, 1, figsize=(12, 12))

# Plot Normalización
axs1[0].set_title('Normalized [MinMax] NIR Spectral Signatures')
for key, value in cocoa_NIR_normalized.items():
    axs1[0].plot(filtered_wavelength, value['data'].squeeze(), label=value['label'])
axs1[0].set_xlabel('Wavelength (nm)')
axs1[0].set_ylabel('Normalized Reflectance')
axs1[0].legend()

# Plot Estandarización
axs1[1].set_title('Standardized [(x-m)\s] NIR Spectral Signatures')
for key, value in cocoa_NIR_standardized.items():
    axs1[1].plot(filtered_wavelength, value['data'].squeeze(), label=value['label'])
axs1[1].set_xlabel('Wavelength (nm)')
axs1[1].set_ylabel('Standardized Reflectance')
axs1[1].legend()

plt.tight_layout()
plt.show()

# Crear subplots para diferenciar abierto y cerrado (Figura 2)
fig2, axs2 = plt.subplots(2, 1, figsize=(12, 12))

# Plot con colores diferenciados para abierto y cerrado (Normalizado)
axs2[0].set_title('Normalized [MinMax] NIR Spectral Signatures (Open vs. Closed)')
for key, value in cocoa_NIR_normalized.items():
    if 'abierto' in value['label']:
        axs2[0].plot(filtered_wavelength, value['data'].squeeze(), label=value['label'], color='blue')
    elif 'cerrado' in value['label']:
        axs2[0].plot(filtered_wavelength, value['data'].squeeze(), label=value['label'], color='red')
axs2[0].set_xlabel('Wavelength (nm)')
axs2[0].set_ylabel('Normalized Reflectance')
axs2[0].legend()

# Plot con colores diferenciados para abierto y cerrado (Estandarizado)
axs2[1].set_title('Standardized [(x-m)\s] NIR Spectral Signatures (Open vs. Closed)')
for key, value in cocoa_NIR_standardized.items():
    if 'abierto' in value['label']:
        axs2[1].plot(filtered_wavelength, value['data'].squeeze(), label=value['label'], color='blue')
    elif 'cerrado' in value['label']:
        axs2[1].plot(filtered_wavelength, value['data'].squeeze(), label=value['label'], color='red')
axs2[1].set_xlabel('Wavelength (nm)')
axs2[1].set_ylabel('Standardized Reflectance')
axs2[1].legend()

plt.tight_layout()
plt.show()

# Calcular y graficar la media de las firmas para "bad", "neutral" y "good" (Normalizado)
bad_normalized = np.mean([value['data'] for key, value in cocoa_NIR_normalized.items() if 'bad' in value['label']], axis=0)
neutral_normalized = np.mean([value['data'] for key, value in cocoa_NIR_normalized.items() if 'neutral' in value['label']], axis=0)
good_normalized = np.mean([value['data'] for key, value in cocoa_NIR_normalized.items() if 'good' in value['label']], axis=0)

# Calcular y graficar la media de las firmas para "bad", "neutral" y "good" (Estandarizado)
bad_standardized = np.mean([value['data'] for key, value in cocoa_NIR_standardized.items() if 'bad' in value['label']], axis=0)
neutral_standardized = np.mean([value['data'] for key, value in cocoa_NIR_standardized.items() if 'neutral' in value['label']], axis=0)
good_standardized = np.mean([value['data'] for key, value in cocoa_NIR_standardized.items() if 'good' in value['label']], axis=0)

# Crear una nueva figura para las medias
fig3, ax = plt.subplots(2, 1, figsize=(12, 12))

# Graficar las medias normalizadas
ax[0].set_title('Mean NIR Spectral Signatures (Normalized)')
ax[0].plot(filtered_wavelength, bad_normalized.squeeze(), label='Bad', color='red')
ax[0].plot(filtered_wavelength, neutral_normalized.squeeze(), label='Neutral', color='green')
ax[0].plot(filtered_wavelength, good_normalized.squeeze(), label='Good', color='blue')
ax[0].set_xlabel('Wavelength (nm)')
ax[0].set_ylabel('Mean Normalized Reflectance')
ax[0].legend()

# Graficar las medias estandarizadas
ax[1].set_title('Mean NIR Spectral Signatures (Standardized)')
ax[1].plot(filtered_wavelength, bad_standardized.squeeze(), label='Bad', color='red')
ax[1].plot(filtered_wavelength, neutral_standardized.squeeze(), label='Neutral', color='green')
ax[1].plot(filtered_wavelength, good_standardized.squeeze(), label='Good', color='blue')
ax[1].set_xlabel('Wavelength (nm)')
ax[1].set_ylabel('Mean Standardized Reflectance')
ax[1].legend()

plt.tight_layout()
plt.show()
