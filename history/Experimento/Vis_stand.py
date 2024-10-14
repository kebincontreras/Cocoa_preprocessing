import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat

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

# Concatenar todas las firmas para calcular la media y desviación estándar
all_signatures = np.concatenate([
    filter_data(cacao1_VIS['my_cacao_50_abierto_1'], mask),
    filter_data(cacao1_VIS['my_cacao_50_cerrado_1'], mask),
    filter_data(cacao2_VIS['my_cacao_73_abierto_1'], mask),
    filter_data(cacao2_VIS['my_cacao_73_cerrado_1'], mask),
    filter_data(cacao2_VIS['my_cacao_73_abierto_2'], mask),
    filter_data(cacao2_VIS['my_cacao_73_cerrado_2'], mask),
    filter_data(cacao1_VIS['my_cacao_95_abierto_1'], mask),
    filter_data(cacao1_VIS['my_cacao_95_cerrado_1'], mask),
    filter_data(cacao2_VIS['my_cacao_96_abierto_1'], mask),
    filter_data(cacao2_VIS['my_cacao_96_cerrado_1'], mask)
])

# Dividir todas las firmas por el blanco_teflon_n filtrado
normalized_all_signatures = all_signatures / blanco_teflon_n

# Calcular la media y desviación estándar por banda
mean_all_signatures = np.mean(normalized_all_signatures, axis=0)
std_all_signatures = np.std(normalized_all_signatures, axis=0)

# Imprimir los vectores de media y desviación estándar
print("Media por banda:")
print(mean_all_signatures)
print("Desviación estándar por banda:")
print(std_all_signatures)

# Estandarizar los datos utilizando la media y desviación estándar calculadas
def standardize_data(data, mean, std):
    return (data - mean) / std

# Aplicar el filtro a los datos, dividir por el blanco_teflon_n filtrado, y luego estandarizar
def process_data(data, blanco_teflon_n, mask, mean, std):
    filtered_data = filter_data(data, mask)
    normalized_data = filtered_data / blanco_teflon_n
    standardized_data = standardize_data(normalized_data, mean, std)
    return standardized_data

cocoa_VIS_standardized = dict(
    open50_1=dict(data=process_data(cacao1_VIS['my_cacao_50_abierto_1'], blanco_teflon_n, mask, mean_all_signatures, std_all_signatures), label='bad_50_abierto_1'),
    closed50_1=dict(data=process_data(cacao1_VIS['my_cacao_50_cerrado_1'], blanco_teflon_n, mask, mean_all_signatures, std_all_signatures), label='bad_50_cerrado_1'),
    open73_1=dict(data=process_data(cacao2_VIS['my_cacao_73_abierto_1'], blanco_teflon_n, mask, mean_all_signatures, std_all_signatures), label='neutral_73_abierto_1'),
    closed73_1=dict(data=process_data(cacao2_VIS['my_cacao_73_cerrado_1'], blanco_teflon_n, mask, mean_all_signatures, std_all_signatures), label='neutral_73_cerrado_1'),
    open73_2=dict(data=process_data(cacao2_VIS['my_cacao_73_abierto_2'], blanco_teflon_n, mask, mean_all_signatures, std_all_signatures), label='good_73_abierto_2'),
    closed73_2=dict(data=process_data(cacao2_VIS['my_cacao_73_cerrado_2'], blanco_teflon_n, mask, mean_all_signatures, std_all_signatures), label='good_73_cerrado_2'),
    open95_1=dict(data=process_data(cacao1_VIS['my_cacao_95_abierto_1'], blanco_teflon_n, mask, mean_all_signatures, std_all_signatures), label='bad_95_abierto_1'),
    closed95_1=dict(data=process_data(cacao1_VIS['my_cacao_95_cerrado_1'], blanco_teflon_n, mask, mean_all_signatures, std_all_signatures), label='bad_95_cerrado_1'),
    open96_1=dict(data=process_data(cacao2_VIS['my_cacao_96_abierto_1'], blanco_teflon_n, mask, mean_all_signatures, std_all_signatures), label='good_96_abierto_1'),
    closed96_1=dict(data=process_data(cacao2_VIS['my_cacao_96_cerrado_1'], blanco_teflon_n, mask, mean_all_signatures, std_all_signatures), label='good_96_cerrado_1')
)

# Crear subplots para estandarización (Figura 1)
fig1, ax1 = plt.subplots(figsize=(12, 6))

# Plot Estandarización
ax1.set_title('Standardized [(x-m)/s] VIS Spectral Signatures')
for key, value in cocoa_VIS_standardized.items():
    ax1.plot(filtered_wavelength, value['data'].squeeze(), label=value['label'])
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Standardized Reflectance')
ax1.legend()

plt.tight_layout()
plt.show()

# Crear subplots para diferenciar abierto y cerrado (Figura 2)
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Plot con colores diferenciados para abierto y cerrado (Estandarizado)
ax2.set_title('Standardized [(x-m)/s] VIS Spectral Signatures (Open vs. Closed)')
for key, value in cocoa_VIS_standardized.items():
    if 'abierto' in value['label']:
        ax2.plot(filtered_wavelength, value['data'].squeeze(), label=value['label'], color='blue')
    elif 'cerrado' in value['label']:
        ax2.plot(filtered_wavelength, value['data'].squeeze(), label=value['label'], color='red')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Standardized Reflectance')
ax2.legend()

plt.tight_layout()
plt.show()

# Calcular y graficar la media de las firmas para "bad", "neutral" y "good" (Estandarizado)
bad_standardized = np.mean([value['data'] for key, value in cocoa_VIS_standardized.items() if 'bad' in value['label']], axis=0)
neutral_standardized = np.mean([value['data'] for key, value in cocoa_VIS_standardized.items() if 'neutral' in value['label']], axis=0)
good_standardized = np.mean([value['data'] for key, value in cocoa_VIS_standardized.items() if 'good' in value['label']], axis=0)

# Crear una nueva figura para las medias
fig3, ax = plt.subplots(figsize=(12, 6))

# Graficar las medias estandarizadas
ax.set_title('Mean VIS Spectral Signatures (Standardized)')
ax.plot(filtered_wavelength, bad_standardized.squeeze(), label='Bad', color='red')
ax.plot(filtered_wavelength, neutral_standardized.squeeze(), label='Neutral', color='green')
ax.plot(filtered_wavelength, good_standardized.squeeze(), label='Good', color='blue')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Mean Standardized Reflectance')
ax.legend()

plt.tight_layout()
plt.show()
