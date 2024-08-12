import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat

# Directorio donde se encuentran los archivos
# base_dir = r"C:\Users\USUARIO\Documents\GitHub\Dataset\Lab_hdsp_cocoa_experimento_jorge"
base_dir = "."

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

# Concatenar todas las firmas para calcular la media y desviación estándar
all_signatures = np.concatenate([
    filter_data(cacao_NIR['my_cacaocnir_73_abierto'], mask),
    filter_data(cacao_NIR['my_cacaocnir_73_cerrado'], mask),
    filter_data(cacao_NIR['my_cacaocnir_73_abierto_2'], mask),
    filter_data(cacao_NIR['my_cacaocnir_73_cerrado_2'], mask),
    filter_data(cacao_NIR['my_cacaocnir_85_abierto'], mask),
    filter_data(cacao_NIR['my_cacaocnir_85_cerrado'], mask),
    filter_data(cacao_NIR['my_cacaocnir_85_abierto_2'], mask),
    filter_data(cacao_NIR['my_cacaocnir_85_cerrado_2'], mask),
    filter_data(cacao_NIR['my_cacaocnir_94_abierto'], mask),
    filter_data(cacao_NIR['my_cacaocnir_94_cerrado'], mask),
    filter_data(cacao_NIR['my_cacaocnir_94_abierto_2'], mask),
    filter_data(cacao_NIR['my_cacaocnir_94_cerrado_2'], mask),
    filter_data(cacao_NIR['my_cacaocnir_96_abierto'], mask),
    filter_data(cacao_NIR['my_cacaocnir_96_cerrado'], mask),
    filter_data(cacao_NIR['my_cacaocnir_96_abierto_2'], mask),
    filter_data(cacao_NIR['my_cacaocnir_96_cerrado_2'], mask)
])

# Dividir todas las firmas por el blanconir filtrado
normalized_all_signatures = all_signatures / blanconir

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

# Aplicar el filtro a los datos, dividir por el blanconir filtrado, y luego estandarizar
def process_data(data, blanconir, mask, mean, std):
    filtered_data = filter_data(data, mask)
    normalized_data = filtered_data / blanconir
    standardized_data = standardize_data(normalized_data, mean, std)
    return standardized_data

cocoa_NIR_standardized = dict(
    open73=dict(data=process_data(cacao_NIR['my_cacaocnir_73_abierto'], blanconir, mask, mean_all_signatures, std_all_signatures), label='neutral_73_abierto'),
    closed73=dict(data=process_data(cacao_NIR['my_cacaocnir_73_cerrado'], blanconir, mask, mean_all_signatures, std_all_signatures), label='neutral_73_cerrado'),
    open73_2=dict(data=process_data(cacao_NIR['my_cacaocnir_73_abierto_2'], blanconir, mask, mean_all_signatures, std_all_signatures), label='bad_73_abierto_2'),
    closed73_2=dict(data=process_data(cacao_NIR['my_cacaocnir_73_cerrado_2'], blanconir, mask, mean_all_signatures, std_all_signatures), label='bad_73_cerrado_2'),
    open85=dict(data=process_data(cacao_NIR['my_cacaocnir_85_abierto'], blanconir, mask, mean_all_signatures, std_all_signatures), label='bad_85_abierto'),
    closed85=dict(data=process_data(cacao_NIR['my_cacaocnir_85_cerrado'], blanconir, mask, mean_all_signatures, std_all_signatures), label='bad_85_cerrado'),
    open85_2=dict(data=process_data(cacao_NIR['my_cacaocnir_85_abierto_2'], blanconir, mask, mean_all_signatures, std_all_signatures), label='good_85_abierto_2'),
    closed85_2=dict(data=process_data(cacao_NIR['my_cacaocnir_85_cerrado_2'], blanconir, mask, mean_all_signatures, std_all_signatures), label='good_85_cerrado_2'),
    open94=dict(data=process_data(cacao_NIR['my_cacaocnir_94_abierto'], blanconir, mask, mean_all_signatures, std_all_signatures), label='good_94_abierto'),
    closed94=dict(data=process_data(cacao_NIR['my_cacaocnir_94_cerrado'], blanconir, mask, mean_all_signatures, std_all_signatures), label='good_94_cerrado'),
    open94_2=dict(data=process_data(cacao_NIR['my_cacaocnir_94_abierto_2'], blanconir, mask, mean_all_signatures, std_all_signatures), label='bad_94_abierto_2'),
    closed94_2=dict(data=process_data(cacao_NIR['my_cacaocnir_94_cerrado_2'], blanconir, mask, mean_all_signatures, std_all_signatures), label='bad_94_cerrado_2'),
    open96=dict(data=process_data(cacao_NIR['my_cacaocnir_96_abierto'], blanconir, mask, mean_all_signatures, std_all_signatures), label='good_96_abierto'),
    closed96=dict(data=process_data(cacao_NIR['my_cacaocnir_96_cerrado'], blanconir, mask, mean_all_signatures, std_all_signatures), label='good_96_cerrado'),
    open96_2=dict(data=process_data(cacao_NIR['my_cacaocnir_96_abierto_2'], blanconir, mask, mean_all_signatures, std_all_signatures), label='good_96_abierto_2'),
    closed96_2=dict(data=process_data(cacao_NIR['my_cacaocnir_96_cerrado_2'], blanconir, mask, mean_all_signatures, std_all_signatures), label='good_96_cerrado_2')
)

# Crear subplots para estandarización (Figura 1)
fig1, ax1 = plt.subplots(figsize=(12, 6))

# crea una lista de colores para cada firma

color_list = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'brown', 'pink',
              'olive', 'lime', 'teal', 'coral']

# Plot Estandarización
ax1.set_title('Standardized [(x-m)/s] NIR Spectral Signatures')
for key, value in cocoa_NIR_standardized.items():
    col = color_list.pop(0)
    ax1.plot(filtered_wavelength, value['data'].squeeze(), label=value['label'], color=col)
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Standardized Reflectance')
ax1.legend()

plt.tight_layout()
plt.show()

# Crear subplots para diferenciar abierto y cerrado (Figura 2)
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Plot con colores diferenciados para abierto y cerrado (Estandarizado)
ax2.set_title('Standardized [(x-m)/s] NIR Spectral Signatures (Open vs. Closed)')
for key, value in cocoa_NIR_standardized.items():
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
bad_standardized = np.mean([value['data'] for key, value in cocoa_NIR_standardized.items() if 'bad' in value['label']], axis=0)
neutral_standardized = np.mean([value['data'] for key, value in cocoa_NIR_standardized.items() if 'neutral' in value['label']], axis=0)
good_standardized = np.mean([value['data'] for key, value in cocoa_NIR_standardized.items() if 'good' in value['label']], axis=0)

# Crear una nueva figura para las medias
fig3, ax = plt.subplots(figsize=(12, 6))

# Graficar las medias estandarizadas
ax.set_title('Mean NIR Spectral Signatures (Standardized)')
ax.plot(filtered_wavelength, bad_standardized.squeeze(), label='Bad', color='red')
ax.plot(filtered_wavelength, neutral_standardized.squeeze(), label='Neutral', color='green')
ax.plot(filtered_wavelength, good_standardized.squeeze(), label='Good', color='blue')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Mean Standardized Reflectance')
ax.legend()

plt.tight_layout()
plt.show()
