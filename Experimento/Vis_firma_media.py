import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

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

# Aplicar el filtro a los datos, dividir por el blanco_teflon_n filtrado y normalizar
cocoa_VIS_normalized = dict(
    open50_1=dict(data=normalize_data(filter_data(cacao1_VIS['my_cacao_50_abierto_1'], mask) / blanco_teflon_n), label='bad_50_abierto_1'),
    closed50_1=dict(data=normalize_data(filter_data(cacao1_VIS['my_cacao_50_cerrado_1'], mask) / blanco_teflon_n), label='bad_50_cerrado_1'),
    open73_1=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_73_abierto_1'], mask) / blanco_teflon_n), label='neutral_73_abierto_1'),
    closed73_1=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_73_cerrado_1'], mask) / blanco_teflon_n), label='neutral_73_cerrado_1'),
    open73_2=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_73_abierto_2'], mask) / blanco_teflon_n), label='good_73_abierto_2'),
    closed73_2=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_73_cerrado_2'], mask) / blanco_teflon_n), label='good_73_cerrado_2'),
    open95_1=dict(data=normalize_data(filter_data(cacao1_VIS['my_cacao_95_abierto_1'], mask) / blanco_teflon_n), label='bad_95_abierto_1'),
    closed95_1=dict(data=normalize_data(filter_data(cacao1_VIS['my_cacao_95_cerrado_1'], mask) / blanco_teflon_n), label='bad_95_cerrado_1'),
    open96_1=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_96_abierto_1'], mask) / blanco_teflon_n), label='good_96_abierto_1'),
    closed96_1=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_96_cerrado_1'], mask) / blanco_teflon_n), label='good_96_cerrado_1')
)

# Calcular la media de las firmas para "bad", "neutral" y "good" separando abierto y cerrado
bad_open = np.mean([value['data'] for key, value in cocoa_VIS_normalized.items() if 'bad' in value['label'] and 'abierto' in value['label']], axis=0)
bad_closed = np.mean([value['data'] for key, value in cocoa_VIS_normalized.items() if 'bad' in value['label'] and 'cerrado' in value['label']], axis=0)

neutral_open = np.mean([value['data'] for key, value in cocoa_VIS_normalized.items() if 'neutral' in value['label'] and 'abierto' in value['label']], axis=0)
neutral_closed = np.mean([value['data'] for key, value in cocoa_VIS_normalized.items() if 'neutral' in value['label'] and 'cerrado' in value['label']], axis=0)

good_open = np.mean([value['data'] for key, value in cocoa_VIS_normalized.items() if 'good' in value['label'] and 'abierto' in value['label']], axis=0)
good_closed = np.mean([value['data'] for key, value in cocoa_VIS_normalized.items() if 'good' in value['label'] and 'cerrado' in value['label']], axis=0)

# Crear una nueva figura para las medias combinadas
fig, ax = plt.subplots(figsize=(12, 8))

# Graficar las medias para grano abierto y cerrado en el mismo gráfico
ax.set_title('Mean VIS Spectral Signatures (Normalized) - Open vs Closed Grain')
ax.plot(filtered_wavelength, bad_open.squeeze(), label='Bad (Open)', color='red', linestyle='--')
ax.plot(filtered_wavelength, bad_closed.squeeze(), label='Bad (Closed)', color='red')
ax.plot(filtered_wavelength, neutral_open.squeeze(), label='Neutral (Open)', color='green', linestyle='--')
ax.plot(filtered_wavelength, neutral_closed.squeeze(), label='Neutral (Closed)', color='green')
ax.plot(filtered_wavelength, good_open.squeeze(), label='Good (Open)', color='blue', linestyle='--')
ax.plot(filtered_wavelength, good_closed.squeeze(), label='Good (Closed)', color='blue')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Mean Normalized Reflectance')
ax.legend()

plt.tight_layout()
plt.show()

