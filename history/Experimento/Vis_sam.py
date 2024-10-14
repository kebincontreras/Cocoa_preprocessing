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
    open50_1=dict(data=normalize_data(filter_data(cacao1_VIS['my_cacao_50_abierto_1'], mask) / blanco_teflon_n), label='bad_50_abierto'),
    closed50_1=dict(data=normalize_data(filter_data(cacao1_VIS['my_cacao_50_cerrado_1'], mask) / blanco_teflon_n), label='bad_50_cerrado'),
    open73_1=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_73_abierto_1'], mask) / blanco_teflon_n), label='neutral_73_abierto'),
    closed73_1=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_73_cerrado_1'], mask) / blanco_teflon_n), label='neutral_73_cerrado'),
    open73_2=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_73_abierto_2'], mask) / blanco_teflon_n), label='good_73_abierto'),
    closed73_2=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_73_cerrado_2'], mask) / blanco_teflon_n), label='good_73_cerrado'),
    open95_1=dict(data=normalize_data(filter_data(cacao1_VIS['my_cacao_95_abierto_1'], mask) / blanco_teflon_n), label='bad_95_abierto'),
    closed95_1=dict(data=normalize_data(filter_data(cacao1_VIS['my_cacao_95_cerrado_1'], mask) / blanco_teflon_n), label='bad_95_cerrado'),
    open96_1=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_96_abierto_1'], mask) / blanco_teflon_n), label='good_96_abierto'),
    closed96_1=dict(data=normalize_data(filter_data(cacao2_VIS['my_cacao_96_cerrado_1'], mask) / blanco_teflon_n), label='good_96_cerrado')
)

cocoa_VIS_standardized = dict(
    open50_1=dict(data=standardize_data(filter_data(cacao1_VIS['my_cacao_50_abierto_1'], mask) / blanco_teflon_n), label='bad_50_abierto'),
    closed50_1=dict(data=standardize_data(filter_data(cacao1_VIS['my_cacao_50_cerrado_1'], mask) / blanco_teflon_n), label='bad_50_cerrado'),
    open73_1=dict(data=standardize_data(filter_data(cacao2_VIS['my_cacao_73_abierto_1'], mask) / blanco_teflon_n), label='neutral_73_abierto'),
    closed73_1=dict(data=standardize_data(filter_data(cacao2_VIS['my_cacao_73_cerrado_1'], mask) / blanco_teflon_n), label='neutral_73_cerrado'),
    open73_2=dict(data=standardize_data(filter_data(cacao2_VIS['my_cacao_73_abierto_2'], mask) / blanco_teflon_n), label='good_73_abierto'),
    closed73_2=dict(data=standardize_data(filter_data(cacao2_VIS['my_cacao_73_cerrado_2'], mask) / blanco_teflon_n), label='good_73_cerrado'),
    open95_1=dict(data=standardize_data(filter_data(cacao1_VIS['my_cacao_95_abierto_1'], mask) / blanco_teflon_n), label='bad_95_abierto'),
    closed95_1=dict(data=standardize_data(filter_data(cacao1_VIS['my_cacao_95_cerrado_1'], mask) / blanco_teflon_n), label='bad_95_cerrado'),
    open96_1=dict(data=standardize_data(filter_data(cacao2_VIS['my_cacao_96_abierto_1'], mask) / blanco_teflon_n), label='good_96_abierto'),
    closed96_1=dict(data=standardize_data(filter_data(cacao2_VIS['my_cacao_96_cerrado_1'], mask) / blanco_teflon_n), label='good_96_cerrado')
)

# Función para calcular el ángulo espectral (SAM)
def spectral_angle_mapper(target, reference):
    dot_product = np.sum(target * reference, axis=1)
    norm_product = np.linalg.norm(target, axis=1) * np.linalg.norm(reference)
    return np.arccos(dot_product / norm_product)

# Vector unitario para comparación
unit_vector = np.ones((1, filtered_wavelength.size))

# Calcular el SAM para cada firma espectral normalizada y estandarizada
for key, value in cocoa_VIS_normalized.items():
    value['sam'] = spectral_angle_mapper(value['data'], unit_vector)

for key, value in cocoa_VIS_standardized.items():
    value['sam'] = spectral_angle_mapper(value['data'], unit_vector)

# Crear una nueva figura para graficar los valores de SAM
fig4, ax4 = plt.subplots(2, 1, figsize=(12, 12))

# Definir colores y símbolos
colors = {'bad': 'red', 'neutral': 'orange', 'good': 'green'}
symbols = {'abierto': '*', 'cerrado': 'o'}

# Plot SAM Normalizado
ax4[0].set_title('SAM [Normalized] VIS Spectral Signatures')
for key, value in cocoa_VIS_normalized.items():
    label_parts = value['label'].split('_')
    category = label_parts[0]
    openness = label_parts[2]
    ax4[0].plot([value['label']], value['sam'], symbols[openness], color=colors[category])

ax4[0].set_xlabel('Sample')
ax4[0].set_ylabel('SAM Value')

# Plot SAM Estandarizado
ax4[1].set_title('SAM [Standardized] VIS Spectral Signatures')
for key, value in cocoa_VIS_standardized.items():
    label_parts = value['label'].split('_')
    category = label_parts[0]
    openness = label_parts[2]
    ax4[1].plot([value['label']], value['sam'], symbols[openness], color=colors[category])

ax4[1].set_xlabel('Sample')
ax4[1].set_ylabel('SAM Value')

# Añadir leyenda
handles = [
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='k', markersize=10, label='Abierto'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=10, label='Cerrado')
]

ax4[0].legend(handles=handles, loc='upper right')
ax4[1].legend(handles=handles, loc='upper right')

plt.tight_layout()
plt.show()
