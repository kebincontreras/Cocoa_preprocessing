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
cocoa_NIR_normalized = {
    'N_73_a': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_73_abierto'], mask) / blanconir), label='N_73_a'),
    'N_73_c': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_73_cerrado'], mask) / blanconir), label='N_73_c'),
    'b_73_a': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_73_abierto_2'], mask) / blanconir), label='b_73_a'),
    'b_73_c': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_73_cerrado_2'], mask) / blanconir), label='b_73_c'),
    'b_85_a': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_85_abierto'], mask) / blanconir), label='b_85_a'),
    'b_85_c': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_85_cerrado'], mask) / blanconir), label='b_85_c'),
    'g_85_a': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_85_abierto_2'], mask) / blanconir), label='g_85_a'),
    'g_85_c': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_85_cerrado_2'], mask) / blanconir), label='g_85_c'),
    'g_94_a': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_94_abierto'], mask) / blanconir), label='g_94_a'),
    'g_94_c': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_94_cerrado'], mask) / blanconir), label='g_94_c'),
    'b_94_a': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_94_abierto_2'], mask) / blanconir), label='b_94_a'),
    'b_94_c': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_94_cerrado_2'], mask) / blanconir), label='b_94_c'),
    'g_96_a': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_96_abierto'], mask) / blanconir), label='g_96_a'),
    'g_96_c': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_96_cerrado'], mask) / blanconir), label='g_96_c'),
    'g_96_a2': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_96_abierto_2'], mask) / blanconir), label='g_96_a2'),
    'g_96_c2': dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_96_cerrado_2'], mask) / blanconir), label='g_96_c2')
}

cocoa_NIR_standardized = {
    'N_73_a': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_73_abierto'], mask) / blanconir), label='N_73_a'),
    'N_73_c': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_73_cerrado'], mask) / blanconir), label='N_73_c'),
    'b_73_a': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_73_abierto_2'], mask) / blanconir), label='b_73_a'),
    'b_73_c': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_73_cerrado_2'], mask) / blanconir), label='b_73_c'),
    'b_85_a': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_85_abierto'], mask) / blanconir), label='b_85_a'),
    'b_85_c': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_85_cerrado'], mask) / blanconir), label='b_85_c'),
    'g_85_a': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_85_abierto_2'], mask) / blanconir), label='g_85_a'),
    'g_85_c': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_85_cerrado_2'], mask) / blanconir), label='g_85_c'),
    'g_94_a': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_94_abierto'], mask) / blanconir), label='g_94_a'),
    'g_94_c': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_94_cerrado'], mask) / blanconir), label='g_94_c'),
    'b_94_a': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_94_abierto_2'], mask) / blanconir), label='b_94_a'),
    'b_94_c': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_94_cerrado_2'], mask) / blanconir), label='b_94_c'),
    'g_96_a': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_96_abierto'], mask) / blanconir), label='g_96_a'),
    'g_96_c': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_96_cerrado'], mask) / blanconir), label='g_96_c'),
    'g_96_a2': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_96_abierto_2'], mask) / blanconir), label='g_96_a2'),
    'g_96_c2': dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_96_cerrado_2'], mask) / blanconir), label='g_96_c2')
}

# Calcular el SAM para cada firma espectral
def calculate_sam(data):
    reference_vector = np.ones(data.shape[1])
    sam_values = []
    for spectrum in data:
        dot_product = np.dot(spectrum, reference_vector)
        norm_spectrum = np.linalg.norm(spectrum)
        norm_reference = np.linalg.norm(reference_vector)
        sam = np.arccos(dot_product / (norm_spectrum * norm_reference))
        sam_values.append(sam)
    return np.array(sam_values)

# Añadir valores de SAM a cada firma espectral
for key in cocoa_NIR_normalized:
    cocoa_NIR_normalized[key]['sam'] = calculate_sam(cocoa_NIR_normalized[key]['data'])
    cocoa_NIR_standardized[key]['sam'] = calculate_sam(cocoa_NIR_standardized[key]['data'])

# Crear subplots para SAM (Figura 3)
fig4, ax4 = plt.subplots(2, 1, figsize=(12, 12))

symbols = {'a': '*', 'c': 'o'}
colors = {'b': 'red', 'N': 'green', 'g': 'blue'}

# Plot SAM Normalizado
ax4[0].set_title('SAM [Normalized] NIR Spectral Signatures')
for key, value in cocoa_NIR_normalized.items():
    label_parts = value['label'].split('_')
    if len(label_parts) == 3:
        category = label_parts[0]
        openness = label_parts[2]
        ax4[0].scatter([value['label']], value['sam'], marker=symbols.get(openness, 'o'), color=colors.get(category, 'black'))

ax4[0].set_xlabel('Sample')
ax4[0].set_ylabel('SAM Value')

# Plot SAM Estandarizado
ax4[1].set_title('SAM [Standardized] NIR Spectral Signatures')
for key, value in cocoa_NIR_standardized.items():
    label_parts = value['label'].split('_')
    if len(label_parts) == 3:
        category = label_parts[0]
        openness = label_parts[2]
        ax4[1].scatter([value['label']], value['sam'], marker=symbols.get(openness, 'o'), color=colors.get(category, 'black'))

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
