import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

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

# Aplicar el filtro a los datos, dividir por el blanconir filtrado, y luego normalizar
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

# Calcular la media de las firmas para "bad", "neutral" y "good" separando abierto y cerrado
bad_open = np.mean([value['data'] for key, value in cocoa_NIR_normalized.items() if 'bad' in value['label'] and 'abierto' in value['label']], axis=0)
bad_closed = np.mean([value['data'] for key, value in cocoa_NIR_normalized.items() if 'bad' in value['label'] and 'cerrado' in value['label']], axis=0)

neutral_open = np.mean([value['data'] for key, value in cocoa_NIR_normalized.items() if 'neutral' in value['label'] and 'abierto' in value['label']], axis=0)
neutral_closed = np.mean([value['data'] for key, value in cocoa_NIR_normalized.items() if 'neutral' in value['label'] and 'cerrado' in value['label']], axis=0)

good_open = np.mean([value['data'] for key, value in cocoa_NIR_normalized.items() if 'good' in value['label'] and 'abierto' in value['label']], axis=0)
good_closed = np.mean([value['data'] for key, value in cocoa_NIR_normalized.items() if 'good' in value['label'] and 'cerrado' in value['label']], axis=0)

# Crear una nueva figura para las medias combinadas
fig, ax = plt.subplots(figsize=(12, 8))

# Graficar las medias para grano abierto y cerrado en el mismo gráfico
ax.set_title('Mean NIR Spectral Signatures (Normalized) - Open vs Closed Grain')
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
