import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Ruta de los archivos de materiales
base_dir_materiales = r"C:\Users\USUARIO\Downloads\ANALISISBANDASCONPICOS"

# Cargar los archivos de materiales
espuma = loadmat(os.path.join(base_dir_materiales, 'ESPUMAEX_O_NIR_2_1_18C_080824.mat'))['BLANCO']
latex = loadmat(os.path.join(base_dir_materiales, 'LATEXEX_O_NIR_2_1_18C_080824.mat'))['BLANCO']
silicona = loadmat(os.path.join(base_dir_materiales, 'SILICONAEX_O_NIR_2_1_18C_080824.mat'))['BLANCO']
aluminio = loadmat(os.path.join(base_dir_materiales, 'ALUMINIOEX_O_NIR_2_1_18C_080824.mat'))['BLANCO']
blanco_materiales = loadmat(os.path.join(base_dir_materiales, 'BEX_O_NIR_2_1_18C_080824.mat'))['BLANCO']
negro = loadmat(os.path.join(base_dir_materiales, 'NEX_O_NIR_2_1_18C_080824.mat'))['NEGRO']

# Ruta para cargar los datos de longitudes de onda y granos
base_dir_granos = r"C:\Users\USUARIO\Documents\GitHub\Dataset\Lab_hdsp_cocoa_experimento_jorge"
cacao_NIR = loadmat(os.path.join(base_dir_granos, 'experimento_cacao_3_fermantation_NIR.mat'))

# Obtener las longitudes de onda
wavelength = cacao_NIR['wavelengths'].squeeze()

# Filtrar las longitudes de onda en el rango de 1100 a 2100 nm (común para ambos)
mask = (wavelength >= 1100) & (wavelength <= 2100)
filtered_wavelength = wavelength[mask]

# Filtrar los datos de los materiales
espuma = espuma[:, mask]
latex = latex[:, mask]
silicona = silicona[:, mask]
aluminio = aluminio[:, mask]
blanco_materiales = blanco_materiales[:, mask]
negro = negro[:, mask]

# Realizar las operaciones: (firma - negro) / (blanco - negro)
def calculate_signature(material, blanco, negro):
    return (material - negro) / (blanco - negro)

# Calcular la firma media para cada material, considerando todas las firmas en cada archivo
espuma_media = np.mean(calculate_signature(espuma, blanco_materiales, negro), axis=0)
latex_media = np.mean(calculate_signature(latex, blanco_materiales, negro), axis=0)
silicona_media = np.mean(calculate_signature(silicona, blanco_materiales, negro), axis=0)
aluminio_media = np.mean(calculate_signature(aluminio, blanco_materiales, negro), axis=0)

# Normalizar todas las firmas de materiales utilizando MinMaxScaler
scaler = MinMaxScaler()
espuma_media = scaler.fit_transform(espuma_media.reshape(-1, 1)).squeeze()
latex_media = scaler.fit_transform(latex_media.reshape(-1, 1)).squeeze()
silicona_media = scaler.fit_transform(silicona_media.reshape(-1, 1)).squeeze()
aluminio_media = scaler.fit_transform(aluminio_media.reshape(-1, 1)).squeeze()

# Filtrar y normalizar los datos de granos
def filter_data(data, mask):
    return data[:, mask]

def normalize_data(data, blanco):
    return data / blanco

blanconir = filter_data(cacao_NIR['my_blanconir'], mask)

cocoa_NIR_normalized = dict(
    open73=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_73_abierto'], mask), blanconir), label='open'),
    closed73=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_73_cerrado'], mask), blanconir), label='closed'),
    open85=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_85_abierto'], mask), blanconir), label='open'),
    closed85=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_85_cerrado'], mask), blanconir), label='closed'),
    open94=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_94_abierto'], mask), blanconir), label='open'),
    closed94=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_94_cerrado'], mask), blanconir), label='closed'),
    open96=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_96_abierto'], mask), blanconir), label='open'),
    closed96=dict(data=normalize_data(filter_data(cacao_NIR['my_cacaocnir_96_cerrado'], mask), blanconir), label='closed')
)

# Combinar las firmas para "closed" y "open" en una sola firma media
closed_combined = np.mean([value['data'] for key, value in cocoa_NIR_normalized.items() if 'closed' in value['label']], axis=0)
open_combined = np.mean([value['data'] for key, value in cocoa_NIR_normalized.items() if 'open' in value['label']], axis=0)

# Normalizar las firmas combinadas utilizando MinMaxScaler
closed_combined = scaler.fit_transform(closed_combined.reshape(-1, 1)).squeeze()
open_combined = scaler.fit_transform(open_combined.reshape(-1, 1)).squeeze()

# Crear una nueva figura para las medias combinadas
fig, ax = plt.subplots(figsize=(12, 8))

# Graficar las firmas espectrales medias para los materiales
ax.plot(filtered_wavelength, espuma_media, label='Espuma')
ax.plot(filtered_wavelength, latex_media, label='Latex')
ax.plot(filtered_wavelength, silicona_media, label='Silicona')
ax.plot(filtered_wavelength, aluminio_media, label='Aluminio')

# Graficar las firmas combinadas para grano abierto y cerrado en color negro
ax.plot(filtered_wavelength, open_combined, label='Cacao Abierto', color='black', linestyle='--')
ax.plot(filtered_wavelength, closed_combined, label='Cacao Cerrado', color='black')

# Configuración del gráfico
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Mean Normalized Reflectance')
ax.set_title('Mean Spectral Signatures (Min-Max Normalized) - Materials and Combined Grains (Open vs Closed)')
ax.legend()

plt.tight_layout()
plt.show()
