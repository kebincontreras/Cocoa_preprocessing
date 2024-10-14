import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Directorios donde se encuentran los archivos
base_dir_wavelength = r"C:\Users\USUARIO\Documents\GitHub\Dataset\Lab_hdsp_cocoa_experimento_jorge"
base_dir = r"C:\Users\USUARIO\Documents\GitHub\Dataset\Experimento_2"

# Cargar los archivos de longitudes de onda
cacao_NIR = loadmat(os.path.join(base_dir_wavelength, 'experimento_cacao_3_fermantation_NIR.mat'))

# Obtener las longitudes de onda
wavelength = cacao_NIR['wavelengths'].squeeze()

# Filtrar las longitudes de onda en el rango de 1100 a 2100 nm
mask = (wavelength >= 1100) & (wavelength <= 2100)

# Aplicar la máscara para obtener las longitudes de onda filtradas
filtered_wavelength = wavelength[mask]

# Cargar los archivos de materiales
gabierto = loadmat(os.path.join(base_dir, 'GABIERTO_O_NIR_2_3_18C_090824.mat'))['BLANCO']
gcascara = loadmat(os.path.join(base_dir, 'GCASCARA_O_NIR_2_3_18C_090824.mat'))['BLANCO']
gexpuesto = loadmat(os.path.join(base_dir, 'GEXPUESTO_O_NIR_2_3_18C_090824.mat'))['BLANCO']
blanco = loadmat(os.path.join(base_dir, 'BEX_O_NIR_2_3_18C_090824.mat'))['BLANCO']
negro = loadmat(os.path.join(base_dir, 'NEX_O_NIR_2_3_18C_090824.mat'))['NEGRO']

# Filtrar los datos basados en las longitudes de onda seleccionadas
def filter_data(data, mask):
    return data[:, mask]

# Filtrar los datos utilizando la máscara
gabierto = filter_data(gabierto, mask)
gcascara = filter_data(gcascara, mask)
gexpuesto = filter_data(gexpuesto, mask)
blanco = filter_data(blanco, mask)
negro = filter_data(negro, mask)

# Calcular la firma media para blanco y negro
blanco_mean = np.mean(blanco, axis=0)
negro_mean = np.mean(negro, axis=0)

# Normalizar los datos
def normalize_data(data, blanco_mean, negro_mean):
    scaler = MinMaxScaler()
    #normalized_data = (data - negro_mean) / (blanco_mean - negro_mean)
    normalized_data = (data ) / (blanco_mean )
    return scaler.fit_transform(normalized_data.T).T

# Calcular la firma normalizada para cada material
gabierto_normalized = normalize_data(gabierto, blanco_mean, negro_mean)
gcascara_normalized = normalize_data(gcascara, blanco_mean, negro_mean)
gexpuesto_normalized = normalize_data(gexpuesto, blanco_mean, negro_mean)

# Seleccionar una firma específica de los 3 disponibles
gabierto_normalized = gabierto_normalized[0]  # Selecciona la primera firma
gcascara_normalized = gcascara_normalized[0]  # Selecciona la primera firma
gexpuesto_normalized = gexpuesto_normalized[0]  # Selecciona la primera firma

# Crear una nueva figura para las firmas normalizadas
fig, ax = plt.subplots(figsize=(12, 8))

# Graficar las firmas espectrales normalizadas
ax.plot(filtered_wavelength, gabierto_normalized, label='Abierto', color='red')
ax.plot(filtered_wavelength, gcascara_normalized, label='Cáscara', color='green')
ax.plot(filtered_wavelength, gexpuesto_normalized, label='Expuesto', color='blue')

# Configuración del gráfico
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Normalized Reflectance')
ax.set_title('Spectral Signatures (Min-Max Normalized) - Abierto, Cáscara, Expuesto')
ax.legend()

plt.tight_layout()
plt.show()
