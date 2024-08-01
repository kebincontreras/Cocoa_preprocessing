import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat

# Directorio donde se encuentran los archivos
base_dir = r"C:\Users\USUARIO\Documents\GitHub\Dataset\Lab_hdsp_cocoa_experimento_jorge"

# Cargar los archivos
cacao_NIR = loadmat(os.path.join(base_dir, 'experimento_cacao_3_fermantation_NIR.mat'))

# Obtener las longitudes de onda (wavelet)
wavelengths = cacao_NIR['wavelengths'].squeeze()

# Obtener los datos del blanco
blanco_nir = cacao_NIR['my_blanconir'].squeeze()

# Normalizar los datos del blanco
blanco_nir_normalized = blanco_nir / np.max(blanco_nir)

# Graficar el blanco normalizado con las longitudes de onda en el eje x
plt.figure()
plt.plot(wavelengths, blanco_nir_normalized, label='Blanco NIR Normalizado')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Reflectancia Normalizada')
plt.title('Reflectancia Normalizada del Blanco en el rango NIR')
plt.legend()
plt.show()
