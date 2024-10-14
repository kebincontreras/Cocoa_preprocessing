import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os
from sklearn.decomposition import PCA

# Ruta a la carpeta que contiene los archivos .mat
folder_path = 'C:\\Users\\USUARIO\\Documents\\Base_Datos_Cacao\\Junio\\prueba_granos'

# Función para cargar y procesar espectros
def cargar_espectro(filename):
    data = loadmat(filename)
    spectral_data = data[list(data.keys())[-1]]
    if spectral_data.shape[0] == 1 or spectral_data.shape[1] == 1:
        spectral_data = spectral_data.flatten()
    return spectral_data

# Inicializar el plot para PCA
plt.figure(figsize=(12, 8))

# Excluir archivos específicos
exclusion_list = ['N1.mat', 'B1.mat']

# Procesar cada archivo .mat en el directorio, excluyendo los especificados
lotes = [file for file in sorted(os.listdir(folder_path)) if file.endswith('.mat') and file not in exclusion_list]
colors = plt.cm.jet(np.linspace(0, 1, len(lotes)))

# Almacenar las firmas para el PCA
firmas_pca = []
labels_pca = []

for i, file in enumerate(lotes):
    filepath = os.path.join(folder_path, file)
    espectros_lote = cargar_espectro(filepath)
    
    if espectros_lote.ndim > 1:
        # Agregar a la lista para el PCA
        firmas_pca.append(espectros_lote)
        labels_pca.extend([file] * espectros_lote.shape[0])

# Concatenar todas las firmas
firmas_pca = np.vstack(firmas_pca)

# Aplicar PCA a todas las firmas
pca = PCA(n_components=2)
pca_result = pca.fit_transform(firmas_pca)

# Graficar las dos primeras componentes principales
for label in set(labels_pca):
    indices = [i for i, l in enumerate(labels_pca) if l == label]
    plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=label.split('.')[0])

plt.title('PCA ')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.show()
