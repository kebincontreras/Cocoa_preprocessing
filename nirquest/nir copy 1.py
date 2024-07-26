import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os
from sklearn.cluster import KMeans
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

# Procesar cada archivo .mat en el directorio, excluyendo b1 y N1
lotes = [file for file in sorted(os.listdir(folder_path)) if file not in ['B1.mat', 'N1.mat'] and file in ['L1.mat', 'L2.mat', 'L5.mat', 'L52.mat', 'L5_3.mat']]
colors = plt.cm.jet(np.linspace(0, 1, len(lotes)))

# Almacenar las firmas para el PCA
firmas_pca = []
labels_pca = []

for i, file in enumerate(lotes):
    filepath = os.path.join(folder_path, file)
    espectros_lote = cargar_espectro(filepath)
    
    if espectros_lote.ndim > 1:
        # Aplicar k-means a las firmas del lote
        kmeans = KMeans(n_clusters=3, random_state=0).fit(espectros_lote)
        labels = kmeans.labels_
        
        # Determinar el clúster más grande
        unique, counts = np.unique(labels, return_counts=True)
        cluster_index = unique[np.argmax(counts)]
        
        # Filtrar las firmas que pertenecen al clúster más grande
        firmas_cluster = espectros_lote[labels == cluster_index]
        
        # Agregar a la lista para el PCA
        firmas_pca.append(firmas_cluster)
        labels_pca.extend([file] * len(firmas_cluster))

# Aplicar PCA a todas las firmas del clúster más grande
pca = PCA(n_components=2)
firmas_pca = np.vstack(firmas_pca)
pca_result = pca.fit_transform(firmas_pca)

# Graficar las dos primeras componentes principales
for label in set(labels_pca):
    indices = [i for i, l in enumerate(labels_pca) if l == label]
    plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=label.split('.')[0])

plt.title('PCA de Firmas Espectrales del Clúster Más Grande')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.show()
