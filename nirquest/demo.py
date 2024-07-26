import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Ruta a la carpeta que contiene los archivos .mat
folder_path = '.'


# Función para cargar y procesar espectros
def cargar_espectro(filename):
    data = loadmat(filename)
    spectral_data = data[list(data.keys())[-1]]
    return spectral_data


# Cargar espectros de blanco y negro para calcular la firma media
espectro_blanco = cargar_espectro(os.path.join(folder_path, 'B1.mat'))
espectro_negro = cargar_espectro(os.path.join(folder_path, 'N1.mat'))

# Calcular las firmas medias para blanco y negro
firma_media_blanco = np.mean(espectro_blanco, axis=0)
firma_media_negro = np.mean(espectro_negro, axis=0)

# Listar todos los archivos que serán procesados
lotes = ['L1.mat', 'L2.mat', 'L5.mat', 'L5_2.mat', 'L5_3.mat']

# Inicializar el plot para PCA
fig, axes = plt.subplots(2, 1, figsize=(12, 16))  # Dos subplots

# Almacenar las firmas para el PCA general y las firmas del clúster más grande
firmas_pca_general = []
labels_pca_general = []
firmas_pca_cluster = []
labels_pca_cluster = []

# Procesar cada archivo especificado
for file in lotes:
    filepath = os.path.join(folder_path, file)
    espectros_lote = cargar_espectro(filepath)
    reflectancias = espectros_lote / (firma_media_blanco - firma_media_negro)

    # Agregar a la lista para el PCA general
    firmas_pca_general.append(reflectancias)
    labels_pca_general.extend([file] * reflectancias.shape[0])

    # Aplicar K-means y usar solo para el segundo subplot
    kmeans = KMeans(n_clusters=3, random_state=0).fit(reflectancias)
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    cluster_index = unique[np.argmax(counts)]
    firmas_cluster = reflectancias[labels == cluster_index]

    # Agregar a la lista para el PCA del clúster más grande
    firmas_pca_cluster.append(firmas_cluster)
    labels_pca_cluster.extend([file] * len(firmas_cluster))

# Aplicar PCA y graficar para ambos subplots
pca_general = PCA(n_components=2)
pca_cluster = PCA(n_components=2)

# PCA para todas las firmas
firmas_pca_general = np.vstack(firmas_pca_general)
pca_result_general = pca_general.fit_transform(firmas_pca_general)

# PCA para firmas del clúster más grande
firmas_pca_cluster = np.vstack(firmas_pca_cluster)
pca_result_cluster = pca_cluster.fit_transform(firmas_pca_cluster)

# Graficar las dos primeras componentes principales para cada subplot
for (ax, pca_result, labels_pca, title) in zip(
        axes,
        [pca_result_general, pca_result_cluster],
        [labels_pca_general, labels_pca_cluster],
        ['PCA de Firmas Espectrales', 'PCA de Firmas Espectrales del Clúster Más Grande']):

    for label in set(labels_pca):
        indices = [j for j, l in enumerate(labels_pca) if l == label]
        ax.scatter(pca_result[indices, 0], pca_result[indices, 1], label=label.split('.')[0], alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.legend()

plt.tight_layout()
plt.show()