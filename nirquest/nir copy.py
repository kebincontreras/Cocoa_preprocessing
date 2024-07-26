import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os
from sklearn.cluster import KMeans

# Ruta a la carpeta que contiene los archivos .mat
folder_path = 'C:\\Users\\USUARIO\\Documents\\Base_Datos_Cacao\\Junio\\prueba_granos'

# Función para cargar y procesar espectros
def cargar_espectro(filename):
    data = loadmat(filename)
    # Asume que los datos son la última variable almacenada en el archivo
    spectral_data = data[list(data.keys())[-1]]
    # Asegúrate de que el array está correctamente dimensionado
    if spectral_data.shape[0] == 1 or spectral_data.shape[1] == 1:
        spectral_data = spectral_data.flatten()
    return spectral_data

# Inicializar el plot
plt.figure(figsize=(12, 8))

# Procesar cada archivo .mat en el directorio, excluyendo b1 y N1
lotes = [file for file in sorted(os.listdir(folder_path)) if file not in ['b1.mat', 'N1.mat']]
colors = plt.cm.jet(np.linspace(0, 1, len(lotes)))

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
        
        # Graficar las firmas del clúster más grande
        for firma in firmas_cluster:
            plt.plot(firma, color=colors[i], label=f'Lote {file.split(".")[0]}' if i == 0 else "")

plt.title('Firmas Espectrales del Clúster Más Grande de Cada Lote')
plt.xlabel('Número de Punto Espectral')
plt.ylabel('Intensidad')
plt.legend()
plt.show()
