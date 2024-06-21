# Importar las librerías necesarias
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from sklearn.cluster import KMeans

# Definición de directorios base y subdirectorios para organizar datos y resultados
base_dir = "C:\\Users\\USUARIO\\Documents\\GitHub\\Preprocessing"
banda_dir = os.path.join(base_dir, "Anexos")
lote_dir = os.path.join(base_dir, "Optical_lab_spectral")
results_dir = os.path.join(base_dir, "Results")
muestra_dir = os.path.join(results_dir, "lote_1")
processed_dir = os.path.join(muestra_dir, "Processed")
delete_dir = os.path.join(muestra_dir, "Delete")

# Asegurar la creación de los directorios si no existen
os.makedirs(results_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(delete_dir, exist_ok=True)

# Cargar datos desde archivos .mat
BANDA = loadmat(os.path.join(banda_dir, "BANDATRANSPORTADORAC090524.mat"))['BANDA']
wavelengths = BANDA[0, :]
BANDA = BANDA[1:]

LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L1F60R290324C070524TRAINFULL.mat"))['LCACAO']
#LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L2F66R310324C070524TRAINFULL.mat"))['LCACAO']
#LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L3F84R020424C090524TRAINFULL.mat"))['LCACAO']
#LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L4F92R130424C090524TRAINFULL.mat"))['LCACAO']
#LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L5F96RDDMMAAC090524TRAINFULL.mat"))['LCACAO']

LOTE_PARA_FILTRAR = LOTE_PARA_FILTRAR[1:]

# Filtrar longitudes de onda entre 400 y 1000 nm
mask = (wavelengths >= 400) & (wavelengths <= 1000)
wavelengths = wavelengths[mask]
BANDA = BANDA[:, mask]
LOTE_PARA_FILTRAR = LOTE_PARA_FILTRAR[:, mask]

# Aplicación de K-means para el clustering
kmeans = KMeans(n_clusters=5, random_state=0).fit(BANDA)
cluster_centers = kmeans.cluster_centers_

# Generación de una máscara de selección para filtrar firmas espectrales (SAM)
sam_mask = []
angulo = 0.275
for Fa in LOTE_PARA_FILTRAR:
    Phis = [np.arccos(np.dot(Fa, nm) / (np.linalg.norm(Fa) * np.linalg.norm(nm))) for nm in cluster_centers]
    sam_mask.append(int(any(phi > angulo for phi in Phis)))

sam_mask = np.array(sam_mask)
sam_mask_matrix = np.tile(sam_mask[:, None], (1, LOTE_PARA_FILTRAR.shape[1]))

firmas_seleccionadas = LOTE_PARA_FILTRAR[sam_mask == 1]
firmas_delete = LOTE_PARA_FILTRAR[sam_mask == 0]

# Visualización inicial de las firmas del lote
plt.figure(figsize=(10, 6))
for firma in LOTE_PARA_FILTRAR:
    plt.plot(wavelengths, firma)
plt.title('Firmas del Lote para Filtrar')
plt.xlabel('Índice Espectral')
plt.ylabel('Intensidad Espectral')
plt.show()

# Solicitar umbral para el filtrado basado en máximos
umbral = float(input("Introduce el umbral para el filtrado basado en el máximo de las firmas: "))

# Aplicar el filtrado basado en umbral
threshold_mask = np.max(LOTE_PARA_FILTRAR, axis=1) >= umbral
threshold_mask_matrix = np.tile(threshold_mask[:, None], (1, LOTE_PARA_FILTRAR.shape[1]))

# Guardar los datos filtrados
firmas_seleccionadas_umbral = LOTE_PARA_FILTRAR[threshold_mask]
firmas_delete_umbral = LOTE_PARA_FILTRAR[~threshold_mask]
savemat(os.path.join(processed_dir, "lote_umbral.mat"), {'data': firmas_seleccionadas_umbral})
savemat(os.path.join(delete_dir, "lote_delete_umbral.mat"), {'data': firmas_delete_umbral})

# Generar subplots para mostrar las máscaras y los datos
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
#plt.imshow(BANDA[:4000], aspect='auto')
#plt.imshow(BANDA[:4000])
plt.imshow(BANDA[:1000])
plt.title('BANDA')
plt.subplot(2, 2, 2)
#plt.imshow(LOTE_PARA_FILTRAR[:4000], aspect='auto')
#plt.imshow(LOTE_PARA_FILTRAR[:4000])
plt.imshow(LOTE_PARA_FILTRAR[:1000])
plt.title('LOTE PARA FILTRAR')
plt.subplot(2, 2, 3)
#plt.imshow(sam_mask_matrix[:4000], aspect='auto')
#plt.imshow(sam_mask_matrix[:4000])
plt.imshow(sam_mask_matrix[:1000])
plt.title('Máscara SAM')
plt.subplot(2, 2, 4)
#plt.imshow(threshold_mask_matrix[:4000], aspect='auto')
#plt.imshow(threshold_mask_matrix[:4000])
plt.imshow(threshold_mask_matrix[:1000])
plt.title('Máscara Umbral')
plt.tight_layout()
plt.savefig(os.path.join(muestra_dir, '2Preprocessing.png'))
#plt.show()

# Nombre del archivo de texto
summary_filename = os.path.join(muestra_dir, 'summary_sam.txt')

# Abrir archivo para escritura
with open(summary_filename, 'w') as file:
    file.write(f"Nombre de la muestra: {os.path.basename(muestra_dir)}\n")
    file.write(f"Total de firmas en LOTE_PARA_FILTRAR: {len(LOTE_PARA_FILTRAR)}\n")
    file.write(f"Total de firmas eliminadas: {len(firmas_delete)}\n")
    file.write(f"Total de firmas seleccionadas: {len(firmas_seleccionadas)}\n")
    file.write(f"Ángulo utilizado para filtrado: {angulo}radiantes\n")
    file.write("---------------------------------------------\n")
    file.write(f"Total de firmas seleccionadas por umbral: {len(firmas_seleccionadas_umbral)}\n")
    file.write(f"Total de firmas eliminadas por umbral: {len(firmas_delete_umbral)}\n")
    file.write(f"Umbral utilizado: {umbral}\n")

print("Proceso completado.")