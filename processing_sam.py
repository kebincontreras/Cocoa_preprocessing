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
muestra_dir = os.path.join(results_dir, "lote_5_kmeans_5_raad_01")
#muestra_dir = os.path.join(results_dir, "banda_5")
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


#LOTE_PARA_FILTRAR = loadmat(os.path.join(banda_dir, "BANDATRANSPORTADORAC090524.mat"))['BANDA']

############################### TRAINT  ###############################
LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L1F60R290324C070524TRAINFULL.mat"))['LCACAO']
#LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L2F66R310324C070524TRAINFULL.mat"))['LCACAO']
#LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L3F84R020424C090524TRAINFULL.mat"))['LCACAO']
#LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L4F92R130424C090524TRAINFULL.mat"))['LCACAO']
#LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L5F96RDDMMAAC090524TRAINFULL.mat"))['LCACAO']

############################### TEST  ###############################
#LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L1F60R290324C070524TESTFULL.mat"))['LCACAO']
#LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L2F66R310324C070524TESTFULL.mat"))['LCACAO']
#LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L3F84R020424C090524TESTFULL.mat"))['LCACAO']
#LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L4F92R130424C090524TESTFULL.mat"))['LCACAO']
#LOTE_PARA_FILTRAR = loadmat(os.path.join(lote_dir, "L5F96RDDMMAAC090524TESTFULL.mat"))['LCACAO']

LOTE_PARA_FILTRAR = LOTE_PARA_FILTRAR[1:]

# Filtrar longitudes de onda entre 400 y 1000 nm
mask = (wavelengths >= 400) & (wavelengths <= 1000)
wavelengths = wavelengths[mask]
BANDA = BANDA[:, mask]
LOTE_PARA_FILTRAR = LOTE_PARA_FILTRAR[:, mask]

# Aplicación de K-means para el clustering
kmeans = KMeans(n_clusters=5, random_state=0).fit(BANDA)
cluster_centers = kmeans.cluster_centers_

# Calculando ángulos y generando la máscara SAM
angulo = 0.1
min_angles = []
sam_mask = []
for Fa in LOTE_PARA_FILTRAR:
    Phis = [np.arccos(np.dot(Fa, nm) / (np.linalg.norm(Fa) * np.linalg.norm(nm))) for nm in cluster_centers]
    min_angle = np.min(Phis)
    min_angles.append(min_angle)
    sam_mask.append(int(min_angle > angulo))

sam_mask = np.array(sam_mask)
sam_mask_matrix = np.tile(sam_mask[:, None], (1, LOTE_PARA_FILTRAR.shape[1]))
min_angles = np.array(min_angles)

firmas_seleccionadas = LOTE_PARA_FILTRAR[sam_mask == 1]
firmas_delete = LOTE_PARA_FILTRAR[sam_mask == 0]

# Guardar firmas seleccionadas y eliminadas en archivos .mat
savemat(os.path.join(processed_dir, 'firmas_seleccionadas.mat'), {'firmas_seleccionadas': firmas_seleccionadas})
savemat(os.path.join(delete_dir, 'firmas_delete.mat'), {'firmas_delete': firmas_delete})


# Generar subplots para mostrar los datos y ángulos
plt.figure(figsize=(12, 10))
plt.subplot(2, 3, 1)
plt.imshow(BANDA[:2000])
plt.title('Banda transportadora')
plt.subplot(2, 3, 2)
plt.imshow(LOTE_PARA_FILTRAR[:2000])
plt.title('Banda transportadora + Cacao')
plt.subplot(2, 3, 3)
plt.imshow(sam_mask_matrix[:2000])
plt.title('Máscara')



# Graficar firmas medias de cada cluster
plt.subplot(2, 3, 4)
for center in cluster_centers:
    plt.plot(wavelengths, center, label=f'Cluster {np.where(cluster_centers == center)[0][0]}')
#plt.legend()
plt.title('rep banda')
#plt.xlabel('Longitud de Onda (nm)')
#plt.ylabel('Intensidad')

# Ordenar ángulos mínimos y graficar con líneas verticales en subplot (3,3,8)
sorted_indices = np.argsort(min_angles)
sorted_min_angles = min_angles[sorted_indices]

plt.subplot(2, 3, 5)
#plt.vlines(range(len(sorted_min_angles)), 0, sorted_min_angles, colors='b', linestyles='solid', linewidth=0.5)
plt.plot(sorted_min_angles)
plt.title('Sorted SAM')


# Filtrar firmas seleccionadas con máximo en el rango [0, 2500] y graficar en subplot (3,3,9)
max_intensities = np.max(firmas_seleccionadas, axis=1)
range_intensity_mask = (max_intensities >= 0) & (max_intensities <= 3000)
range_intensity_firmas = firmas_seleccionadas[range_intensity_mask]

plt.subplot(2, 3, 6)
for firma in range_intensity_firmas:
    plt.plot(wavelengths, firma)
plt.title('Sel 3000')
plt.xlabel('Índice Espectral')
plt.ylabel('Intensidad Espectral')


plt.tight_layout()
plt.savefig(os.path.join(muestra_dir, '2Preprocessing.png'))

# Guardar y cerrar el archivo de resumen
summary_filename = os.path.join(muestra_dir, 'summary_sam.txt')
with open(summary_filename, 'w') as file:
    file.write(f"Nombre de la muestra: {os.path.basename(muestra_dir)}\n")
    file.write(f"Total de firmas en LOTE_PARA_FILTRAR: {len(LOTE_PARA_FILTRAR)}\n")
    file.write(f"Total de firmas eliminadas: {len(firmas_delete)}\n")
    file.write(f"Total de firmas seleccionadas: {len(firmas_seleccionadas)}\n")
    file.write(f"Ángulo utilizado para filtrado: {angulo} radianes\n")
print("Proceso completado.")

