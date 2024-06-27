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
muestra_dir = os.path.join(results_dir, "lote_1_final")
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
angulo = 0.275
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

# Generar subplots para mostrar los datos y ángulos
plt.figure(figsize=(12, 10))
plt.subplot(3, 3, 1)
plt.imshow(BANDA[:1000])
plt.title('BANDA')
plt.subplot(3, 3, 2)
plt.imshow(LOTE_PARA_FILTRAR[:1000])
plt.title('LOTE PARA FILTRAR')
plt.subplot(3, 3, 3)
plt.imshow(sam_mask_matrix[:1000])
plt.title('Máscara SAM')

plt.subplot(3, 3, 4)
for firma in LOTE_PARA_FILTRAR:
    plt.plot(wavelengths, firma)
plt.title('Firmas del Lote para Filtrar')
plt.xlabel('Índice Espectral')
plt.ylabel('Intensidad Espectral')

plt.subplot(3, 3, 5)
for firma in firmas_seleccionadas:
    plt.plot(wavelengths, firma)
plt.title('Firmas Seleccionadas')

plt.subplot(3, 3, 6)
for firma in firmas_delete:
    plt.plot(wavelengths, firma)
plt.title('Firmas eliminadas')



"""
# Graficar los ángulos mínimos
plt.subplot(3, 3, 7)
#plt.plot(min_angles, marker='o')
plt.scatter(range(len(min_angles)), min_angles, s=1) 
plt.title('Ángulo Mínimo por Firma')
plt.xlabel('Índice de la Firma')
plt.ylabel('Ángulo Mínimo (radianes)')




# Ordenar ángulos mínimos y graficar
sorted_indices = np.argsort(min_angles)
sorted_min_angles = min_angles[sorted_indices]

plt.subplot(3, 3, 8)
plt.plot(sorted_min_angles, marker='o', markersize=2)
#plt.scatter(range(len(sorted_min_angles)), min_angles, s=1) 
plt.title('Ángulos Mínimos Ordenados')
plt.xlabel('Índice Ordenado')
plt.ylabel('Ángulo (radianes)')

# Filtrar firmas seleccionadas con máximo >= 2500 y graficar algunas de ellas
max_intensities = np.max(firmas_seleccionadas, axis=1)
high_intensity_mask = max_intensities >= 2500
high_intensity_firmas = firmas_seleccionadas[high_intensity_mask]

plt.subplot(3, 3, 9)
for firma in high_intensity_firmas[:10]:  # Graficar solo las primeras 10 para evitar sobrecarga
    plt.plot(wavelengths, firma)
plt.title('Firmas Seleccionadas con Intensidad Alta')
plt.xlabel('Índice Espectral')
plt.ylabel('Intensidad Espectral')
"""

# Graficar ángulos mínimos con líneas verticales en subplot (3,3,7)
plt.subplot(3, 3, 7)
plt.vlines(range(len(min_angles)), 0, min_angles, colors='b', linestyles='solid', linewidth=0.5)
#plt.title('Ángulo Mínimo por Firma')
plt.xlabel('Índice de la Firma')
plt.ylabel('Ángulo Mínimo (radianes)')

# Ordenar ángulos mínimos y graficar con líneas verticales en subplot (3,3,8)
sorted_indices = np.argsort(min_angles)
sorted_min_angles = min_angles[sorted_indices]

plt.subplot(3, 3, 8)
plt.vlines(range(len(sorted_min_angles)), 0, sorted_min_angles, colors='b', linestyles='solid', linewidth=0.5)
#plt.title('Ángulos Mínimos Ordenados')
plt.xlabel('Índice Ordenado')
plt.ylabel('Ángulo (radianes)')

# Filtrar firmas seleccionadas con máximo en el rango [0, 2500] y graficar en subplot (3,3,9)
max_intensities = np.max(firmas_seleccionadas, axis=1)
range_intensity_mask = (max_intensities >= 0) & (max_intensities <= 2500)
range_intensity_firmas = firmas_seleccionadas[range_intensity_mask]

plt.subplot(3, 3, 9)
for firma in range_intensity_firmas:
    plt.plot(wavelengths, firma)
plt.title('Firmas Seleccionadas max [0, 2500]')
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

