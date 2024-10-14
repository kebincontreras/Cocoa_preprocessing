import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os

# Ruta a la carpeta que contiene los archivos .mat
folder_path = ''


# Función para cargar y procesar espectros
def cargar_espectro(filename):
    print(filename)
    data = loadmat(filename)
    # Asume que los datos son la última variable almacenada en el archivo
    spectral_data = data[list(data.keys())[-1]]
    # Asegúrate de que el array está correctamente dimensionado
    if spectral_data.shape[0] == 1 or spectral_data.shape[1] == 1:
        spectral_data = spectral_data.flatten()
    return spectral_data


# Contar el número de archivos de lotes, excluyendo b1 y N1
num_lotes = sum(1 for file in os.listdir(folder_path) if file not in ['b1.mat', 'N1.mat'])

# Preparar los colores para los diferentes lotes
colors = plt.cm.jet(np.linspace(0, 1, num_lotes))

# Inicializar el plot
plt.figure(figsize=(12, 8))

# Variables para los espectros base (negro y blanco)
espectro_negro = cargar_espectro(os.path.join(folder_path, 'B1.mat'))
espectro_blanco = cargar_espectro(os.path.join(folder_path, 'N1.mat'))
if espectro_blanco.ndim > 1:
    espectro_blanco = np.mean(espectro_blanco, axis=0)  # Promedio a lo largo de las filas
if espectro_negro.ndim > 1:
    idx_max = np.argmax(np.max(espectro_negro, axis=1))
    espectro_negro = espectro_negro[idx_max, :]

# Procesar cada archivo .mat en el directorio
i = 0
for file in sorted(os.listdir(folder_path)):
    if file not in ['B1.mat', 'N1.mat']:  # Excluir los archivos de espectros negro y blanco
        filepath = os.path.join(folder_path, file)
        espectros_lote = cargar_espectro(filepath)

        if espectros_lote.ndim > 1:
            reflectancia = (espectros_lote - espectro_negro[None, :]) / (
                        espectro_negro[None, :] - espectro_blanco[None, :])
            for firma in reflectancia:
                plt.plot(firma, color=colors[i], label=f'Lote {file.split(".")[0]}' if i == 0 else "")
        i += 1

plt.title('Reflectancia de Firmas Espectrales de Cacao')
plt.xlabel('Número de Punto Espectral')
plt.ylabel('Reflectancia')
# plt.legend()
plt.show()
