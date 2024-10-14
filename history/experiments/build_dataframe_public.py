import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


# Ruta al archivo .mat
archivo_mat = "C:/Users/USUARIO/Documents/GitHub/dataset_HDSP_cocoa_karen_sanchez/data_for_classification.mat"

# Cargar el archivo .mat
datos = scipy.io.loadmat(archivo_mat)

# Extraer las matrices
cmp_data = datos['cmp']
hyperimg = datos['hyperimg']
mix1_gt = datos['mix1_gt']

# Crear un colormap personalizado
colores = [(0, 0, 0),   # Negro para 0
           (1, 1, 0),   # Amarillo para 1
           (0, 0, 1),   # Azul para 2
           (0, 1, 0)]   # Verde para 3
cmp_personalizado = ListedColormap(colores)

# Visualizar la matriz mix1_gt usando el colormap personalizado
plt.figure(figsize=(8, 6))
plt.imshow(mix1_gt, cmap=cmp_personalizado)
plt.colorbar(ticks=[0, 1, 2, 3])
plt.show()


# Contar las ocurrencias de cada valor
conteo_0 = np.count_nonzero(mix1_gt == 0)
conteo_1 = np.count_nonzero(mix1_gt == 1)
conteo_2 = np.count_nonzero(mix1_gt == 2)
conteo_3 = np.count_nonzero(mix1_gt == 3)

# Mostrar los conteos
print(f"Conteo de 0: {conteo_0}")
print(f"Conteo de 1: {conteo_1}")
print(f"Conteo de 2: {conteo_2}")
print(f"Conteo de 3: {conteo_3}")

print(f"Total firmas Cacao: {conteo_1+conteo_2+conteo_3}")



# Inicializar listas para almacenar las firmas espectrales y las etiquetas
firmas_espectrales = []
etiquetas = []

# Iterar sobre mix1_gt para extraer firmas espectrales con etiquetas 1, 2, y 3
for i in range(mix1_gt.shape[0]):
    for j in range(mix1_gt.shape[1]):
        etiqueta = mix1_gt[i, j]
        if etiqueta in [1, 2, 3]:  # Consideramos solo píxeles con estas etiquetas
            # Extraer la firma espectral de hyperimg
            firma_espectral = hyperimg[i, j, :]
            firmas_espectrales.append(firma_espectral)
            etiquetas.append(etiqueta)

# Crear un DataFrame con las firmas espectrales y las etiquetas
df_firmas = pd.DataFrame(firmas_espectrales)
df_firmas['Etiqueta'] = etiquetas

# Mostrar las primeras filas del DataFrame para verificar
print(df_firmas.head())
print(df_firmas.shape)
# Opcional: Guardar el DataFrame en un archivo CSV
df_firmas.to_csv('firmas_espectrales_cacao.csv', index=False)

