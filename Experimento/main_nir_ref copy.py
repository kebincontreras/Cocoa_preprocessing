import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Directorio donde se encuentran los archivos
base_dir = r"C:\Users\USUARIO\Documents\GitHub\Dataset\Lab_hdsp_cocoa_experimento_jorge"

# Cargar los archivos
cacao_NIR = loadmat(os.path.join(base_dir, 'experimento_cacao_3_fermantation_NIR.mat'))

# Obtener las longitudes de onda
wavelength = cacao_NIR['wavelengths'].squeeze()

# Filtrar las longitudes de onda en el rango de 1000 a 2300 nm
mask = (wavelength >= 1000) & (wavelength <= 2300)

# Filtrar las longitudes de onda
filtered_wavelength = wavelength[mask]

# Filtrar los datos basados en las longitudes de onda seleccionadas
def filter_data(data, mask):
    return data[:, mask]

# Asegurarse de que el divisor tenga la misma longitud que los datos
blanconir = filter_data(cacao_NIR['my_blanconir'], mask)

# Aplicar el filtro a los datos y dividir por el blanconir filtrado
cocoa_NIR = dict(
    open73=dict(data=filter_data(cacao_NIR['my_cacaocnir_73_abierto'], mask) / blanconir, label='neutral'),
    closed73=dict(data=filter_data(cacao_NIR['my_cacaocnir_73_cerrado'], mask) / blanconir, label='neutral'),
    open73_2=dict(data=filter_data(cacao_NIR['my_cacaocnir_73_abierto_2'], mask) / blanconir, label='bad'),
    closed73_2=dict(data=filter_data(cacao_NIR['my_cacaocnir_73_cerrado_2'], mask) / blanconir, label='bad'),
    open85=dict(data=filter_data(cacao_NIR['my_cacaocnir_85_abierto'], mask) / blanconir, label='bad'),
    closed85=dict(data=filter_data(cacao_NIR['my_cacaocnir_85_cerrado'], mask) / blanconir, label='bad'),
    open85_2=dict(data=filter_data(cacao_NIR['my_cacaocnir_85_abierto_2'], mask) / blanconir, label='good'),
    closed85_2=dict(data=filter_data(cacao_NIR['my_cacaocnir_85_cerrado_2'], mask) / blanconir, label='good'),
    open94=dict(data=filter_data(cacao_NIR['my_cacaocnir_94_abierto'], mask) / blanconir, label='good'),
    closed94=dict(data=filter_data(cacao_NIR['my_cacaocnir_94_cerrado'], mask) / blanconir, label='good'),
    open94_2=dict(data=filter_data(cacao_NIR['my_cacaocnir_94_abierto_2'], mask) / blanconir, label='bad'),
    closed94_2=dict(data=filter_data(cacao_NIR['my_cacaocnir_94_cerrado_2'], mask) / blanconir, label='bad'),
    open96=dict(data=filter_data(cacao_NIR['my_cacaocnir_96_abierto'], mask) / blanconir, label='good'),
    closed96=dict(data=filter_data(cacao_NIR['my_cacaocnir_96_cerrado'], mask) / blanconir, label='good'),
    open96_2=dict(data=filter_data(cacao_NIR['my_cacaocnir_96_abierto_2'], mask) / blanconir, label='good'),
    closed96_2=dict(data=filter_data(cacao_NIR['my_cacaocnir_96_cerrado_2'], mask) / blanconir, label='good')
)

# plot dataset where each pair consists of a spectral signature and a label
# plot in a single figure with legend

plt.figure()

for key, value in cocoa_NIR.items():
    plt.plot(filtered_wavelength, value['data'].squeeze(), label=value['label'])

plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('NIR Spectral Signatures')
plt.show()

# perform the same plot but separating each pair open and closed in different subplots

plt.figure()

for key, value in cocoa_NIR.items():
    plt.subplot(2, 8, list(cocoa_NIR.keys()).index(key) + 1)
    plt.plot(filtered_wavelength, value['data'].squeeze())
    plt.title(key)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')

plt.tight_layout()
plt.show()

# plot in a same subplot the open and closed pairs

plt.figure(figsize=(20, 5))

index = 1
for i, (key, value) in enumerate(cocoa_NIR.items()):
    plt.subplot(2, 4, index)
    plt.plot(filtered_wavelength, value['data'].squeeze(), label=key)

    if (i + 1) % 2 == 0:
        index += 1
        plt.legend()

plt.tight_layout()
plt.show()

# plot in a same subplot the open and closed pairs

plt.figure(figsize=(20, 5))

index = 1
for i, (key, value) in enumerate(cocoa_NIR.items()):
    plt.subplot(2, 4, index)
    plt.plot(filtered_wavelength, value['data'].squeeze() / value['data'].max(), label=key)
    plt.title(value['label'])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Reflectance')

    if (i + 1) % 2 == 0:
        index += 1
        plt.legend()

plt.tight_layout()
plt.show()

# build a dataset with the NIR data

X_VIS = []
y_VIS = []

for key, value in cocoa_NIR.items():
    if 'open' in key:
        X_VIS.append(value['data'].squeeze())
        # set label by a number
        if value['label'] == 'bad':
            y_VIS.append(0)
        elif value['label'] == 'neutral':
            y_VIS.append(1)
        else:
            y_VIS.append(2)

X_VIS = np.array(X_VIS)
y_VIS = np.array(y_VIS)

# compute pca for 2 components

X_scaled = StandardScaler().fit_transform(X_VIS)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# plot results

plt.figure(figsize=(8, 6))

for i in range(3):
    mask = y_VIS == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Class {i}')

plt.legend()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Cocoa dataset of open cocoa beans')

plt.tight_layout()
plt.show()

X_VIS = []
y_VIS = []

for key, value in cocoa_NIR.items():
    if 'closed' in key:
        X_VIS.append(value['data'].squeeze())
        # set label by a number
        if value['label'] == 'bad':
            y_VIS.append(0)
        elif value['label'] == 'neutral':
            y_VIS.append(1)
        else:
            y_VIS.append(2)

X_VIS = np.array(X_VIS)
y_VIS = np.array(y_VIS)

X_scaled = StandardScaler().fit_transform(X_VIS)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

for i in range(3):
    mask = y_VIS == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Class {i}')

plt.legend()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Cocoa dataset of closed cocoa beans')

plt.tight_layout()
plt.show()

print('end')
