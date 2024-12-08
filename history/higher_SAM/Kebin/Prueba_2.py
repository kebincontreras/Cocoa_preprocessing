import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA

# Ruta de la carpeta donde están los archivos
# path = "C://Users//USUARIO//Documents//UIS_Cacao//Base_Datos_Cacao//cocoa_ultimate_dataset//cocoa_ultimate_dataset//"
path = '/home/enmartz/Jobs/cacao/Base_Datos_Cacao/cocoa_ultimate_dataset_entrega2/'

# Cargar archivos .mat
blanco_VIS = loadmat(path + 'BLANCO_EXP50G_010824_VIS.mat')['BLANCO']
negro_VIS = loadmat(path + 'NEGRO_EXP50G_010824_VIS.mat')['NEGRO']
lote_VIS = loadmat(path + 'LOTE4EXP50G_010824_VIS.mat')['CAPTURA_SP']
loteMIX_VIS = loadmat(path + 'LOTEMIXEXP50G_010824_VIS.mat')['CAPTURA_SP']

blanco_NIR = loadmat(path + 'BLANCO_EXP50G_010824_NIR.mat')['BLANCO']
negro_NIR = loadmat(path + 'NEGRO_EXP50G_010824_NIR.mat')['NEGRO']
lote_NIR = loadmat(path + 'LOTE4EXP50G_010824_NIR.mat')['CAPTURA_SP']
loteMIX_NIR = loadmat(path + 'LOTEMIXEXP50G_010824_NIR.mat')['CAPTURA_SP']

# Cargar longitudes de onda
wavelengths_VIS = np.squeeze(loadmat(path + 'wavelengths_VIS.mat')['wavelengths'])
wavelengths_NIR = np.squeeze(loadmat(path + 'wavelengths_NIR.mat')['wavelengths'])

# Primera figura: Graficar las firmas puras (sin reflectancia) junto con blanco y negro
plt.figure(figsize=(14, 12))

# 1. Firmas puras de Lote VIS + Blanco y Negro VIS
plt.subplot(2, 2, 1)
plt.plot(wavelengths_VIS, lote_VIS.T, label='Lote VIS', color='gray', alpha=0.7)
plt.plot(wavelengths_VIS, blanco_VIS.mean(axis=0)*21, label='Blanco VIS', color='blue')
plt.plot(wavelengths_VIS, negro_VIS.mean(axis=0), label='Negro VIS', color='black')
plt.title('Firmas Puras VIS + Blanco y Negro')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Intensidad')
plt.xlim(300, 1100)


# 2. Firmas puras de Lote MIX VIS + Blanco y Negro VIS
plt.subplot(2, 2, 2)
plt.plot(wavelengths_VIS, loteMIX_VIS.T, label='Lote MIX VIS', color='gray', alpha=0.7)
plt.plot(wavelengths_VIS, blanco_VIS.mean(axis=0)*21, label='Blanco VIS', color='blue')
plt.plot(wavelengths_VIS, negro_VIS.mean(axis=0), label='Negro VIS', color='black')
plt.title('Firmas Puras MIX VIS + Blanco y Negro')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Intensidad')
plt.xlim(300, 1100)


# 3. Firmas puras de Lote NIR + Negro NIR sumado + Blanco y Negro NIR
plt.subplot(2, 2, 3)
lote_NIR_adjusted = lote_NIR + negro_NIR.mean(axis=0)  # Sumar el negro a las firmas de NIR
plt.plot(wavelengths_NIR, lote_NIR_adjusted.T, label='Lote NIR + Negro', color='gray', alpha=0.7)
plt.plot(wavelengths_NIR, blanco_NIR.mean(axis=0), label='Blanco NIR', color='blue')
plt.plot(wavelengths_NIR, negro_NIR.mean(axis=0), label='Negro NIR', color='black')
plt.title('Firmas Puras NIR + Negro NIR Sumado + Blanco y Negro')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Intensidad')


# 4. Firmas puras de Lote MIX NIR + Negro NIR sumado + Blanco y Negro NIR
plt.subplot(2, 2, 4)
loteMIX_NIR_adjusted = loteMIX_NIR + negro_NIR.mean(axis=0)  # Sumar el negro a las firmas de MIX NIR
plt.plot(wavelengths_NIR, loteMIX_NIR_adjusted.T, label='Lote MIX NIR + Negro', color='gray', alpha=0.7)
plt.plot(wavelengths_NIR, blanco_NIR.mean(axis=0), label='Blanco NIR', color='blue')
plt.plot(wavelengths_NIR, negro_NIR.mean(axis=0), label='Negro NIR', color='black')
plt.title('Firmas Puras MIX NIR + Negro NIR Sumado + Blanco y Negro')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Intensidad')


plt.tight_layout()
plt.show()

# 2. Graficar la reflectancia (VIS y NIR) después del corte de NIR
# Filtrar las longitudes de onda NIR para el rango de 1000 a 2000 nm
nir_mask = (wavelengths_NIR >= 1000) & (wavelengths_NIR <= 2000)
wavelengths_NIR_filtered = wavelengths_NIR[nir_mask]

# Filtrar las firmas NIR correspondientes
lote_NIR_filtered = lote_NIR[:, nir_mask]
loteMIX_NIR_filtered = loteMIX_NIR[:, nir_mask]

# Calcular reflectancia
reflectancia_lote_VIS = (lote_VIS - negro_VIS.mean(axis=0)) / (blanco_VIS.mean(axis=0) - negro_VIS.mean(axis=0))
reflectancia_loteMIX_VIS = (loteMIX_VIS - negro_VIS.mean(axis=0)) / (blanco_VIS.mean(axis=0) - negro_VIS.mean(axis=0))
reflectancia_lote_NIR_filtered = lote_NIR_filtered / blanco_NIR.mean(axis=0)[nir_mask]
reflectancia_loteMIX_NIR_filtered = loteMIX_NIR_filtered / blanco_NIR.mean(axis=0)[nir_mask]

plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 1)
plt.plot(wavelengths_VIS, reflectancia_lote_VIS.T, label='Lote VIS')
plt.title('Reflectancia Lote VIS')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Reflectancia')
plt.ylim(0, 1)
plt.xlim(500, 900)
#plt.xlim(300, 1100)


plt.subplot(2, 2, 2)
plt.plot(wavelengths_VIS, reflectancia_loteMIX_VIS.T, label='Lote MIX VIS')
plt.title('Reflectancia Lote MIX VIS')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Reflectancia')
plt.ylim(0, 1)
plt.xlim(500, 900)
#plt.xlim(300, 1100)


plt.subplot(2, 2, 3)
plt.plot(wavelengths_NIR_filtered, reflectancia_lote_NIR_filtered.T, label='Lote NIR Filtrado')
plt.title('Reflectancia Lote NIR Filtrado (1000-2000 nm)')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Reflectancia')


plt.subplot(2, 2, 4)
plt.plot(wavelengths_NIR_filtered, reflectancia_loteMIX_NIR_filtered.T, label='Lote MIX NIR Filtrado')
plt.title('Reflectancia Lote MIX NIR Filtrado (1000-2000 nm)')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Reflectancia')
#plt.legend()

plt.tight_layout()
plt.show()

# 3. Graficar PCA para VIS y NIR filtrado
# Realizar PCA para las firmas VIS y NIR
pca = PCA(n_components=2)
pca_VIS = pca.fit_transform(np.vstack((lote_VIS, loteMIX_VIS)))
pca_NIR = pca.fit_transform(np.vstack((lote_NIR_filtered, loteMIX_NIR_filtered)))

# Obtener la varianza explicada por cada componente
explained_variance_VIS = pca.explained_variance_ratio_ * 100
explained_variance_NIR = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 1)
plt.scatter(pca_VIS[:lote_VIS.shape[0], 0], pca_VIS[:lote_VIS.shape[0], 1], alpha=0.7, label='Lote VIS', color='blue')
plt.scatter(pca_VIS[lote_VIS.shape[0]:, 0], pca_VIS[lote_VIS.shape[0]:, 1], alpha=0.7, label='Lote MIX VIS', color='red')
plt.title(f'PCA VIS (Varianza Explicada: {explained_variance_VIS[0]:.2f}%, {explained_variance_VIS[1]:.2f}%)')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(pca_VIS[:lote_VIS.shape[0], 0], pca_VIS[:lote_VIS.shape[0], 1], alpha=0.7, label='Lote VIS', color='blue')
plt.scatter(pca_VIS[lote_VIS.shape[0]:, 0], pca_VIS[lote_VIS.shape[0]:, 1], alpha=0.7, label='Lote MIX VIS', color='red')
plt.title(f'PCA VIS (Varianza Explicada: {explained_variance_VIS[0]:.2f}%, {explained_variance_VIS[1]:.2f}%)')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(pca_NIR[:lote_NIR_filtered.shape[0], 0], pca_NIR[:lote_NIR_filtered.shape[0], 1], alpha=0.7, label='Lote NIR', color='blue')
plt.scatter(pca_NIR[lote_NIR_filtered.shape[0]:, 0], pca_NIR[lote_NIR_filtered.shape[0]:, 1], alpha=0.7, label='Lote MIX NIR', color='red')
plt.title(f'PCA NIR (Varianza Explicada: {explained_variance_NIR[0]:.2f}%, {explained_variance_NIR[1]:.2f}%)')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(pca_NIR[:lote_NIR_filtered.shape[0], 0], pca_NIR[:lote_NIR_filtered.shape[0], 1], alpha=0.7, label='Lote NIR', color='blue')
plt.scatter(pca_NIR[lote_NIR_filtered.shape[0]:, 0], pca_NIR[lote_NIR_filtered.shape[0]:, 1], alpha=0.7, label='Lote MIX NIR', color='red')
plt.title(f'PCA NIR (Varianza Explicada: {explained_variance_NIR[0]:.2f}%, {explained_variance_NIR[1]:.2f}%)')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend()

plt.tight_layout()
plt.show()


# 4. Graficar PCA para VIS y NIR filtrado
# Realizar PCA para las firmas VIS y NIR
pca = PCA(n_components=2)
pca_VIS = pca.fit_transform(np.vstack((lote_VIS, loteMIX_VIS)))
pca_NIR = pca.fit_transform(np.vstack((lote_NIR_filtered, loteMIX_NIR_filtered)))

# Obtener la varianza explicada por cada componente
explained_variance_VIS = pca.explained_variance_ratio_ * 100
explained_variance_NIR = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 1)
plt.scatter(pca_VIS[:lote_VIS.shape[0], 0], pca_VIS[:lote_VIS.shape[0], 1], alpha=0.7, label='Lote VIS', color='blue')
plt.scatter(pca_VIS[lote_VIS.shape[0]:, 0], pca_VIS[lote_VIS.shape[0]:, 1], alpha=0.7, label='Lote MIX VIS', color='red')
plt.title(f'PCA VIS (Varianza Explicada: {explained_variance_VIS[0]:.2f}%, {explained_variance_VIS[1]:.2f}%)')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(pca_VIS[:lote_VIS.shape[0], 0], pca_VIS[:lote_VIS.shape[0], 1], alpha=0.7, label='Lote VIS', color='blue')
plt.scatter(pca_VIS[lote_VIS.shape[0]:, 0], pca_VIS[lote_VIS.shape[0]:, 1], alpha=0.7, label='Lote MIX VIS', color='red')
plt.title(f'PCA VIS (Varianza Explicada: {explained_variance_VIS[0]:.2f}%, {explained_variance_VIS[1]:.2f}%)')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(pca_NIR[:lote_NIR_filtered.shape[0], 0], pca_NIR[:lote_NIR_filtered.shape[0], 1], alpha=0.7, label='Lote NIR', color='blue')
plt.scatter(pca_NIR[lote_NIR_filtered.shape[0]:, 0], pca_NIR[lote_NIR_filtered.shape[0]:, 1], alpha=0.7, label='Lote MIX NIR', color='red')
plt.title(f'PCA NIR (Varianza Explicada: {explained_variance_NIR[0]:.2f}%, {explained_variance_NIR[1]:.2f}%)')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(pca_NIR[:lote_NIR_filtered.shape[0], 0], pca_NIR[:lote_NIR_filtered.shape[0], 1], alpha=0.7, label='Lote NIR', color='blue')
plt.scatter(pca_NIR[lote_NIR_filtered.shape[0]:, 0], pca_NIR[lote_NIR_filtered.shape[0]:, 1], alpha=0.7, label='Lote MIX NIR', color='red')
plt.title(f'PCA NIR (Varianza Explicada: {explained_variance_NIR[0]:.2f}%, {explained_variance_NIR[1]:.2f}%)')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend()

plt.tight_layout()
plt.show()