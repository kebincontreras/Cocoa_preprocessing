import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Ruta de la carpeta donde están los archivos
path = "C://Users//USUARIO//Documents//UIS_Cacao//Base_Datos_Cacao//cocoa_ultimate_dataset//cocoa_ultimate_dataset//"

# Cargar archivos .mat
blanco_VIS = loadmat(path + 'BLANCO_EXP50G_010824_VIS.mat')['BLANCO']
negro_VIS = loadmat(path + 'NEGRO_EXP50G_010824_VIS.mat')['NEGRO']
lote_VIS = loadmat(path + 'LOTE4EXP50G_010824_VIS.mat')['CAPTURA_SP']
loteMIX_VIS = loadmat(path + 'LOTEMIXEXP50G_010824_VIS.mat')['CAPTURA_SP']

blanco_NIR = loadmat(path + 'BLANCO_EXP50G_010824_NIR.mat')['BLANCO']
negro_NIR = loadmat(path + 'NEGRO_EXP50G_010824_NIR.mat')['NEGRO']
lote_NIR = loadmat(path + 'LOTE4EXP50G_010824_NIR.mat')['CAPTURA_SP']
loteMIX_NIR = loadmat(path + 'LOTEMIXEXP50G_010824_NIR.mat')['CAPTURA_SP']

# Calcular el promedio de las imágenes blanco y negro
blanco_VIS_mean = blanco_VIS.mean(axis=0)*21
negro_VIS_mean = negro_VIS.mean(axis=0)
blanco_NIR_mean = blanco_NIR.mean(axis=0)
negro_NIR_mean = negro_NIR.mean(axis=0)

# Calcular reflectancia
reflectancia_lote_VIS = (lote_VIS - negro_VIS_mean) / (blanco_VIS_mean - negro_VIS_mean)
reflectancia_loteMIX_VIS = (loteMIX_VIS - negro_VIS_mean) / (blanco_VIS_mean - negro_VIS_mean)


#reflectancia_lote_NIR = (lote_NIR - negro_NIR_mean) / (blanco_NIR_mean - negro_NIR_mean)
#reflectancia_loteMIX_NIR = (loteMIX_NIR - negro_NIR_mean) / (blanco_NIR_mean - negro_NIR_mean)

reflectancia_lote_NIR = (lote_NIR ) / (blanco_NIR_mean)
reflectancia_loteMIX_NIR = (loteMIX_NIR ) / (blanco_NIR_mean)

# Cargar longitudes de onda y ajustar si es necesario
wavelengths_VIS = np.squeeze(loadmat(path + 'wavelengths_VIS.mat')['wavelengths'])
wavelengths_NIR = np.squeeze(loadmat(path + 'wavelengths_NIR.mat')['wavelengths'])

# Graficar en 4 subplots
plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 1)
plt.plot(wavelengths_VIS, reflectancia_lote_VIS.T, label='Lote VIS')
plt.title('Reflectancia Lote VIS')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Reflectancia')
plt.ylim(0, 1)  
plt.xlim(500, 900)  

plt.subplot(2, 2, 2)
plt.plot(wavelengths_VIS, reflectancia_loteMIX_VIS.T, label='Lote MIX VIS')
plt.title('Reflectancia Lote MIX VIS')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Reflectancia')
plt.ylim(0, 1) 
plt.xlim(500, 900)  


plt.subplot(2, 2, 3)
plt.plot(wavelengths_NIR, reflectancia_lote_NIR.T, label='Lote NIR')
plt.title('Reflectancia Lote NIR')
plt.xlabel('Longitud de onda (nm)')


plt.subplot(2, 2, 4)
plt.plot(wavelengths_NIR, reflectancia_loteMIX_NIR.T, label='Lote MIX NIR')
plt.title('Reflectancia Lote MIX NIR')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Reflectancia')

plt.tight_layout()
plt.show()



# Crear una nueva figura para los datos blanco y negro
plt.figure(figsize=(14, 6))

# Subplot para datos VIS
plt.subplot(1, 2, 1)
plt.plot(wavelengths_VIS, blanco_VIS_mean, label='Blanco VIS', color='blue')
plt.plot(wavelengths_VIS, negro_VIS_mean, label='Negro VIS', color='black')
plt.title('Firmas Espectrales Blanco y Negro - VIS')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Intensidad media')
plt.legend()

# Subplot para datos NIR
plt.subplot(1, 2, 2)
plt.plot(wavelengths_NIR, blanco_NIR_mean, label='Blanco NIR', color='blue')
plt.plot(wavelengths_NIR, negro_NIR_mean, label='Negro NIR', color='black')
plt.title('Firmas Espectrales Blanco y Negro - NIR')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Intensidad media')
plt.legend()

plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Seleccionar 100 firmas aleatorias de cada lote
np.random.seed(0)  # Para reproducibilidad
indices_VIS = np.random.choice(lote_VIS.shape[0], 100, replace=False)
indices_NIR = np.random.choice(lote_NIR.shape[0], 100, replace=False)

# Crear una nueva figura para los datos de firmas aleatorias
plt.figure(figsize=(14, 6))

# Subplot para datos VIS
plt.subplot(1, 2, 1)
for idx in indices_VIS:
    adjusted_signature = lote_VIS[idx, :] + negro_VIS_mean  # Sumar la señal del negro a cada firma
    plt.plot(wavelengths_VIS, adjusted_signature, color='gray', alpha=0.5)  # Firmas aleatorias ajustadas
plt.plot(wavelengths_VIS, blanco_VIS_mean, label='Blanco VIS', color='blue')
plt.plot(wavelengths_VIS, negro_VIS_mean, label='Negro VIS', color='black')
plt.title('100 Firmas Aleatorias Ajustadas y Referencias - VIS')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Intensidad')
plt.legend()

# Subplot para datos NIR
plt.subplot(1, 2, 2)
for idx in indices_NIR:
    adjusted_signature = lote_NIR[idx, :] + negro_NIR_mean  # Sumar la señal del negro a cada firma
    plt.plot(wavelengths_NIR, adjusted_signature, color='gray', alpha=0.5)  # Firmas aleatorias ajustadas
plt.plot(wavelengths_NIR, blanco_NIR_mean, label='Blanco NIR', color='blue')
plt.plot(wavelengths_NIR, negro_NIR_mean, label='Negro NIR', color='black')
plt.title('100 Firmas Aleatorias Ajustadas y Referencias - NIR')
plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Intensidad')
plt.legend()

plt.tight_layout()
plt.show()