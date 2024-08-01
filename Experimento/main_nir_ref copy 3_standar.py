import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
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

# Estandarizar los datos
def standardize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data.T).T

# Aplicar el filtro a los datos, dividir por el blanconir filtrado, y luego estandarizar
cocoa_NIR = dict(
    open73=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_73_abierto'], mask) / blanconir), label='neutral_abierto'),
    closed73=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_73_cerrado'], mask) / blanconir), label='neutral_cerrado'),
    open73_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_73_abierto_2'], mask) / blanconir), label='bad_abierto'),
    closed73_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_73_cerrado_2'], mask) / blanconir), label='bad_cerrado'),
    open85=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_85_abierto'], mask) / blanconir), label='bad_abierto'),
    closed85=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_85_cerrado'], mask) / blanconir), label='bad_cerrado'),
    open85_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_85_abierto_2'], mask) / blanconir), label='good_abierto'),
    closed85_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_85_cerrado_2'], mask) / blanconir), label='good_cerrado'),
    open94=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_94_abierto'], mask) / blanconir), label='good_abierto'),
    closed94=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_94_cerrado'], mask) / blanconir), label='good_cerrado'),
    open94_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_94_abierto_2'], mask) / blanconir), label='bad_abierto'),
    closed94_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_94_cerrado_2'], mask) / blanconir), label='bad_cerrado'),
    open96=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_96_abierto'], mask) / blanconir), label='good_abierto'),
    closed96=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_96_cerrado'], mask) / blanconir), label='good_cerrado'),
    open96_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_96_abierto_2'], mask) / blanconir), label='good_abierto'),
    closed96_2=dict(data=standardize_data(filter_data(cacao_NIR['my_cacaocnir_96_cerrado_2'], mask) / blanconir), label='good_cerrado')
)

# plot dataset where each pair consists of a spectral signature and a label
# plot in a single figure with legend

plt.figure()

for key, value in cocoa_NIR.items():
    plt.plot(filtered_wavelength, value['data'].squeeze(), label=value['label'])

plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Standardized Reflectance')
plt.title('Standardized NIR Spectral Signatures')
plt.show()
