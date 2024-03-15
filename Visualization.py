import pandas as pd
from load_data import cargar_datos 
import matplotlib.pyplot as plt


# Uso de la función
ruta_base = "C:/Users/USUARIO/Documents/dataset_cocoa_hdsp"
nombre_blanco = "BLANCO_1.csv"
nombre_lote = "LOTE3EJ_1.csv"

lote, blanco_saturado, blanco_ref, wavelengths = cargar_datos(ruta_base, nombre_blanco, nombre_lote)

# Cálculo del factor de escala y ajuste
factor_de_escala = lote.iloc[1, 1608] / blanco_ref.iloc[1608]  
blanco_escalado = blanco_ref * factor_de_escala


# Cálculo del factor de escala y ajuste
factor_de_escala = lote.iloc[1, 1608] / blanco_ref.iloc[1608]  
blanco_escalado = blanco_ref * factor_de_escala

# Visualización de blanco escalado vs. blanco saturado
plt.figure()
plt.plot(wavelengths, blanco_escalado, label='Blanco escalado')
plt.plot(wavelengths, lote.iloc[1, :], label='Blanco saturado')  # Asumiendo que quieres comparar estos dos
plt.legend()
plt.show()

# Para visualizar las firmas espectrales
plt.figure()
plt.plot(wavelengths, blanco_escalado, label='Blanco escalado')
# Usar slicing para seleccionar rangos específicos si es necesario
plt.plot(wavelengths, lote.iloc[3::100, :], label='LOTE3EJ1')
plt.legend()
plt.show()




