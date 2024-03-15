import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Cargar los datos
blanco = pd.read_csv("C:/Users/USUARIO/Documents/dataset_cocoa_hdsp/BLANCO_1.csv", header=None)
lote3ej1 = pd.read_csv("C:/Users/USUARIO/Documents/dataset_cocoa_hdsp/LOTE3EJ_1.csv", header=None)
# Repite para los demás archivos según sea necesario


blanco_ref = blanco.iloc[1, :]  # Segunda fila como referencia
wavelengths = blanco.iloc[0, :]  # Primera fila para las longitudes de onda

# Cálculo del factor de escala y ajuste
factor_de_escala = lote3ej1.iloc[1, 1608] / blanco_ref.iloc[1608]  # Asumiendo que la columna 1609 en MATLAB es 1608 en Python (base 0)
blanco_escalado = blanco_ref * factor_de_escala

blanco_ref = blanco.iloc[1, :]  # Segunda fila como referencia
wavelengths = blanco.iloc[0, :]  # Primera fila para las longitudes de onda

# Cálculo del factor de escala y ajuste
factor_de_escala = lote3ej1.iloc[1, 1608] / blanco_ref.iloc[1608]  # Asumiendo que la columna 1609 en MATLAB es 1608 en Python (base 0)
blanco_escalado = blanco_ref * factor_de_escala



# Visualización de blanco escalado vs. blanco saturado
plt.figure()
plt.plot(wavelengths, blanco_escalado, label='Blanco escalado')
plt.plot(wavelengths, lote3ej1.iloc[1, :], label='Blanco saturado')  # Asumiendo que quieres comparar estos dos
plt.legend()
plt.show()

# Para visualizar las firmas espectrales
plt.figure()
plt.plot(wavelengths, blanco_escalado, label='Blanco escalado')
# Usar slicing para seleccionar rangos específicos si es necesario
plt.plot(wavelengths, lote3ej1.iloc[3::100, :], label='LOTE3EJ1')
plt.legend()
plt.show()

# Repite la visualización para los demás lotes según sea necesario



# Visualización de blanco escalado vs. blanco saturado
plt.figure()
plt.plot(wavelengths, blanco_escalado, label='Blanco escalado')
plt.plot(wavelengths, lote3ej1.iloc[1, :], label='Blanco saturado')  # Asumiendo que quieres comparar estos dos
plt.legend()
plt.show()

# Para visualizar las firmas espectrales
plt.figure()
plt.plot(wavelengths, blanco_escalado, label='Blanco escalado')
# Usar slicing para seleccionar rangos específicos si es necesario
plt.plot(wavelengths, lote3ej1.iloc[3::100, :], label='LOTE3EJ1')
plt.legend()
plt.show()

# Repite la visualización para los demás lotes según sea necesario
