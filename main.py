import pandas as pd
from load_data import cargar_datos, graficar_firmas_espectrales, graficar_reflectancia
import matplotlib.pyplot as plt
import pandas as pd

# Uso de la función
ruta_base = "C:/Users/USUARIO/Documents/dataset_cocoa_hdsp"
nombre_blanco = "BLANCO_1.csv"
nombre_lote = "LOTE3EJ_1.csv"
NO_firmas = 200 # Cantidad de firmas a graficar
lote, blanco_saturado, blanco_ref, wavelengths = cargar_datos(ruta_base, nombre_blanco, nombre_lote)
print("Número total de filas en el lote:", lote.shape[0])
graficar_firmas_espectrales(blanco_ref, lote, wavelengths, blanco_saturado, NO_firmas)
graficar_reflectancia(lote, wavelengths, NO_firmas)

