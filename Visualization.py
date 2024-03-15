import pandas as pd
from load_data import cargar_datos, graficar_firmas_espectrales
import matplotlib.pyplot as plt


# Uso de la función
ruta_base = "C:/Users/USUARIO/Documents/dataset_cocoa_hdsp"
nombre_blanco = "BLANCO_1.csv"
nombre_lote = "LOTE3EJ_1.csv"
NO_firmas = 200

lote, blanco_saturado, blanco_ref, wavelengths = cargar_datos(ruta_base, nombre_blanco, nombre_lote)
graficar_firmas_espectrales(blanco_ref, lote, wavelengths, blanco_saturado, NO_firmas)



