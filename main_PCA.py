import pandas as pd
from load_data import cargar_datos, graficar_firmas_espectrales, graficar_reflectancia, realizar_y_graficar_pca_con_listas
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Definir la ruta base y los nombres de los archivos
ruta_base = "C:/Users/USUARIO/Documents/dataset_cocoa_hdsp"
nombres_lotes = ["LOTE3EJ_1.csv", "LOTE4EJ_2.csv", "LOTE6EJ1.csv"]  
nombres_etiquetas = ["D 0.82", "F 0.89", "C 0.82"]  
# Lista para almacenar los DataFrames de reflectancia
lista_reflectancias = []
for nombre_lote in nombres_lotes:
    lote, blanco_saturado, blanco_ref, wavelengths = cargar_datos(ruta_base, "BLANCO_1.csv", nombre_lote)
    reflectancia = graficar_reflectancia(lote, wavelengths, 200)  # Asume que esta función devuelve el DataFrame de reflectancia
    lista_reflectancias.append(reflectancia)

# Llamar a la función para realizar PCA y graficar los resultados
realizar_y_graficar_pca_con_listas(lista_reflectancias, nombres_etiquetas)
