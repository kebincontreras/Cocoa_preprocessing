from load_data import cargar_datos, graficar_firmas_espectrales, graficar_reflectancia, realizar_y_graficar_pca_con_listas, graficar_firmas_medias, preparar_evaluar_modelo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Esta es la corrección
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Definir la ruta base y los nombres de los archivos
ruta_base = "C:/Users/USUARIO/Documents/dataset_cocoa_hdsp"
nombres_lotes = ["LOTE3EJ_1.csv", "LOTE3EJ_2.csv", "LOTE3EJ_3.csv", "LOTE3EJ_4.csv", "LOTE4EJ_1.csv", "LOTE4EJ_2.csv","LOTE4EJ_3.csv","LOTE4EJ_4.csv",   "LOTE6EJ1.csv","LOTE6EJ2.csv","LOTE6EJ3.csv","LOTE6EJ4.csv", "LOTE2EJ_A_1.csv", "LOTE2EJ_A_2.csv"]
nombres_etiquetas = ["D 0.82", "D 0.82","D 0.82","D 0.82", "F 0.89", "F 0.89", "F 0.89", "F 0.89",  "C 0.82", "C 0.82", "C 0.82", "C 0.82", "A 0.50" , "A 0.50"]

# Lista para almacenar los DataFrames de reflectancia
lista_reflectancias = []


for nombre_lote in nombres_lotes:
    lote, blanco_saturado, blanco_ref, wavelengths = cargar_datos(ruta_base, "BLANCO_1.csv", nombre_lote)
    titulo = nombre_lote.split('.')[0]  # Esto asume que deseas usar el nombre del archivo sin la extensión como título
    reflectancia = graficar_reflectancia(lote, wavelengths, 200, titulo)  # Asume que esta función devuelve el DataFrame de reflectancia
    lista_reflectancias.append(reflectancia)




preparar_evaluar_modelo(lista_reflectancias, nombres_etiquetas, realizar_pca=True, test_size=0.9995, random_state=42)
graficar_firmas_medias(lista_reflectancias, wavelengths, nombres_etiquetas)