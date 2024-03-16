from load_data import cargar_datos, graficar_reflectancia, preparar_evaluar_modelo, crear_directorio_resultados, graficar_firmas_medias, realizar_y_graficar_pca_con_listas
import pandas as pd
import numpy as np

ruta_base = "C:/Users/USUARIO/Documents/dataset_cocoa_hdsp"
nombres_lotes = ["LOTE3EJ_1.csv", "LOTE3EJ_2.csv", "LOTE3EJ_3.csv", "LOTE3EJ_4.csv", "LOTE4EJ_1.csv", "LOTE4EJ_2.csv","LOTE4EJ_3.csv","LOTE4EJ_4.csv", "LOTE6EJ1.csv","LOTE6EJ2.csv","LOTE6EJ3.csv","LOTE6EJ4.csv", "LOTE2EJ_A_1.csv", "LOTE2EJ_A_2.csv"]
nombres_etiquetas = ["D 0.82", "D 0.82","D 0.82","D 0.82", "F 0.89", "F 0.89", "F 0.89", "F 0.89", "C 0.82", "C 0.82", "C 0.82", "C 0.82", "A 0.50" , "A 0.50"]
lista_reflectancias = []
save_path = crear_directorio_resultados()

for nombre_lote in nombres_lotes:
    lote, blanco_saturado, blanco_ref, wavelengths = cargar_datos(ruta_base, "BLANCO_1.csv", nombre_lote)
    titulo = nombre_lote.split('.')[0]
    reflectancia = graficar_reflectancia(lote, wavelengths, 200, titulo, save_path)
    lista_reflectancias.append(reflectancia)

realizar_y_graficar_pca_con_listas(lista_reflectancias, nombres_etiquetas, save_path)
graficar_firmas_medias(lista_reflectancias, wavelengths, nombres_etiquetas, save_path)
preparar_evaluar_modelo(lista_reflectancias, nombres_etiquetas, realizar_pca=True, test_size=0.95, random_state=42)





