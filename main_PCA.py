from preProcessing import cargar_datos, graficar_firmas_espectrales, graficar_reflectancia, preparar_evaluar_modelo, crear_directorio_resultados, graficar_firmas_medias, realizar_y_graficar_pca_con_listas, realizar_y_graficar_tsne_con_listas
import pandas as pd
import numpy as np

ruta_base = "C:/Users/USUARIO/Documents/dataset_cocoa_hdsp"
nombres_lotes = ["D_82_1.csv", "D_82_2.csv", "D_82_3.csv", "D_82_4.csv", "F_89_1.csv", "F_89_2.csv","F_89_3.csv","F_89_4.csv", "C_82_1.csv","C_82_2.csv","C_82_3.csv","C_82_4.csv", "A_50_1.csv", "A_50_2.csv"]
nombres_etiquetas = ["D 0.82", "D 0.82","D 0.82","D 0.82", "F 0.89", "F 0.89", "F 0.89", "F 0.89", "C 0.82", "C 0.82", "C 0.82", "C 0.82", "A 0.50" , "A 0.50"]


#nombres_lotes = ["D_82_4.csv","F_89_4.csv", "A_50_2.csv"]
#nombres_etiquetas = ["D 0.82", "F 0.89", "A 0.50"]

lista_reflectancias = []
save_path = crear_directorio_resultados()

for nombre_lote in nombres_lotes:
    lote, blanco_saturado, blanco_ref, wavelengths = cargar_datos(ruta_base, "BLANCO_2.csv", nombre_lote)
    titulo = nombre_lote.split('.')[0]
    reflectancia = graficar_reflectancia(lote, blanco_ref, wavelengths, 200, titulo, save_path)
    lista_reflectancias.append(reflectancia)
    graficar_firmas_espectrales(blanco_ref, lote, wavelengths, blanco_saturado, 200, titulo, save_path)

realizar_y_graficar_pca_con_listas(lista_reflectancias, nombres_etiquetas, save_path)
realizar_y_graficar_tsne_con_listas(lista_reflectancias, nombres_etiquetas, save_path, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, random_state=42)
graficar_firmas_medias(lista_reflectancias, wavelengths, nombres_etiquetas, save_path)
preparar_evaluar_modelo(lista_reflectancias, nombres_etiquetas, realizar_pca=True, test_size=0.99, random_state=42, save_path="metrics.txt")





