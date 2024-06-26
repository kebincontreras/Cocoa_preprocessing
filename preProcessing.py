import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import os
from datetime import datetime
from sklearn.manifold import TSNE

def crear_directorio_resultados():
    base_dir = "Result"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base_dir, timestamp)
    os.makedirs(path, exist_ok=True)
    return path

def cargar_datos(ruta_base, nombre_blanco, nombre_lote):
    ruta_blanco = f"{ruta_base}/{nombre_blanco}"
    ruta_lote = f"{ruta_base}/{nombre_lote}"
    blanco = pd.read_csv(ruta_blanco, header=None)
    lote = pd.read_csv(ruta_lote, header=None)
    
    # Extraer los valores de longitud de onda
    wavelengths = lote.iloc[0, :]
    
    # Identificar las columnas dentro del rango deseado
    columnas_dentro_del_rango = (wavelengths >= 450) & (wavelengths <= 900)
    
    # Filtrar los DataFrames para incluir solo las columnas dentro del rango deseado
    blanco_filtrado = blanco.loc[:, columnas_dentro_del_rango]
    lote_filtrado = lote.loc[:, columnas_dentro_del_rango]
    
    # Actualizar 'blanco_ref' y 'blanco_saturado' para que solo incluyan las columnas filtradas
    blanco_ref = blanco_filtrado.iloc[0, :]
    blanco_saturado = lote_filtrado.iloc[1, :]

    # Asegurarse de que 'wavelengths' también esté filtrado
    wavelengths = lote_filtrado.iloc[0, :]

    return lote_filtrado, blanco_saturado, blanco_ref, wavelengths

def graficar_firmas_espectrales(blanco_ref, lote, wavelengths, blanco_saturado, num_firmas, titulo, save_path):
    # Crea la ruta para la nueva carpeta dentro de save_path llamada "Firmas_espectral"
    new_save_path = os.path.join(save_path, "Firmas_espectral")
    
    # Verifica si la carpeta existe, si no, la crea
    if not os.path.exists(new_save_path):
        os.makedirs(new_save_path)

    # Cálculo del factor de escala y ajuste para el blanco
    #factor_de_escala = lote.iloc[2, :].max() / blanco_ref.max()  # Ajusta esta línea según tu lógica de escala
    #blanco_escalado = blanco_ref * factor_de_escala

    #factor_de_escala = 
    blanco_escalado = blanco_ref




    # Gráfica de blanco escalado vs. blanco saturado
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, blanco_escalado, label='Blanco_referencia', color='black', linewidth=2)
    plt.plot(wavelengths, blanco_saturado, label='Blanco saturado', color='grey', linestyle='--')
    plt.title(f'{titulo} - Blanco referencia vs. Blanco saturado')
    plt.xlabel('Longitud de Onda (nm)')
    plt.ylabel('Intensidad')
    plt.legend()
    plt.savefig(os.path.join(new_save_path, f"{titulo}_Blanco_Referencia_vs_Saturado.png"))
    plt.close()

    # Gráfica de firmas espectrales de las muestras
    plt.figure(figsize=(10, 6))
    for i in range(3, 3 + num_firmas):  # Ajusta los índices según tus datos
        plt.plot(wavelengths, lote.iloc[i, :], label=f'Muestra {i - 2}')
    plt.title(f'{titulo} - Firmas espectrales')
    plt.xlabel('Longitud de Onda (nm)')
    plt.ylabel('Intensidad')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(new_save_path, f"{titulo}_Firmas_Espectrales.png"))
    plt.close()





def graficar_reflectancia(lote, blanco_ref, wavelengths, NO_firmas, titulo, save_path):
    # Ruta para la nueva carpeta dentro de save_path
    new_save_path = os.path.join(save_path, "reflectancia")
    
    # Verificar si la carpeta existe, si no, crearla
    if not os.path.exists(new_save_path):
        os.makedirs(new_save_path)
    
    I_negro = lote.iloc[2, :]
    #I_blanco = lote.iloc[1, :]
    I_blanco = blanco_ref
    I_muestra = lote.iloc[3:, :]
    reflectancia = (I_muestra - I_negro) / (I_blanco - I_negro)
    
    plt.figure(figsize=(10, 6))
    for index, row in reflectancia.iloc[:NO_firmas].iterrows():
        plt.plot(wavelengths, row, linestyle='-', marker='', linewidth=1, label=f'Muestra {index + 1}')
    plt.xlabel('Longitud de Onda')
    plt.ylabel('Reflectancia')
    plt.title(titulo)
    plt.xlim([300, 1000])
    plt.ylim([0, 1])
    
    # Guardar la figura en la nueva ruta
    plt.savefig(os.path.join(new_save_path, f"{titulo}.png"))
    plt.close()
    
    return reflectancia


def realizar_y_graficar_pca_con_listas(lista_reflectancias, lista_etiquetas, save_path):
    if len(lista_reflectancias) != len(lista_etiquetas):
        raise ValueError("El número de conjuntos de datos de reflectancia y etiquetas debe coincidir.")
    
    datos = pd.concat(lista_reflectancias, ignore_index=True)
    etiquetas = []
    for reflectancia, etiqueta in zip(lista_reflectancias, lista_etiquetas):
        etiquetas.extend([etiqueta] * len(reflectancia))
    
    # Imputación de valores faltantes
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    datos_imputados = imputer.fit_transform(datos)
    
    # Estandarización de los datos antes de PCA
    datos_escalados = StandardScaler().fit_transform(datos_imputados)  # Corregido para usar datos_imputados
    
    # Aplicar PCA
    pca = PCA(n_components=2)
    componentes_principales = pca.fit_transform(datos_escalados)
    df_pca = pd.DataFrame(data=componentes_principales, columns=['PC1', 'PC2'])
    df_pca['Etiqueta'] = etiquetas
    
    # Graficar los resultados de PCA
    plt.figure(figsize=(8, 6))
    for etiqueta in set(etiquetas):
        indices = df_pca['Etiqueta'] == etiqueta
        plt.scatter(df_pca.loc[indices, 'PC1'], df_pca.loc[indices, 'PC2'], label=etiqueta, alpha=0.5)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.title('PCA de Reflectancia')
    #plt.xlim([-50, 100])
    #plt.ylim([-50, 50])
    plt.savefig(os.path.join(save_path, "PCA.png"))
    #plt.show()
    plt.close

def graficar_firmas_medias(lista_reflectancias, wavelengths, etiquetas, save_path):
    plt.figure(figsize=(10, 6))
    
    # Asumiendo que 'wavelengths' es un pandas Series o un ndarray que necesita ser convertido para comparación
    wavelengths = np.array(wavelengths)
    
    # Filtrar índices de longitudes de onda en el rango de 450 a 900
    indices = (wavelengths >= 400) & (wavelengths <= 900)
    
    # Colores para las gráficas
    colores = plt.cm.jet(np.linspace(0, 1, len(lista_reflectancias)))
    
    # Calcular y graficar la firma media para cada lote
    for i, (df, etiqueta) in enumerate(zip(lista_reflectancias, etiquetas)):
        # Filtrar las columnas de reflectancia por longitudes de onda
        df_filtrado = df.iloc[:, indices]
        
        # Calcular la firma media
        firma_media = df_filtrado.mean()
        
        # Graficar
        plt.plot(wavelengths[indices], firma_media, label=etiqueta, color=colores[i])
    
    # Configuración de la gráfica
    plt.xlabel('Longitud de Onda (nm)')
    plt.ylabel('Reflectancia Media')
    plt.title('Firmas media de Reflectancia')
    plt.legend()
    plt.savefig(os.path.join(save_path, "Firmas_Medias.png"))
    #plt.show()
    plt.close

def preparar_evaluar_modelo(lista_reflectancias, nombres_etiquetas, realizar_pca=True, test_size=0.95, random_state=42, save_path=None):
    # Concatenar todos los DataFrames de reflectancia en uno solo y preparar las etiquetas
    datos = pd.concat(lista_reflectancias, ignore_index=True)
    etiquetas = np.repeat(nombres_etiquetas, [len(df) for df in lista_reflectancias])

    # Preparar los datos para el entrenamiento del modelo SVC
    # Imputación de valores NaN en los datos
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    datos_imputados = imputer.fit_transform(datos)

    # Estandarización de los datos imputados
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos_imputados)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(datos_escalados, etiquetas, test_size=test_size, random_state=random_state)

    # Entrenar el modelo SVC
    svc_model = SVC(kernel='linear')  # Ajustar el kernel según sea necesario
    svc_model.fit(X_train, y_train)

    # Realizar predicciones y evaluar el modelo
    y_pred = svc_model.predict(X_test)
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # Guardar las métricas en un archivo de texto si se proporciona un path de guardado
    if save_path is not None:
        with open(save_path, "w") as f:
            f.write("Matriz de Confusión:\n")
            f.write(str(confusion_matrix(y_test, y_pred)))
            f.write("\n\nReporte de Clasificación:\n")
            f.write(classification_report(y_test, y_pred))







def realizar_y_graficar_tsne_con_listas(lista_reflectancias, lista_etiquetas, save_path, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, random_state=None):
    if len(lista_reflectancias) != len(lista_etiquetas):
        raise ValueError("El número de conjuntos de datos de reflectancia y etiquetas debe coincidir.")
    
    # Concatenar todos los DataFrames de reflectancia en uno solo
    datos = pd.concat(lista_reflectancias, ignore_index=True)
    
    # Preparar las etiquetas
    etiquetas = []
    for reflectancia, etiqueta in zip(lista_reflectancias, lista_etiquetas):
        etiquetas.extend([etiqueta] * len(reflectancia))
    
    # Imputación de valores faltantes
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    datos_imputados = imputer.fit_transform(datos)
    
    # Estandarización de los datos antes de t-SNE
    datos_escalados = StandardScaler().fit_transform(datos_imputados)
    
    # Aplicar t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=random_state)
    tsne_resultados = tsne.fit_transform(datos_escalados)
    
    # Graficar los resultados de t-SNE
    df_tsne = pd.DataFrame(data=tsne_resultados, columns=['Dimension 1', 'Dimension 2'])
    df_tsne['Etiqueta'] = etiquetas
    
    plt.figure(figsize=(8, 6))
    for etiqueta in set(etiquetas):
        indices = df_tsne['Etiqueta'] == etiqueta
        plt.scatter(df_tsne.loc[indices, 'Dimension 1'], df_tsne.loc[indices, 'Dimension 2'], label=etiqueta, alpha=0.5)
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.legend()
    plt.title('t-SNE de Reflectancia')
    plt.savefig(os.path.join(save_path, "t-SNE.png"))
    plt.close()



