
import numpy as np
import pandas as pd
import scipy.io
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

def dividir_visualizar_imagen(datos, etiquetas, porcentaje_train=0.5):
    """
    Divide la imagen hiperespectral y sus etiquetas verticalmente, visualiza la división,
    guarda los conjuntos de entrenamiento y prueba en archivos CSV, e imprime la cantidad de clases.
    
    :param datos: La imagen hiperespectral como un array de NumPy.
    :param etiquetas: La matriz de etiquetas correspondiente.
    :param porcentaje_train: El porcentaje del conjunto de entrenamiento.
    :return: Nombres de los archivos CSV generados para entrenamiento y prueba.
    """
    colores = [(0, 0, 0), (1, 1, 0), (0, 0, 1), (0, 1, 0)]
    cmp_personalizado = ListedColormap(colores)
    
    corte = int(etiquetas.shape[1] * porcentaje_train)
    plt.figure(figsize=(10, 8))
    plt.imshow(etiquetas, cmap=cmp_personalizado)
    plt.axvline(x=corte, color='r', linestyle='--')
    plt.colorbar(ticks=[0, 1, 2, 3])
    plt.title("División de la Imagen Hiperespectral")
    plt.show()

    # Dividir y reestructurar los datos y etiquetas
    datos_train, datos_test = datos[:, :corte, :].reshape(-1, datos.shape[2]), datos[:, corte:, :].reshape(-1, datos.shape[2])
    etiquetas_train, etiquetas_test = etiquetas[:, :corte].flatten(), etiquetas[:, corte:].flatten()
    
    # Guardar los conjuntos de entrenamiento y prueba en archivos CSV
    df_train = pd.DataFrame(datos_train)
    df_train['Etiqueta'] = etiquetas_train
    df_test = pd.DataFrame(datos_test)
    df_test['Etiqueta'] = etiquetas_test
    
    # Filtrar los píxeles con etiqueta 0
    df_train = df_train[df_train['Etiqueta'] != 0]
    df_test = df_test[df_test['Etiqueta'] != 0]
    
    archivo_train = 'entrenamiento_cacao.csv'
    archivo_test = 'prueba_cacao.csv'
    df_train.to_csv(archivo_train, index=False)
    df_test.to_csv(archivo_test, index=False)
    
    # Contar e imprimir la cantidad de clases en cada conjunto
    clases_train = df_train['Etiqueta'].value_counts()
    clases_test = df_test['Etiqueta'].value_counts()
    print(f"Clases en entrenamiento:\n{clases_train}\n")
    print(f"Clases en prueba:\n{clases_test}\n")
    
    return archivo_train, archivo_test

# Usar la función y especificar la ruta al archivo .mat y el porcentaje de división deseado
archivo_mat = "C:/Users/USUARIO/Documents/GitHub/dataset_HDSP_cocoa_karen_sanchez/data_for_classification.mat"
datos = scipy.io.loadmat(archivo_mat)
hyperimg = datos['hyperimg']
mix1_gt = datos['mix1_gt']

archivo_train, archivo_test = dividir_visualizar_imagen(hyperimg, mix1_gt, porcentaje_train=0.8)
#print(f"Archivos generados: {archivo_train, archivo_test}")


# Cargar los DataFrames desde los archivos CSV
df_train = pd.read_csv(archivo_train)
df_test = pd.read_csv(archivo_test)

# Separar características (X) y etiquetas (y)
X_train = df_train.drop('Etiqueta', axis=1)
y_train = df_train['Etiqueta']
X_test = df_test.drop('Etiqueta', axis=1)
y_test = df_test['Etiqueta']

# Opcional: Estandarizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Mostrar las métricas de evaluación y la matriz de confusión
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))





