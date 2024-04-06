
import numpy as np
import pandas as pd
import scipy.io
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os

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



def procesar_datos(archivo_train, archivo_test, ruta_guardado, test_size_val=0.1, estandarizar=True):
    # Cargar los DataFrames desde los archivos CSV
    df_train = pd.read_csv(archivo_train)
    df_test = pd.read_csv(archivo_test)

    # Separar características (X) y etiquetas (y)
    Xtrain = df_train.drop('Etiqueta', axis=1)
    Ytrain = df_train['Etiqueta']
    Xtest = df_test.drop('Etiqueta', axis=1)
    Ytest = df_test['Etiqueta']

    # Dividir Xtest y Ytest en conjuntos de validación y test
    Xtest, Xval, Ytest, Yval = train_test_split(Xtest, Ytest, test_size=test_size_val, random_state=42)

    # Opcional: Estandarizar los datos
    if estandarizar:
        scaler = StandardScaler()
        Xtrain = scaler.fit_transform(Xtrain)
        Xtest = scaler.transform(Xtest)
        Xval = scaler.transform(Xval)

    # Crear la carpeta si no existe
    if not os.path.exists(ruta_guardado):
        os.makedirs(ruta_guardado)

    # Guardar los conjuntos de datos en archivos .npy
    np.save(os.path.join(ruta_guardado, 'Xtrain.npy'), Xtrain)
    np.save(os.path.join(ruta_guardado, 'Ytrain.npy'), Ytrain)
    np.save(os.path.join(ruta_guardado, 'Xtest.npy'), Xtest)
    np.save(os.path.join(ruta_guardado, 'Ytest.npy'), Ytest)
    np.save(os.path.join(ruta_guardado, 'Xval.npy'), Xval)
    np.save(os.path.join(ruta_guardado, 'Yval.npy'), Yval)




# Usar la función y especificar la ruta al archivo .mat y el porcentaje de división deseado
archivo_mat = "C:/Users/USUARIO/Documents/GitHub/dataset_HDSP_cocoa_karen_sanchez/data_for_classification.mat"
datos = scipy.io.loadmat(archivo_mat)
hyperimg = datos['hyperimg']
mix1_gt = datos['mix1_gt']

# Paso 1: Dividir y Visualizar los Datos
archivo_train, archivo_test = dividir_visualizar_imagen(hyperimg, mix1_gt, porcentaje_train=0.5)

# Paso 2: Procesar los Datos
# Asegúrate de especificar la ruta correcta donde quieres guardar los archivos .npy
ruta_guardado = './Dataset/cocoa_public'
procesar_datos(archivo_train, archivo_test, ruta_guardado, test_size_val=0.1, estandarizar=True)

# Paso 3: Entrenar el Modelo SVM
# Cargar los conjuntos de datos
Xtrain = np.load(os.path.join(ruta_guardado, 'Xtrain.npy'))
Ytrain = np.load(os.path.join(ruta_guardado, 'Ytrain.npy'))
Xtest = np.load(os.path.join(ruta_guardado, 'Xtest.npy'))
Ytest = np.load(os.path.join(ruta_guardado, 'Ytest.npy'))
Xval = np.load(os.path.join(ruta_guardado, 'Xval.npy'))  # Si necesitas usarlo más adelante
Yval = np.load(os.path.join(ruta_guardado, 'Yval.npy'))  # Si necesitas usarlo más adelante

# Entrenar el modelo
model = SVC(kernel='linear')
model.fit(Xtrain, Ytrain)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(Xtest)

# Mostrar las métricas de evaluación y la matriz de confusión
print("Matriz de Confusión:")
print(confusion_matrix(Ytest, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(Ytest, y_pred))





