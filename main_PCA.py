'''
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

'''


from load_data import cargar_datos, graficar_firmas_espectrales, graficar_reflectancia, realizar_y_graficar_pca_con_listas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Esta es la corrección
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Definir la ruta base y los nombres de los archivos
ruta_base = "C:/Users/USUARIO/Documents/dataset_cocoa_hdsp"
nombres_lotes = ["LOTE3EJ_3.csv", "LOTE4EJ_2.csv", "LOTE6EJ4.csv", "LOTE2EJ_A_1.csv"]
nombres_etiquetas = ["D 0.82", "F 0.89", "C 0.82", "A 0.50"]

# Lista para almacenar los DataFrames de reflectancia
lista_reflectancias = []

for nombre_lote in nombres_lotes:
    lote, blanco_saturado, blanco_ref, wavelengths = cargar_datos(ruta_base, "BLANCO_1.csv", nombre_lote)
    reflectancia = graficar_reflectancia(lote, wavelengths, 200)  # Asume que esta función devuelve el DataFrame de reflectancia
    lista_reflectancias.append(reflectancia)


# Concatenar todos los DataFrames de reflectancia en uno solo y preparar las etiquetas
datos = pd.concat(lista_reflectancias, ignore_index=True)
etiquetas = np.repeat(nombres_etiquetas, [len(df) for df in lista_reflectancias])

# Realizar PCA y graficar los resultados (opcional, si quieres incluir este paso)
realizar_y_graficar_pca_con_listas(lista_reflectancias, nombres_etiquetas)

# Preparar los datos para el entrenamiento del modelo SVC
# Imputación de valores NaN en los datos
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
datos_imputados = imputer.fit_transform(datos)

# Estandarización de los datos imputados
scaler = StandardScaler()
datos_escalados = scaler.fit_transform(datos_imputados)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(datos_escalados, etiquetas, test_size=0.99, random_state=42)

# Entrenar el modelo SVC
svc_model = SVC(kernel='linear')  # Ajusta el kernel según sea necesario
svc_model.fit(X_train, y_train)

# Realizar predicciones y evaluar el modelo
y_pred = svc_model.predict(X_test)
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))


