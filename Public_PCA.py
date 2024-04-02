import pandas as pd
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Ruta al archivo CSV
archivo_csv = 'firmas_espectrales_cacao.csv'

# Cargar el archivo CSV en un DataFrame
df_cacao = pd.read_csv(archivo_csv)

# Mostrar las primeras filas del DataFrame para verificar que se cargó correctamente
print(df_cacao.head())
print(df_cacao.shape)


# Separar las características (X) de las etiquetas (y)
X = df_cacao.iloc[:, :-1]  # Todas las filas, todas las columnas excepto la última
y = df_cacao.iloc[:, -1]   # Todas las filas, solo la última columna


'''
# Aplicar PCA
pca = PCA(n_components=2)  # Instanciar el PCA para 2 componentes principales
X_pca = pca.fit_transform(X)  # Ajustar y transformar los datos

# Graficar los resultados del PCA
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=40, alpha=0.7)
plt.title('PCA de Firmas Espectrales de Cacao')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(scatter, label='Etiqueta')
plt.show()



# Configurar t-SNE
tsne = TSNE(n_components=2, random_state=42)  # Usar un estado aleatorio para reproducibilidad
# Aplicar t-SNE a las características (X)
X_tsne = tsne.fit_transform(X)
# Graficar los resultados de t-SNE
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', edgecolor='k', s=40, alpha=0.7)
plt.title('t-SNE de Firmas Espectrales de Cacao')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(scatter, label='Etiqueta')
plt.show()
'''

def preparar_evaluar_modelo(datos, etiquetas, realizar_pca=True, test_size=0.25, random_state=42, save_path=None):
    # Imputación de valores NaN en los datos
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    datos_imputados = imputer.fit_transform(datos)

    # Estandarización de los datos imputados
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos_imputados)

    # Aplicar PCA si se requiere
    if realizar_pca:
        pca = PCA(n_components=2)
        datos_escalados = pca.fit_transform(datos_escalados)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(datos_escalados, etiquetas, test_size=test_size, random_state=random_state)

    # Entrenar el modelo SVC
    svc_model = SVC(kernel='linear')
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


# Llamar a la función
preparar_evaluar_modelo(X, y, realizar_pca=False, test_size=0.95, random_state=42)
