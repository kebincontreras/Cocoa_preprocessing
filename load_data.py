import pandas as pd
import matplotlib.pyplot as plt

def cargar_datos(ruta_base, nombre_blanco, nombre_lote):

    ruta_blanco = f"{ruta_base}/{nombre_blanco}"
    ruta_lote = f"{ruta_base}/{nombre_lote}"
    
    blanco = pd.read_csv(ruta_blanco, header=None)
    lote = pd.read_csv(ruta_lote, header=None)
    
    # Extracción de blanco_ref y wavelengths
    blanco_ref = blanco.iloc[1, :]
    wavelengths = blanco.iloc[0, :]
    
    # Cálculo de blanco saturado (segunda fila del lote)
    blanco_saturado = lote.iloc[1, :]

    return lote, blanco_saturado, blanco_ref, wavelengths


def graficar_firmas_espectrales(blanco_ref, lote, wavelengths, blanco_saturado, num_firmas):
   
    # Cálculo del factor de escala y ajuste
    factor_de_escala = lote.iloc[1, 1608] / blanco_ref.iloc[1608]
    blanco_escalado = blanco_ref * factor_de_escala

    # Visualización de blanco escalado vs. blanco saturado
    plt.figure()
    plt.plot(wavelengths, blanco_escalado, label='Blanco escalado')
    plt.plot(wavelengths, blanco_saturado, label='Blanco saturado')
    plt.legend()
    plt.show(block=False)

    # Visualización de firmas espectrales seleccionadas del lote contra el blanco escalado
    plt.figure()
    plt.plot(wavelengths, blanco_escalado, label='Blanco escalado', linewidth=2, color='k')

    # Asegurar que no se intenta graficar más firmas de las disponibles
    num_firmas = min(num_firmas, lote.shape[0] - 3)

    for i in range(3, 3 + num_firmas):
        plt.plot(wavelengths, lote.iloc[i, :], label=f'Firma LOTE - Muestra {i}')

    plt.show(block=False)


def graficar_reflectancia(lote, wavelengths,NO_firmas):

    # Extracción de las mediciones necesarias del DataFrame
    I_negro = lote.iloc[2, :]
    I_blanco = lote.iloc[1, :]
    I_muestra = lote.iloc[3:, :]

    # Cálculo de la reflectancia para cada muestra
    reflectancia = (I_muestra - I_negro) / (I_blanco - I_negro)

    # Creación de la figura para graficar
    plt.figure(figsize=(10, 6))

    # Graficar las 10 primeras filas de 'reflectancia'
    for index, row in reflectancia.iloc[:NO_firmas].iterrows():
        #plt.plot(wavelengths, row, linestyle='-', marker='', linewidth=1, label=f'Muestra {index + 1 - 3}') 
        plt.plot(wavelengths, row, linestyle='-', marker='', linewidth=1, label=f'Muestra {index + 1}')

    # Configuración de la gráfica
    plt.xlabel('Longitud de Onda')
    plt.ylabel('Reflectancia')
    plt.title('Curvas de Reflectancia')
    plt.xlim([450, 900])
    plt.ylim([0, 1])
    #plt.legend()
    plt.show()

    return reflectancia