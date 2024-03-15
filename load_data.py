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

    # Omitir la leyenda si hay muchas líneas para mantener la claridad
    # if num_firmas <= 10:
    #     plt.legend()

    plt.show()
