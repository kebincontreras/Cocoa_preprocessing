import pandas as pd

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