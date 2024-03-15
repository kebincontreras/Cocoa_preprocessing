import torch
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data into pandas DataFrames and then convert to PyTorch tensors
def load_csv_to_tensor(filename):
    return torch.tensor(pd.read_csv(filename, header=None).values)

# Loading data
BLANCO = load_csv_to_tensor('BLANCO_1.csv')
BLANCO_REF = BLANCO[1, :]
wavelengths = BLANCO[0, :]

# Load other datasets
LOTE3EJ1 = load_csv_to_tensor('LOTE3EJ_1.csv')
LOTE3EJ2 = load_csv_to_tensor('LOTE3EJ_2.csv')
LOTE3EJ3 = load_csv_to_tensor('LOTE3EJ_3.csv')
LOTE3EJ4 = load_csv_to_tensor('LOTE3EJ_4.csv')
LOTE4EJ1 = load_csv_to_tensor('LOTE4EJ_1.csv')
LOTE4EJ2 = load_csv_to_tensor('LOTE4EJ_2.csv')
LOTE4EJ3 = load_csv_to_tensor('LOTE4EJ_3.csv')
LOTE4EJ4 = load_csv_to_tensor('LOTE4EJ_4.csv')
LOTE6EJ1 = load_csv_to_tensor('LOTE6EJ_1.csv')
LOTE6EJ2 = load_csv_to_tensor('LOTE6EJ_2.csv')
LOTE6EJ3 = load_csv_to_tensor('LOTE6EJ_3.csv')
LOTE6EJ4 = load_csv_to_tensor('LOTE6EJ_4.csv')

# Calculate the scale factor and scaled white reference
BLANCO_SATURADO = LOTE3EJ1[1, :]
FACTOR_DE_ESCALA_8CAPTURAS = BLANCO_SATURADO[1608] / BLANCO_REF[1608]  # Adjusted indexing for Python (0-based)
BLANCO_ESCALADO = BLANCO_REF * FACTOR_DE_ESCALA_8CAPTURAS

# Plotting functions
def plot_with_scale(wavelengths, reference, title, samples=None):
    plt.figure()
    plt.plot(wavelengths.numpy(), reference.numpy(), label='Scaled Reference')
    if samples is not None:
        for sample in samples:
            plt.plot(wavelengths.numpy(), sample.numpy()[3::100, :].squeeze(), label='Sample')
    plt.title(title)
    plt.ylim([0, 1])
    plt.xlim([400, 997])
    plt.legend()
    plt.show()

# Plot scaled reference vs saturated reference
plt.figure()
plt.plot(wavelengths.numpy(), BLANCO_ESCALADO.numpy(), label='BLANCO_ESCALADO')
plt.plot(wavelengths.numpy(), BLANCO_SATURADO.numpy(), label='BLANCO_SATURADO')
plt.legend()
plt.show()

# Example of plotting with the function
plot_with_scale(wavelengths, BLANCO_ESCALADO, 'Reflectance LOTE 3#1', [LOTE3EJ1, LOTE3EJ2])

# Calculating reflectance (example for LOTE3EJ1)
REFLECTANCE_L3EJ1 = (LOTE3EJ1 - LOTE3EJ1[2, :]) / (BLANCO_ESCALADO - LOTE3EJ1[2, :])
# For saving, you can convert tensors back to DataFrames and use the to_csv function
# pd.DataFrame(REFLECTANCE_L3EJ1.numpy()).to_csv('REFLECTANCE_L3EJ1.csv', mode='a', header=False)
