import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

with h5py.File('results_old/test_cocoa_hdsp_sam015_ultra_small_kmeans3.h5', 'r') as f:
    X = f['spec'][:]
    y = f['label'][:]

# mean and std of X por class

plt.figure(figsize=(8, 6))

for i in range(3):
    mask = y.squeeze() == i
    X_class = X[mask]
    mean = X_class.mean(axis=0)
    std = X_class.std(axis=0)
    plt.plot(mean, label=f'Class {i + 1}')
    plt.fill_between(range(mean.shape[0]), mean - std, mean + std, alpha=0.3)

plt.legend()
plt.xlabel('Spectral Index')
plt.ylabel('Intensity')
plt.title('Mean and std of Cocoa dataset by class')

plt.grid()
plt.tight_layout()
plt.show()
