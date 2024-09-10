import os
import h5py
import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt



with h5py.File('results/test_real_cocoa_hdsp_oneCenter_squarelots.h5', 'r') as f:
    wavelengths = f['wavelengths'][:]
    X = f['spec'][:]
    y = f['label'][:]

X_mean = X.mean(axis=1)
X_std = X.std(axis=1)

X_standarize = (X - X_mean[:, None]) / X_std[:, None]
X_standarize_mean = X_standarize.mean(axis=1)

# plot samples with different colors for each class

ferm_levels = [60, 66, 73, 84, 85, 92, 94, 96]
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'orange']

plt.figure(figsize=(10, 5))

for i in range(8):
    X_class = X_mean[y[:, 0] == i]
    plt.plot(wavelengths, X_class[::100].T, color=colors[i], alpha=0.5)
    plt.scatter(wavelengths.min(), X_class.min(), color=colors[i], label=f'ferm level {ferm_levels[i]}')

plt.title('cocoa mean')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# compute pca with X_mean using sklearn

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_mean)

plt.figure(figsize=(10, 5))

for i in range(8):
    X_class = X_pca[y[:, 0] == i]
    plt.scatter(X_class[:, 0], X_class[:, 1], color=colors[i], alpha=0.5, label=f'ferm level {ferm_levels[i]}')

plt.title('cocoa mean PCA')
plt.grid()

plt.legend()
plt.tight_layout()
plt.show()

# plot samples with different colors for each class

plt.figure(figsize=(10, 5))

for i in range(8):
    X_class = X_standarize_mean[y[:, 0] == i]
    plt.plot(wavelengths, X_class[::100].T, color=colors[i], alpha=0.5)
    plt.scatter(wavelengths.min(), X_class.min(), color=colors[i], label=f'ferm level {ferm_levels[i]}')

plt.title('cocoa standarize')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# compute pca with X_mean using sklearn

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standarize_mean)

plt.figure(figsize=(10, 5))

for i in range(8):
    X_class = X_pca[y[:, 0] == i]
    plt.scatter(X_class[:, 0], X_class[:, 1], color=colors[i], alpha=0.5, label=f'ferm level {ferm_levels[i]}')

plt.title('cocoa standarize mean PCA')
plt.grid()

plt.legend()
plt.tight_layout()
plt.show()


print('Fin')
