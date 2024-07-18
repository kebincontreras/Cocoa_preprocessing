import h5py
from sklearn.cluster import k_means
from sklearn.preprocessing import StandardScaler

with h5py.File('results_old/test_cocoa_hdsp_sam015_ultra_small.h5', 'r') as f:
    X = f['spec'][:]
    y = f['label'][:]

# kmeans with 3 clusters

cluster_centers, labels, _ = k_means(X, n_clusters=3, n_init='auto', random_state=0)

# plot clusters and std

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

for i in range(3):
    mask = labels.squeeze() == i
    X_class = X[mask]
    std = X_class.std(axis=0)
    plt.plot(cluster_centers[i], label=f'Class {i + 1}')
    plt.fill_between(range(cluster_centers[i].shape[0]), cluster_centers[i] - std, cluster_centers[i] + std, alpha=0.3)

plt.legend()
plt.xlabel('Spectral Index')
plt.ylabel('Intensity')
plt.title('Mean and std of Cocoa dataset by class')

plt.grid()
plt.tight_layout()
plt.show()
