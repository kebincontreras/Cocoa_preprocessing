import os

import h5py
import numpy as np
from sklearn.cluster import k_means
from sklearn.preprocessing import StandardScaler

subset_name = 'train'

with h5py.File(f'Results/{subset_name}_cocoa_hdsp_sam015_ultra_small.h5', 'r') as f:
    X = f['spec'][:]
    y = f['label'][:]

# kmeans with 3 clusters

cluster_centers, labels, _ = k_means(X, n_clusters=3, n_init='auto', random_state=0)

# build h5 dataset

with h5py.File(os.path.join('Results', f'{subset_name}_cocoa_hdsp_sam015_ultra_small_kmeans3.h5'), 'w') as d:
    dataset = d.create_dataset('spec', shape=(0, 2000), maxshape=(None, 2000), chunks=(256, 2000),
                               dtype=np.float32)
    labelset = d.create_dataset('label', (0, 1), maxshape=(None, 1), chunks=(256, 1), dtype=np.uint8)

    # Append new data to dataset
    def append_to_dataset(dataset, new_data):
        current_shape = dataset.shape
        new_shape = (current_shape[0] + new_data.shape[0], current_shape[1])
        dataset.resize(new_shape)
        dataset[current_shape[0]:] = new_data

    min_max_intensity = np.argsort(np.max(cluster_centers, 1))

    for label, t_label in enumerate(min_max_intensity):
        mask = labels == t_label
        selected_samples = X[mask]
        new_labels = np.ones((len(selected_samples), 1), dtype=np.uint8) * label

        append_to_dataset(dataset, selected_samples)
        append_to_dataset(labelset, new_labels)
