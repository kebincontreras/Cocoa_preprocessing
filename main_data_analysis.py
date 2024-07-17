import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

with h5py.File('Results/train_cocoa_hdsp_sam015_ultra_small.h5', 'r') as f:
    X = f['spec'][:]
    y = f['label'][:]

X_scaled = StandardScaler().fit_transform(X)

# pca 2d

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# plot

plt.figure(figsize=(8, 6))

for i in range(5):
    mask = y.squeeze() == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Class {i + 1}', alpha=0.1)

plt.legend()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Cocoa dataset')

plt.tight_layout()
plt.show()

# pca 3d

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# plot

fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection='3d')

for i in range(5):
    mask = y.squeeze() == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], label=f'Class {i + 1}', alpha=0.1)

ax.legend()
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA of Cocoa dataset')

plt.tight_layout()
plt.show()


# t-SNE 2d

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X_scaled)

# plot

plt.figure(figsize=(8, 6))

for i in range(5):
    mask = y.squeeze() == i
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=f'Class {i + 1}')

plt.legend()

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Cocoa dataset')

plt.tight_layout()
plt.show()

# t-SNE 3d

tsne = TSNE(n_components=3, random_state=0)
X_tsne = tsne.fit_transform(X_scaled)

# plot

fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection='3d')

for i in range(5):
    mask = y.squeeze() == i
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], X_tsne[mask, 2], label=f'Class {i + 1}')

ax.legend()

ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
ax.set_title('t-SNE of Cocoa dataset')

plt.tight_layout()
plt.show()
