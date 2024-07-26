import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

cacao1_VIS = loadmat('experimento_cacao_1_fermentation.mat')
cacao2_VIS = loadmat('experimento_cacao_2_fermentation.mat')
cacao_NIR =  loadmat('experimento_cacao_3_fermantation_NIR.mat')


# fermentation level: bad=0, neutral=1, good=2

cocoa_VIS = dict(
    open50_1=dict(data=cacao1_VIS['my_cacao_50_abierto_1'], label='bad'),
    closed50_1=dict(data=cacao1_VIS['my_cacao_50_cerrado_1'], label='bad'),
    open73_1=dict(data=cacao2_VIS['my_cacao_73_abierto_1'], label='neutral'),
    closed73_1=dict(data=cacao2_VIS['my_cacao_73_cerrado_1'], label='neutral'),
    open73_2=dict(data=cacao2_VIS['my_cacao_73_abierto_2'], label='good'),
    closed73_2=dict(data=cacao2_VIS['my_cacao_73_cerrado_2'], label='good'),
    open95_1=dict(data=cacao1_VIS['my_cacao_95_abierto_1'], label='bad'),
    closed95_1=dict(data=cacao1_VIS['my_cacao_95_cerrado_1'], label='bad'),
    open96_1=dict(data=cacao2_VIS['my_cacao_96_abierto_1'], label='good'),
    closed96_1=dict(data=cacao2_VIS['my_cacao_96_cerrado_1'], label='good')
)

# plot dataset where each pair consists of a spectral signature and a label
# plot in a single figure with legend

plt.figure()

for key, value in cocoa_VIS.items():
    plt.plot(value['data'].squeeze(), label=value['label'])

plt.legend()
plt.show()

# perform the same plot but separating each pair open an closed in different subplots

plt.figure()

for key, value in cocoa_VIS.items():
    plt.subplot(2, 5, list(cocoa_VIS.keys()).index(key) + 1)
    plt.plot(value['data'].squeeze())
    plt.title(key)

plt.show()

# plot in a same subplot the open an closed pairs

plt.figure(figsize=(20, 5))

index = 1
for i, (key, value) in enumerate(cocoa_VIS.items()):
    plt.subplot(1, 5, index)
    plt.plot(value['data'].squeeze(), label=key)

    if (i + 1) % 2 == 0:
        index += 1
        plt.legend()

plt.tight_layout()
plt.show()


# plot in a same subplot the open an closed pairs

plt.figure(figsize=(20, 5))

index = 1
for i, (key, value) in enumerate(cocoa_VIS.items()):
    plt.subplot(1, 5, index)
    plt.plot(value['data'].squeeze() / value['data'].max(), label=key)
    plt.title(value['label'])

    if (i + 1) % 2 == 0:
        index += 1
        plt.legend()

plt.tight_layout()
plt.show()


# build a dataset with the VIS data

X_VIS = []
y_VIS = []

for key, value in cocoa_VIS.items():
    if 'open' in key:
        X_VIS.append(value['data'].squeeze())
        # set label by a number
        if value['label'] == 'bad':
            y_VIS.append(0)
        elif value['label'] == 'neutral':
            y_VIS.append(1)
        else:
            y_VIS.append(2)

X_VIS = np.array(X_VIS)
y_VIS = np.array(y_VIS)

# compute pca for 2 components

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X_VIS)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# plot results

plt.figure(figsize=(8, 6))

for i in range(3):
    mask = y_VIS == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Class {i}')

plt.legend()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Cocoa dataset of open cocoa beans')

plt.tight_layout()
plt.show()

X_VIS = []
y_VIS = []

for key, value in cocoa_VIS.items():
    if 'closed' in key:
        X_VIS.append(value['data'].squeeze())
        # set label by a number
        if value['label'] == 'bad':
            y_VIS.append(0)
        elif value['label'] == 'neutral':
            y_VIS.append(1)
        else:
            y_VIS.append(2)

X_VIS = np.array(X_VIS)
y_VIS = np.array(y_VIS)

X_scaled = StandardScaler().fit_transform(X_VIS)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

for i in range(3):
    mask = y_VIS == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Class {i}')

plt.legend()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Cocoa dataset of closed cocoa beans')

plt.tight_layout()
plt.show()

print('end')
