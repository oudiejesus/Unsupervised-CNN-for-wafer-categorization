import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

encoded_all_data = np.load('encoded_all_data.npy')
pca_data = np.load('pca_data.npy')
k_values = range(4, 10)

fig = plt.figure(figsize=(15, 10))
fig.suptitle('Klusteroitu PCA-data', fontsize=16)

for i, k in enumerate(k_values, start=1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_data)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    ax = fig.add_subplot(2, 3, i, projection='3d')

    scatter = ax.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        pca_data[:, 2],
        c=labels, cmap='tab10'
    )
    ax.set_title(f'k = {k}', fontsize=10)
    ax.set_xlabel('Ulottuvuus 1')
    ax.set_ylabel('Ulottuvuus 2')
    ax.set_zlabel('Ulottuvuus 3')
plt.show()
