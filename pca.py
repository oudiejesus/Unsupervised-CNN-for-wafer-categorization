import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

images = np.load('data_all.npy')

print(images.shape)



#SPLIT DATA FOR TRAIN, PCA
#train_images, test_images = train_test_split(images, test_size=0.9, random_state=42, shuffle = False)
#dont shuffle so can compare to pca
#Number of wafers and channels OF TRAIN DATA 
#n_wafers = train_images.shape[0]
#n_y = train_images.shape[1]
#n_x = train_images.shape[2]
#n_channels = train_images.shape[3]
#flatten TRAIN data
#flattened_data = train_images.reshape(n_wafers, -1)  # Shape: (n_wafers, 47*47*n_channels)


#Number of wafers and channels OF ALL DATA
n_wafers2 = images.shape[0]
n_y2 = images.shape[1]
n_x2 = images.shape[2]
n_channels2 = images.shape[3]
#flatten ALL data
flattened_data_all = images.reshape(n_wafers2, -1)  # Shape: (n_wafers, 47*47*n_channels)

#normalize
scaler = StandardScaler()
flattened_data_all = scaler.fit_transform(flattened_data_all)

# Apply PCA
N=50
pca = PCA(n_components=N)  # Choose N principal components for reduction
pca.fit_transform(flattened_data_all) #train PCA
pca_result = pca.transform(flattened_data_all) #fit all the data to PCA

np.save('pca_data.npy', pca_result[:,0:3])

# PLOT THE EXPLAINED VARIANCES
fig, ax1 = plt.subplots(figsize=(8, 4))
color = 'tab:blue'
ax1.bar(range(1,N+1), pca.explained_variance_ratio_, color = color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylabel("Selitetyn varianssin osuus", color=color)
ax1.set_xlabel("Pääkomponentti")

ax2 = ax1.twinx()
color = 'tab:red'
ax2.tick_params(axis='y', labelcolor=color)
ax2.plot(range(1,N+1), np.cumsum(pca.explained_variance_ratio_), color=color)
ax2.set_ylabel("Kumulatiivinen selitetyn varianssin osuus", color=color)
fig.tight_layout()
plt.show()


#SCATTER PLOT OF THE FIRST 2 PRINCIPAL COMPONENTS
plt.figure(figsize=(8, 6))
n_points = pca_result.shape[0]
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.xlabel('Pääkomponentti 1')
plt.ylabel('Pääkomponentti 2')
plt.title('PCA: 2 komponenttia')
plt.grid()
plt.show()


# Esimerkki 3 komponentin PCA-tuloksesta
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter-plot 3D
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], alpha=0.5)

# Akselien nimet ja otsikko
ax.set_xlabel('Pääkomponentti 1')
ax.set_ylabel('Pääkomponentti 2')
ax.set_zlabel('Pääkomponentti 3')
ax.set_title('PCA: 3 komponenttia')

plt.show()




#RECONSTRUCT DATA
#wafers_reconstructed = pca.inverse_transform(pca_result)
#original_shape = n_y * n_x * n_channels  # 47 * 47 * n_channels

#reshape to (n_wafers, 47 * 47 * n_channels)
#wafers_reconstructed = wafers_reconstructed.reshape(n_wafers, original_shape)
# reshape to the original 4D shape (n_wafers, n_y, n_x, n_channels)
#wafers_reconstructed = wafers_reconstructed.reshape(n_wafers, n_y, n_x, n_channels)

#Print the shape of the reconstructed data
#print("Shape of reconstructed data:", wafers_reconstructed.shape)

#visualize the first channel of the first wafer for verification
#plt.imshow(images[0, :, :, 0], cmap='gray')
#plt.title('Wafer 1, Channel 0')
#plt.axis('off')
#plt.show()
#plt.imshow(wafers_reconstructed[0, :, :, 0], cmap='gray')
#plt.title('Reconstructed Wafer 0, Channel 0')
#plt.axis('off')
#plt.show()


#DISTANCE FUNCTION
#def euclidean_distance(original, reconstructed):
#    return np.sqrt(np.sum((original - reconstructed)**2, axis=1)).flatten()
#distances = euclidean_distance(images, wafers_reconstructed)

#print("Shape of distances variable",distances)
#mean_distance = np.mean(distances)
#median_distance = np.median(distances)
#distribution of distances
#plt.figure(figsize=(8, 6))
#plt.hist(distances, bins=20, color='blue', alpha=0.7, edgecolor='black', label='Distances')
#plt.axvline(mean_distance, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_distance:.2f}')
#plt.axvline(median_distance, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_distance:.2f}')
#plt.title('Distribution of Euclidean Distances (Reconstruction Errors)')
#plt.xlabel('Euclidean Distance')
#plt.ylabel('Frequency')
#plt.legend()
#plt.grid(True)
#plt.show()
