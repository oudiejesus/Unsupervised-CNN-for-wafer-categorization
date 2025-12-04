import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

images = np.load('data_all.npy')

images = images.reshape(3949,14,2,109,74)
images = images[:, :-1, :, :, :] #no defect kanavat pois

slice_array = images[0, 0, 0, :, :]  # Shape: (H, W)

# Count the number of -1 entries
count_minus_one = np.sum(slice_array == -1)

print("Number of -1 entries:", count_minus_one)

countsums = np.sum(images[:, :, 0, :, :], axis=(1, 2, 3)) +(13 * 1646)  # Sum over channels, height, and width
areasums = np.sum(images[:, :, 1, :, :], axis=(1, 2, 3)) + (13 * 1646)


#indices of datapoints that satisfy the condition
valid_indices = (countsums < 10000) & (areasums < 15000)


# filter the dataset 
filtered_images = images[valid_indices, :, :, :, :]

print("Original dataset shape:", images.shape)
print("Filtered dataset shape:", filtered_images.shape)




plt.figure(figsize=(10, 6))
plt.bar(range(len(countsums)), countsums, color='blue', alpha=0.7)
plt.plot(range(len(areasums)), areasums, color='red', linestyle='-', linewidth=0.2)
plt.title("Sum for Each Datapoint (Depth Fixed to k=0)")
plt.xlabel("Datapoint Index")
plt.ylabel("Sum")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()