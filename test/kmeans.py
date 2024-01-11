from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(0)
data = np.random.rand(100, 2)

# Choose the number of clusters
K = 3

# Create the K-means model
kmeans = KMeans(n_clusters=K)

# Fit the model to the data
kmeans.fit(data)

# Get cluster labels and cluster centers
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

print(labels)
print(cluster_centers)

# Visualize the clusters
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.show()
