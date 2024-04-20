import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def kmeans(data, k, max_iter=100):
    # Initialize centroids randomly
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iter):
        # Assign each data point to the nearest centroid
        labels = np.argmin(((data - centroids[:, np.newaxis])**2).sum(axis=2), axis=0)

        # Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

def plot_clusters(data, labels, centroids):
    plt.figure(figsize=(8, 6))
    for i in range(k):
        plt.scatter(data[labels == i][:, 0], data[labels == i][:, 1], label=f'Cluster {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Centroids')
    plt.title('k-Means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

# Load data from CSV
df = pd.read_csv('sample_data.csv')
data = df.values

# Specify the number of clusters (k) and maximum iterations
k = 2
max_iter = 100

# Perform k-Means clustering
labels, centroids = kmeans(data, k, max_iter)

# Plot the clustered data
plot_clusters(data, labels, centroids)

# Output the clusters and final centroids
print("Clusters:", labels)
print("Final Centroids:")
for i, centroid in enumerate(centroids):
    print(f"Centroid {i+1}: {centroid}")
