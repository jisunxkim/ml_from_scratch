import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Assign each data point to the nearest centroid
            clusters = [[] for _ in range(self.n_clusters)]
            for x in X:
                distances = [np.linalg.norm(x - centroid) for centroid in self.centroids]
                nearest_cluster = np.argmin(distances)
                clusters[nearest_cluster].append(x)

            # Update centroids based on the mean of data points in each cluster
            new_centroids = []
            for cluster in clusters:
                new_centroid = np.mean(cluster, axis=0)
                new_centroids.append(new_centroid)
            new_centroids = np.array(new_centroids)

            # Check for convergence
            if np.array_equal(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [np.linalg.norm(x - centroid) for centroid in self.centroids]
            nearest_cluster = np.argmin(distances)
            predictions.append(nearest_cluster)
        return predictions

# Example usage:
X = np.array([
    [1, 2],
    [1, 3],
    [2, 2],
    [8, 8],
    [7, 9],
    [9, 8]
])

# Create KMeans instance
kmeans = KMeans(n_clusters=2)

# Fit the model
kmeans.fit(X)

# Get cluster assignments
cluster_assignments = kmeans.predict(X)
print("Cluster assignments:", cluster_assignments)
