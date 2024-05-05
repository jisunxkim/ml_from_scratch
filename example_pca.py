import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Calculate mean of each feature
        self.mean = np.mean(X, axis=0)

        # Center the data
        X_centered = X - self.mean

        # Calculate covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Calculate eigenvalues and eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Select top n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Center the data
        X_centered = X - self.mean

        # Project data onto the principal components
        return np.dot(X_centered, self.components)

# Example usage:
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# Create PCA instance
pca = PCA(n_components=2)

# Fit the model
pca.fit(X)

# Transform the data
X_transformed = pca.transform(X)
print(X_transformed)
