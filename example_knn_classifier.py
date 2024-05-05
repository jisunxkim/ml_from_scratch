import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = [self.y_train[i] for i in nearest_indices]
            predicted_label = max(set(nearest_labels), key=nearest_labels.count)
            predictions.append(predicted_label)
        return predictions

# Example usage:
X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])
y_train = np.array([0, 0, 1, 1, 1])

X_test = np.array([
    [1, 3],
    [5, 5]
])

# Create KNN classifier instance
knn_classifier = KNNClassifier()

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Make predictions
predictions = knn_classifier.predict(X_test)
print(predictions)
