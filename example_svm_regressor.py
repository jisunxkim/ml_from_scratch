import numpy as np

class SVR:
    def __init__(self, C=1.0, epsilon=0.1, learning_rate=0.01, num_epochs=1000):
        self.C = C
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def _cost_function(self, y, predictions):
        errors = np.maximum(0, np.abs(y - predictions) - self.epsilon)
        return np.sum(errors) / len(y)

    def _calculate_gradient(self, X, y, predictions):
        errors = np.maximum(0, np.abs(y - predictions) - self.epsilon)
        gradient = np.zeros_like(self.weights)

        for i in range(len(X)):
            if errors[i] == 0:
                gradient += 0
            elif errors[i] < self.epsilon:
                gradient -= X[i] * (self.epsilon - errors[i])
            else:
                gradient -= X[i] * self.epsilon

        return gradient / len(X)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_epochs):
            predictions = np.dot(X, self.weights) + self.bias
            cost = self._cost_function(y, predictions)
            gradient = self._calculate_gradient(X, y, predictions)

            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * np.sum(gradient)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage:
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 3, 4, 5, 6])

# Create SVR instance
svr = SVR()

# Train the model
svr.fit(X_train, y_train)

# Make predictions
predictions = svr.predict(X_train)
print(predictions)
