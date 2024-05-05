import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_layers = len(hidden_sizes) + 1

        # Initialize weights and biases
        self.weights = [np.random.randn(self.input_size, hidden_sizes[0])]
        self.weights += [np.random.randn(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes) - 1)]
        self.weights.append(np.random.randn(hidden_sizes[-1], self.output_size))

        self.biases = [np.zeros((1, size)) for size in hidden_sizes]
        self.biases.append(np.zeros((1, self.output_size)))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward pass through the network
        self.outputs = [X]
        for i in range(self.num_layers):
            self.outputs.append(self.sigmoid(np.dot(self.outputs[-1], self.weights[i]) + self.biases[i]))

    def backward(self, X, y):
        # Backward pass through the network
        d_weights = [np.zeros_like(weight) for weight in self.weights]
        d_biases = [np.zeros_like(bias) for bias in self.biases]

        error = y - self.outputs[-1]
        d_output = error * self.sigmoid_derivative(self.outputs[-1])

        d_weights[-1] = np.dot(self.outputs[-2].T, d_output)
        d_biases[-1] = np.sum(d_output, axis=0, keepdims=True)

        for i in range(self.num_layers-2, -1, -1):
            error_hidden = np.dot(d_output, self.weights[i+1].T)
            d_hidden = error_hidden * self.sigmoid_derivative(self.outputs[i+1])

            d_weights[i] = np.dot(self.outputs[i].T, d_hidden)
            d_biases[i] = np.sum(d_hidden, axis=0, keepdims=True)
            d_output = d_hidden

        # Update weights and biases
        for i in range(self.num_layers):
            self.weights[i] += self.learning_rate * d_weights[i]
            self.biases[i] += self.learning_rate * d_biases[i]

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Forward and backward pass
            self.forward(X)
            self.backward(X, y)

            # Calculate and print loss
            loss = np.mean(np.square(y - self.outputs[-1]))
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

    def predict(self, X):
        # Forward pass to get predictions
        self.forward(X)
        return self.outputs[-1]

# Example usage:
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Define network architecture
input_size = 2
hidden_sizes = [3, 2]  # Number of units in each hidden layer
output_size = 1
learning_rate = 0.1
nn = NeuralNetwork(input_size, hidden_sizes, output_size, learning_rate)

# Train the network
epochs = 1000
nn.train(X_train, y_train, epochs)

# Make predictions
predictions = nn.predict(X_train)
print(predictions)

