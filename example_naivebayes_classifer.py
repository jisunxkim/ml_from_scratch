import math

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}

    def fit(self, X_train, y_train):
        n_samples = len(X_train)
        n_features = len(X_train[0])
        
        # Calculate class probabilities
        self.class_probs = {}
        for label in y_train:
            if label not in self.class_probs:
                self.class_probs[label] = 1
            else:
                self.class_probs[label] += 1
        for label in self.class_probs:
            self.class_probs[label] /= n_samples

        # Calculate feature probabilities
        self.feature_probs = {}
        for label in self.class_probs:
            self.feature_probs[label] = {}
            for i in range(n_features):
                self.feature_probs[label][i] = {}
                for value in set(row[i] for row, y in zip(X_train, y_train) if y == label):
                    count = sum(1 for row, y in zip(X_train, y_train) if row[i] == value and y == label)
                    self.feature_probs[label][i][value] = count / self.class_probs[label]

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            max_prob = -1
            predicted_class = None
            for label in self.class_probs:
                class_prob = self.class_probs[label]
                for i, value in enumerate(x):
                    if value in self.feature_probs[label][i]:
                        class_prob *= self.feature_probs[label][i][value]
                    else:
                        class_prob = 0
                if class_prob > max_prob:
                    max_prob = class_prob
                    predicted_class = label
            predictions.append(predicted_class)
        return predictions

# Example usage:
X_train = [
    [1, 'S'],
    [1, 'M'],
    [1, 'M'],
    [1, 'S'],
    [1, 'S'],
    [2, 'S'],
    [2, 'M'],
    [2, 'M'],
    [2, 'L'],
    [2, 'L'],
    [3, 'L'],
    [3, 'M'],
    [3, 'M'],
    [3, 'L'],
    [3, 'L']
]

y_train = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]

X_test = [
    [2, 'S'],
    [3, 'M'],
    [3, 'S']
]

# Create Naive Bayes classifier instance
nb_classifier = NaiveBayesClassifier()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Make predictions
predictions = nb_classifier.predict(X_test)
print(predictions)
