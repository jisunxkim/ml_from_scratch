class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def _calculate_variance(self, y):
        mean = sum(y) / len(y)
        variance = sum((yi - mean) ** 2 for yi in y) / len(y)
        return variance

    def _split_dataset(self, X, y, feature_index, threshold):
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][feature_index] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        return left_X, left_y, right_X, right_y

    def _find_best_split(self, X, y):
        best_feature_index, best_threshold, best_variance_reduction = None, None, 0
        initial_variance = self._calculate_variance(y)
        n_features = len(X[0])

        for feature_index in range(n_features):
            feature_values = set(X[i][feature_index] for i in range(len(X)))

            for threshold in feature_values:
                left_X, left_y, right_X, right_y = self._split_dataset(X, y, feature_index, threshold)
                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                total_samples = len(left_y) + len(right_y)
                left_weight = len(left_y) / total_samples
                right_weight = len(right_y) / total_samples
                variance_reduction = initial_variance - (left_weight * self._calculate_variance(left_y) + right_weight * self._calculate_variance(right_y))

                if variance_reduction > best_variance_reduction:
                    best_feature_index = feature_index
                    best_threshold = threshold
                    best_variance_reduction = variance_reduction

        return best_feature_index, best_threshold

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1 or depth == self.max_depth:
            return {'prediction': sum(y) / len(y)}

        best_feature_index, best_threshold = self._find_best_split(X, y)
        if best_feature_index is None:
            return {'prediction': sum(y) / len(y)}

        left_X, left_y, right_X, right_y = self._split_dataset(X, y, best_feature_index, best_threshold)

        left_subtree = self._build_tree(left_X, left_y, depth + 1)
        right_subtree = self._build_tree(right_X, right_y, depth + 1)

        return {'feature_index': best_feature_index, 'threshold': best_threshold,
                'left': left_subtree, 'right': right_subtree}

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)

    def _predict_sample(self, sample, tree):
        if 'prediction' in tree:
            return tree['prediction']
        if sample[tree['feature_index']] <= tree['threshold']:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])

    def predict(self, X):
        predictions = []
        for sample in X:
            predictions.append(self._predict_sample(sample, self.tree))
        return predictions

# Example usage:
X_train = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y_train = [2, 3, 4, 5, 6]

# Create Decision Tree Regressor instance
dt_regressor = DecisionTreeRegressor(max_depth=2)

# Train the model
dt_regressor.fit(X_train, y_train)

# Make predictions
predictions = dt_regressor.predict(X_train)
print(predictions)
