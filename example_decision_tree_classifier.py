class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Index of feature to split on
        self.threshold = threshold  # Threshold value for splitting
        self.left = left            # Left subtree
        self.right = right          # Right subtree
        self.value = value          # Majority class in leaf node (for classification)

def calculate_gini_index(groups, classes):
    total_samples = sum(len(group) for group in groups)
    gini_index = 0.0
    for group in groups:
        group_size = len(group)
        if group_size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / group_size
            score += p * p
        gini_index += (1.0 - score) * (group_size / total_samples)
    return gini_index

def split_dataset(index, threshold, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < threshold:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = float('inf'), float('inf'), float('inf'), None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = split_dataset(index, row[index], dataset)
            gini_index = calculate_gini_index(groups, class_values)
            if gini_index < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini_index, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # Check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # Check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # Process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_best_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # Process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_best_split(right)
        split(node['right'], max_depth, min_size, depth+1)

def build_tree(train, max_depth, min_size):
    root = get_best_split(train)
    split(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = []
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions

# Example usage:
dataset = [
    [2.771244718,1.784783929,0],
    [1.728571309,1.169761413,0],
    [3.678319846,2.81281357,0],
    [3.961043357,2.61995032,0],
    [2.999208922,2.209014212,0],
    [7.497545867,3.162953546,1],
    [9.00220326,3.339047188,1],
    [7.444542326,0.476683375,1],
    [10.12493903,3.234550982,1],
    [6.642287351,3.319983761,1]
]
# Example of a train/test split
split_index = int(len(dataset) * 0.7)
train_set, test_set = dataset[:split_index], dataset[split_index:]
# Example of using the decision tree algorithm
max_depth = 3
min_size = 1
predictions = decision_tree(train_set, test_set, max_depth, min_size)
print(predictions)
