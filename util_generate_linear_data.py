import numpy as np 

def gen_regression_data(m: int, n: int, random_status: int = 1):
    """
    generate data for linear regression model
    args:
        m: number of samples
        n: number of features
    """
    np.random.seed(random_status)
    X = np.random.randint(-100, 100, size=(m, n))
    X = np.hstack((np.ones((m, 1)), X))
    
    np.random.seed(random_status)
    weights = np.random.random(size=(X.shape[1]))
    weights = np.round(weights, 3)
    y = X@weights
    
    return X[:, 1:], y, weights

X, y, true_weights = gen_regression_data(100, 3)

# # validate
# print("first row of X: ", X[0, :])
# print("true weights: ", true_weights)
# print(np.hstack(([1], X[0, :]))@true_weights)
# print(y[0])
