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

def gen_binary_classification_data(m: int, n: int, k: int, random_state=1):
    """
    Generate a dataset of binary clssification
    args:
        m: number of observation
        n: number of features
        k: number of classes
    return:
        reuturn X, y, weights
        X: random generated m x n array, features 
        y: class of the observations by logistic regression
        weights: random generated coefficients which
            used to define y
    """
    def sigmoid(arr):
        return 1 / (1 + np.exp(-arr))
    
    np.random.seed(random_state)
    X = np.random.randint(-100, 100, size=(m,n))
    # add bias (intercept) component
    X = np.hstack((np.ones((m,1)), X))
    # wieght = num of features + bias
    weights = np.random.random(size=(n+1))
    logodds = X@weights
    prob = sigmoid(logodds)
    y = (prob > 0.5).astype(int)
    
    return X[:, 1:], y, weights 