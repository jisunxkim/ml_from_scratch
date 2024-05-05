import numpy as np
import argparse

class LinearRegression():

    def __init__(self):
        # m: number of observations, n: number of features 
        self.w = None 
        self.b = None 
        self.m = None  
        self.n = None 
        self.loss_history = []

    def _initialize(self):
        np.random.seed(1)
        self.w = np.random.randn(1, self.n)
        self.b = np.random.randn(1)
    
    def predict(self, X):
        return np.dot(self.w, X)

    def loss(self, pred:np.ndarray, y:np.ndarray) -> float:
        # loss = 1/2 * (pred - actual)**2
        loss_val = 1/2 * np.sum((pred - y)**2)
        return loss_val

    def gradient(self, X, y, pred):
        diff = (pred - y).reshape(1, -1)

        dw = 1/self.m * np.sum(diff*X, axis=1, keepdims=True)
        db = 1/self.m * np.sum(pred, axis=1, keepdims=True)
        return dw, db

    def train(self,  X:np.ndarray, y:np.ndarray, n_iter:int, learning_rate:float = 0.00001) -> None:
        """
        Args:
            X: (n, m); n=number of features, m = number of observations
        """
        self.n, self.m = X.shape
        self._initialize()
        # initial prediction and loss
        pred = self.predict(X)
        loss_val = self.loss(pred, y)
        self.loss_history.append(loss_val)
        print(f"--- iteration 0: loss{loss_val:0.3f}")

        for i in range(n_iter):
            # gradient
            dw, db = self.gradient(X, y, pred)
            # update weights 
            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db 
            # updated prediction and loss 
            pred = self.predict(X)
            loss_val = self.loss(pred, y)
            
            print(f"--- iteration {i+1}: loss{loss_val:0.3f}")
        
        print("completed training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_itter', help="number of iterations", default=10, type=int)
    parser.add_argument('--learning_rate', help="learning rate", default=0.00001, type=float)
    
    args = parser.parse_args()

    # Example usage:
    X_train = np.array([[1], [2], [3], [4], [5]])
    X_train = X_train.T
    y_train = np.array([2, 4, 6, 8, 10])
    y_train = y_train.reshape(1, -1)
    # Create Linear Regression instance
    linear_reg = LinearRegression()
    
    # Train the model
    linear_reg.train(X_train, y_train, n_iter=args.n_itter, learning_rate=args.learning_rate)
    
    # Make predictions
    predictions = linear_reg.predict(X_train)
    print(predictions)
