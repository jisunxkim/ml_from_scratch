import numpy as np 
import argparse    

class LogisticRegression():
    def __init__(self, n_iterations=10, learning_rate=0.0001):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.n_samples = 0
        self.n_features = 0
        self.w = None 
        self.b = None 
    
    def initialize(self):
        np.random.seed(1)
        self.w = np.random.randn(1, self.n_features) # (1, n)
        self.b = np.random.randn(1) # (1, )

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z)) # (1, m)

    def predict(self, X): # w (1, n), X (n, m)
        z = np.dot(self.w, X) + self.b # (1, m)
        pred = self.sigmoid(z) # (1, m)
        return pred 

    def get_loss(self, pred, y):
        # loss = -1/m * sum[ylog(pred) + (1-y)log(1-pred)]
        return - 1 / self.n_samples * \
            np.sum(y * np.log(pred) + (1-y)*np.log(1-pred))

    def get_gradients(self, X, y, a):
        # z = w*X + b
        # a = sigmoid(z) = 1 / (1 + exp(-z))
        # L = -yloga - (1-y)log(1-a)
        # dw = dL/dw = dL/da * da/dz * dz/dw
        # db = dL/db = dL/da * da/dz * dz/db 
        # da = dL/da = -y/a + (1-y)/(1-a)
        # da/dz = a(1-a)
        # dz/dw = X 
        # dz/db = 1
        da = -y/a + (1-y)/(1-a) # (1, m)
        da_dz = a*(1-a) # (1, m)
        dw = 1/self.n_samples * np.dot(da * da_dz, X.T) # (1,m)*(1, m), (n, m).T => (1, n) 
        db = 1/self.n_samples *np.sum(a * da_dz * 1) # (1)

        return dw, db 

    def train(self, X:np.ndarray, y:np.ndarray) -> None:
        self.n_samples, self.n_features = X.shape 
        X = X.T  
        y = y.reshape(1, -1)
        
        # initialize parameters 
        self.initialize()
        
        # initial prediciton and loss 
        pred = self.predict(X)
        loss = self.get_loss(pred, y)
        print(f"--- iteration 0 -- loss: {loss:0.4f}")

        for i in range(self.n_iterations):
            # gradients 
            dw, db = self.get_gradients(X, y, pred)
            # update parameters by the gradients 
            self.w = self.w - self.learning_rate * dw 
            self.b = self.b - self.learning_rate * db 
            # get prediciton and loss with the updated parameters 
            pred = self.predict(X)
            loss = self.get_loss(pred, y)
            print(f"--- iteration {i+1} -- loss: {loss:0.4f}")
        
        print("Completed training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iterations', help="number of iterations", 
                        default=10, type=int)
    parser.add_argument('--learning_rate', help="learning rate", 
                        default=0.0001, type=float)
    args = parser.parse_args()

    # Example usage:
    X_train = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7]
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    X_test = np.array([
        [1, 3],
        [5, 5]
    ])
    
    # Create Logistic Regression classifier instance
    log_reg = LogisticRegression(args.n_iterations, args.learning_rate)
    
    # Train the classifier
    log_reg.train(X_train, y_train)
    
    # Make predictions
    predictions = log_reg.predict(X_train.T)
    print(predictions)
    print(predictions > 0.5)

    predictions = log_reg.predict(X_test.T)
    print(predictions)
