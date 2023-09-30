import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    dj_dw = 2 * self.lambda_param * self.w
                    self.w -= self.learning_rate * dj_dw

                else:
                    dj_dw = 2 * self.lambda_param * self.w - np.dot(x_i, y[idx])
                    dj_db = y_[idx]
                    self.w -= self.learning_rate * dj_dw
                    self.b -= self.learning_rate * dj_db

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
