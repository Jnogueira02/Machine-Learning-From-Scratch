import numpy as np
from linear_regression import BaseRegression


class LogisticRegression(BaseRegression):

    def _approximation(self, X, w, b):
        z = np.dot(X, w) + b
        return self._sigmoid(z)

    def _predict(self, X, w, b):
        z = np.dot(X, w) + b
        y_predicted = self._sigmoid(z)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
