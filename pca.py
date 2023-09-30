import numpy as np


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # covariance
        # row = 1 sample, columns = feature
        cov = np.cov(X.T)
        # eigenvectors, eigenvalues
        eigenvals, eigenvects = np.linalg.eig(cov)
        # v[:, 1]
        # sort eigenvectors
        eigenvects = eigenvects.T
        idxs = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idxs]
        eigenvects = eigenvects[idxs]

        # store first n eigenvectors
        self.components = eigenvects[:self.n_components]

    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)
        # return np.matmul(X, self.components.T)
