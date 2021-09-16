from solver import Solver
import numpy as np
from sklearn.base import BaseEstimator


class LinearSVR(BaseEstimator):
    def __init__(self, C=1, max_iter=1000, epsilon=0, tol=0.0001) -> None:
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.C = C
        self.tol = tol

    def fit(self, X, y):
        X, z = np.array(X), np.array(y)
        l, self.n_features = X.shape
        C = self.C
        # 计算Q矩阵（2l*2l）
        X_stack = np.vstack((X, X))
        y = np.vstack((np.ones((l, 1)), -np.ones((l, 1))))
        Q = (y @ y.T) * (X_stack @ X_stack.T)

        p = self.epsilon + np.hstack((z, -z))  # 2l
        s = Solver(2 * l, Q, p, y.reshape(-1), C, C, self.max_iter)
        s.solve()
        alpha2 = s.get_alpha()
        alpha = alpha2[:l]
        alpha_star = alpha2[l:]
        self.w = (alpha_star - alpha) @ X
        is_sv = np.logical_and(alpha_star > 0, alpha_star < self.C)
        self.b = np.mean(z[is_sv] - self.w @ X[is_sv].T) + self.epsilon
        return self

    def predict(self, x):
        x = np.array(x).reshape(-1, self.n_features)
        return self.w @ x.T + self.b

    def score(self, test_X, test_y):
        X = np.array(test_X).reshape(-1, self.n_features)
        y = np.array(test_y).reshape(-1)
        pred = self.predict(X)
        return -np.mean((pred - y)**2)