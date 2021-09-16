from solver import Solver
import numpy as np
from sklearn.base import BaseEstimator


class LinearSVC(BaseEstimator):
    def __init__(self, C=1, max_iter=1000, eps=0.00001, tol=0.0001) -> None:
        self.max_iter = max_iter
        self.eps = eps
        self.C = C
        self.tol = tol

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y).astype(float)
        # 二分类
        if len(np.unique(self.y)) > 2:
            print("LinearSVC only support binary classification")
        self.y[self.y != 1] = -1
        self.l, self.n_features = self.X.shape

        y = self.y.reshape(self.l, 1)
        self.Q = (y @ y.T) * (self.X @ self.X.T)  # Q矩阵
        del y

        s = Solver(
            self.l,
            self.Q,
            -np.ones(self.l),
            self.y,
            self.C,
            self.C,
            self.max_iter,
        )
        s.solve()
        alpha = s.alpha

        # 计算支持向量, 从而计算b
        self.sv = self.X[alpha != 0]
        self.w = alpha * self.y @ self.X
        self.b = np.mean(self.y[alpha != 0] - self.w @ self.sv.T)
        return self

    def predict(self, X):
        X = np.array(X).reshape(-1, self.n_features)  # (l * n_f)
        pred = self.w @ X.T + self.b
        pred[pred > 0] = 1
        pred[pred < 0] = 0
        return pred

    def score(self, X, y):
        X = np.array(X).reshape(-1, self.n_features)
        y = np.array(y).reshape(-1)
        pred = self.predict(X)
        return np.mean(pred == y)
