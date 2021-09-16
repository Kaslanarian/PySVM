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
        self.X, self.y = np.array(X), np.array(y, dtype=float)
        if len(np.unique(self.y)) > 2:
            print("LinearSVC only support binary classification")

        self.y[self.y != 1] = -1
        self.l, self.n_features = self.X.shape

        y = self.y.reshape(self.l, 1)
        Q = (y @ y.T) * (self.X @ self.X.T)  # Q矩阵
        del y

        s = Solver(
            l=self.l,
            Q=Q,
            p=-np.ones(self.l),
            y=self.y,
            alpha=np.zeros(self.l),
            Cp=self.C,
            Cn=self.C,
            max_iter=self.max_iter,
        )
        s.solve()
        alpha = s.get_alpha()

        # 计算支持向量, 从而计算b
        self.sv = self.X[alpha != 0]
        self.w = alpha * self.y @ self.X
        if len(self.sv) > 0:
            self.b = np.mean(self.y[alpha != 0] - self.w @ self.sv.T)
        else:
            print("no sv")
            ub_id = np.logical_or(
                np.logical_and(alpha == 0, y == -1),
                np.logical_and(alpha == self.C, y == 1),
            )
            lb_id = np.logical_or(
                np.logical_and(alpha == 0, y == 1),
                np.logical_and(alpha == self.C, y == -1),
            )
            grad = Q @ alpha - 1
            b = (np.max((y * grad)[lb_id]) + np.min((y * grad)[ub_id])) / 2
        return self

    def predict(self, X):
        X = np.array(X).reshape(-1, self.n_features)  # (l * n_f)
        pred = self.w @ X.T + self.b
        pred[pred > 0] = 1
        pred[pred < 0] = 0
        return pred.astype('int')

    def score(self, X, y):
        X = np.array(X).reshape(-1, self.n_features)
        y = np.array(y).reshape(-1)
        pred = self.predict(X)
        return np.mean(pred == y)


class KernelSVC(BaseEstimator):
    def __init__(self,
                 C=1,
                 max_iter=1000,
                 kernel='rbf',
                 degree=3,
                 gamma='scale',
                 coef0=0,
                 tol=1e-3) -> None:
        super().__init__()
        self.C = C
        self.max_iter = 1000
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        self.X, self.y = np.array(X), np.array(y, dtype=float)
        if len(np.unique(self.y)) > 2:
            print("LinearSVC only support binary classification")

        self.y[self.y != 1] = -1
        self.l, self.n_features = self.X.shape

        # 注册核函数相关
        if type(self.gamma) == float:
            gamma = self.gamma
        else:
            gamma = {
                'scale': 1 / (self.n_features * self.X.std()),
                'auto': 1 / self.n_features,
            }[self.gamma]
        degree = self.degree
        coef0 = self.coef0
        self.kernel_func = {
            "linear":
            lambda x, y: x @ y.T,
            "poly":
            lambda x, y: (gamma * x @ y.T + coef0)**degree,
            "rbf":
            lambda x, y: np.exp(-gamma * np.linalg.norm(
                np.expand_dims(x, axis=-1) - y.T, axis=1)**2),
            "sigmoid":
            lambda x, y: np.tanh(gamma * (x @ y.T) + coef0)
        }[self.kernel]

        # 计算Q
        y = self.y.reshape(self.l, 1)
        Q = (y @ y.T) * self.kernel_func(self.X, self.X)
        del y

        s = Solver(
            l=self.l,
            Q=Q,
            p=-np.ones(self.l),
            y=self.y,
            alpha=np.zeros(self.l),
            Cp=self.C,
            Cn=self.C,
            max_iter=self.max_iter,
        )
        s.solve()
        alpha = s.get_alpha()

        self.sv = self.X[alpha != 0]
        if len(self.sv) > 0:
            b = np.mean(self.y[alpha != 0] -
                        alpha * self.y @ self.kernel_func(self.X, self.sv))
        else:
            print("no sv")
            ub_id = np.logical_or(
                np.logical_and(alpha == 0, y == -1),
                np.logical_and(alpha == self.C, y == 1),
            )
            lb_id = np.logical_or(
                np.logical_and(alpha == 0, y == 1),
                np.logical_and(alpha == self.C, y == -1),
            )
            grad = Q @ alpha - 1
            b = (np.max((y * grad)[lb_id]) + np.min((y * grad)[ub_id])) / 2

        self.decision_function = lambda x: alpha * self.y @ self.kernel_func(
            self.X, x) + b

        return self

    def predict(self, X):
        X = np.array(X).reshape(-1, self.n_features)
        pred = self.decision_function(X)
        pred[pred > 0] = 1
        pred[pred < 0] = 0
        return pred.astype('int')

    def score(self, X, y):
        X = np.array(X).reshape(-1, self.n_features)
        y = np.array(y).reshape(-1)
        pred = self.predict(X)
        return np.mean(pred == y)
