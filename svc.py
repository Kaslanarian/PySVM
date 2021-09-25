from solver import Solver, NuSolver
import numpy as np
from sklearn.base import BaseEstimator


class LinearSVC(BaseEstimator):
    def __init__(self, C=1, max_iter=1000, tol=1e-3, verbose=False) -> None:
        self.max_iter = max_iter
        self.C = C
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y):
        X, y = np.array(X), np.array(y, dtype=float)
        if len(np.unique(y)) > 2:
            print("LinearSVC only support binary classification")
            return self

        y[y != 1] = -1
        l, self.n_features = X.shape

        y = y.reshape(l, 1)
        Q = (y @ y.T) * (X @ X.T)  # Q矩阵
        y = y.reshape(-1)

        s = Solver(l=l,
                   Q=Q,
                   p=-np.ones(l),
                   y=y,
                   alpha=np.zeros(l),
                   Cp=self.C,
                   Cn=self.C,
                   max_iter=self.max_iter,
                   eps=self.tol)
        s.solve(self.verbose)
        w = s.get_alpha() * y @ X
        b = s.get_b()
        self.decision_function = lambda x: w @ x.T + b
        return self

    def predict(self, X):
        X = np.array(X).reshape(-1, self.n_features)  # (l * n_f)
        pred = self.decision_function(X)
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
                 tol=1e-3,
                 verbose=False) -> None:
        super().__init__()
        self.C = C
        self.max_iter = 1000
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y):
        X, y = np.array(X), np.array(y, dtype=float)
        if len(np.unique(y)) > 2:
            print("KernelSVC only support binary classification")
            return self

        y[y != 1] = -1
        l, self.n_features = X.shape

        # 注册核函数相关
        if type(self.gamma) == float:
            gamma = self.gamma
        else:
            gamma = {
                'scale': 1 / (self.n_features * X.std()),
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
        y = y.reshape(l, 1)
        Q = (y @ y.T) * self.kernel_func(X, X)
        y = y.reshape(-1)

        s = Solver(
            l=l,
            Q=Q,
            p=-np.ones(l),
            y=y,
            alpha=np.zeros(l),
            Cp=self.C,
            Cn=self.C,
            max_iter=self.max_iter,
            eps=self.tol,
        )
        s.solve(verbose=self.verbose)
        alpha = s.get_alpha()
        b = s.get_b()
        self.decision_function = lambda x: alpha * y @ self.kernel_func(
            X,
            x,
        ) + b
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


class NuSVC(KernelSVC):
    def __init__(
        self,
        nu=0.5,
        max_iter=1000,
        kernel='rbf',
        degree=3,
        gamma='scale',
        coef0=0,
        tol=1e-3,
        verbose=False,
    ) -> None:
        self.max_iter = max_iter
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y):
        X, y = np.array(X), np.array(y, dtype=float)
        if len(np.unique(y)) > 2:
            print("NuSVC only support binary classification")
            return self

        y[y != 1] = -1
        l, self.n_features = X.shape

        if type(self.gamma) == float:
            gamma = self.gamma
        else:
            gamma = {
                'scale': 1 / (self.n_features * X.std()),
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

        y = y.reshape(l, 1)
        Q = (y @ y.T) * self.kernel_func(X, X)  # Q矩阵
        y = y.reshape(-1)

        # 计算alpha初始值
        sum_pos = self.nu * l / 2
        sum_neg = self.nu * l / 2
        alpha = np.zeros(l)

        for i in range(l):
            if y[i] == +1:
                alpha[i] = min(1., sum_pos)
                sum_pos -= alpha[i]
            else:
                alpha[i] = min(1., sum_neg)
                sum_neg -= alpha[i]

        s = NuSolver(
            l=l,
            Q=Q,
            p=np.zeros(l),
            y=y,
            alpha=alpha,
            Cp=1,
            Cn=1,
            eps=self.tol,
            max_iter=self.max_iter,
        )
        s.solve(verbose=self.verbose)
        rho = s.get_rho()
        alpha = s.get_alpha() / rho
        b = s.get_b() / rho
        self.decision_function = lambda x: alpha * y @ self.kernel_func(
            X,
            x
        ) + b
        return self

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        return super().score(X, y)