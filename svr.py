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
        s = Solver(
            l=2 * l,
            Q=Q,
            p=p,
            y=y.reshape(-1),
            alpha=np.zeros(2 * l),
            Cp=C,
            Cn=C,
            max_iter=self.max_iter,
        )
        s.solve()
        alpha2 = s.get_alpha()
        alpha = alpha2[:l]
        alpha_star = alpha2[l:]
        self.w = (alpha_star - alpha) @ X
        is_sv = np.logical_and(alpha_star > 0, alpha_star < C)
        self.b = np.mean(z[is_sv] - self.w @ X[is_sv].T) + self.epsilon

        return self

    def predict(self, x):
        x = np.array(x).reshape(-1, self.n_features)
        return self.w @ x.T + self.b

    def score(self, test_X, test_y):
        X = np.array(test_X).reshape(-1, self.n_features)
        y = np.array(test_y).reshape(-1)
        pred = self.predict(test_X)
        return -np.mean((pred - y)**2)


class KernelSVR(BaseEstimator):
    def __init__(self,
                 C=1,
                 epsilon=0,
                 max_iter=1000,
                 kernel='rbf',
                 degree=3,
                 gamma='scale',
                 coef0=0,
                 tol=1e-3) -> None:
        super().__init__()
        self.C = C
        self.epsilon = epsilon
        self.max_iter = 1000
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        X, z = np.array(X), np.array(y, dtype=float)
        l, self.n_features = X.shape
        C = self.C

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

        # 计算Q：
        Q0 = self.kernel_func(X, X)
        Q = np.hstack((Q0, -Q0))
        Q = np.vstack((Q, -Q))

        p = self.epsilon + np.hstack((z, -z))  # 2l
        y = np.hstack((np.ones(l), -np.ones(l)))
        s = Solver(2 * l, Q, p, y, np.zeros(2 * l), C, C, self.max_iter)
        s.solve()
        alpha2 = s.get_alpha()
        alpha = alpha2[:l]
        alpha_star = alpha2[l:]
        alpha_diff = alpha_star - alpha
        is_sv = np.logical_and(alpha_star > 0, alpha_star < self.C)
        if np.sum(is_sv) > 0:
            b = np.mean(z[is_sv] - alpha_diff @ self.kernel_func(X, X[is_sv])
                        ) + self.epsilon
        else:
            print("no sv")
            ub_id = np.logical_or(
                np.logical_and(alpha2 == 0, y == -1),
                np.logical_and(alpha2 == C, y == 1),
            )
            lb_id = np.logical_or(
                np.logical_and(alpha2 == 0, y == 1),
                np.logical_and(alpha2 == C, y == -1),
            )
            grad = Q @ alpha2 + p
            b = (np.max((y * grad)[lb_id]) + np.min((y * grad)[ub_id])) / 2

        self.decision_function = lambda x: alpha_diff @ self.kernel_func(
            X,
            x,
        ) + b

        return self

    def predict(self, X):
        X = np.array(X).reshape(-1, self.n_features)
        pred = self.decision_function(X)
        return pred

    def score(self, X, y):
        X = np.array(X).reshape(-1, self.n_features)
        y = np.array(y).reshape(-1)
        pred = self.predict(X)
        return -np.mean((pred - y)**2)
