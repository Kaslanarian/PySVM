from solver import Solver
import numpy as np
from rff import RFF
from sklearn.base import BaseEstimator


class OneClassSVM(BaseEstimator):
    def __init__(self,
                 nu=0.5,
                 kernel='rbf',
                 degree=3,
                 gamma='scale',
                 coef0=0,
                 max_iter=1000,
                 rff=False,
                 D=10000,
                 tol=1e-3,
                 verbose=False) -> None:
        super().__init__()
        self.nu = nu
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.kernel = kernel
        self.gamma = gamma
        self.rff = rff
        self.D = D
        self.degree = degree
        self.coef0 = coef0

    def fit(self, X):
        X = np.array(X)
        l, self.n_features = X.shape

        if type(self.gamma) == float:
            gamma = self.gamma
        else:
            gamma = {
                'scale': 1 / (self.n_features * X.std()),
                'auto': 1 / self.n_features,
            }[self.gamma]

        if self.rff:
            rff = RFF(gamma, self.D).fit(X)
            rbf_func = lambda x, y: rff.transform(x) @ rff.transform(y).T
        else:
            rbf_func = lambda x, y: np.exp(-gamma * (
                (x**2).sum(1).reshape(-1, 1) + (y**2).sum(1) - 2 * x @ y.T))

        degree = self.degree
        coef0 = self.coef0
        self.kernel_func = {
            "linear": lambda x, y: x @ y.T,
            "poly": lambda x, y: (gamma * x @ y.T + coef0)**degree,
            "rbf": rbf_func,
            "sigmoid": lambda x, y: np.tanh(gamma * (x @ y.T) + coef0)
        }[self.kernel]

        Q = self.kernel_func(X, X)

        n = int(self.nu * l)
        alpha = np.ones(l)
        if n < l:
            alpha[n] = self.nu * l - n
        for i in range(n + 1, l):
            alpha[i] = 0

        s = Solver(
            l,
            Q,
            np.zeros(l),
            np.ones(l),
            alpha,
            1.,
            1.,
            self.tol,
            max_iter=self.max_iter,
        )
        s.solve(verbose=self.verbose)
        alpha = s.get_alpha()
        rho = s.get_rho()
        self.decision_function = lambda x: alpha @ self.kernel_func(
            X,
            x,
        ) - rho
        return self

    def predict(self, X):
        X = np.array(X).reshape(-1, self.n_features)
        pred = self.decision_function(X)
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        return pred
