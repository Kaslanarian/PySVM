import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from .rff import NormalRFF
from .solver import Solver, SolverWithCache, NuSolver, NuSolverWithCache


class _LinearSVC(BaseEstimator):
    def __init__(self, C=1, max_iter=1000, tol=1e-5, cache_size=256) -> None:
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.cache_size = cache_size

    def fit(self, X, y):
        X, y = np.array(X), np.array(y, dtype=float)
        y[y != 1] = -1
        l, self.n_features = X.shape
        p = -np.ones(l)

        w = np.zeros(self.n_features)
        if self.cache_size == 0:
            Q = y.reshape(-1, 1) * y * np.matmul(X, X.T)
            solver = Solver(Q, p, y, self.C, self.tol)
        else:
            solver = SolverWithCache(p, y, self.C, self.tol, self.cache_size)

        def func(i):
            return y * (X @ X[i]) * y[i]

        for n_iter in range(self.max_iter):
            i, j = solver.working_set_select()
            if i < 0:
                break

            delta_i, delta_j = solver.update(i, j, func)
            w += delta_i * y[i] * X[i] + delta_j * y[j] * X[j]
        else:
            print("LinearSVC not coverage with {} iterations".format(
                self.max_iter))

        self.coef_ = (w, solver.calculate_rho())
        return self

    def decision_function(self, X):
        return self.coef_[0] @ np.array(X).T - self.coef_[-1]

    def predict(self, X):
        return (self.decision_function(np.array(X)) >= 0).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class LinearSVC(_LinearSVC):
    def __init__(
        self,
        C=1,
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovr",
        n_jobs=None,
    ) -> None:
        super().__init__(C, max_iter, tol, cache_size)
        self.multiclass = multiclass
        self.n_jobs = n_jobs
        params = {
            "estimator": _LinearSVC(C, max_iter, tol, cache_size),
            "n_jobs": n_jobs,
        }
        self.multiclass_model: OneVsOneClassifier = {
            "ovo": OneVsOneClassifier(**params),
            "ovr": OneVsRestClassifier(**params),
        }[multiclass]

    def fit(self, X, y):
        self.multiclass_model.fit(X, y)
        return self

    def decision_function(self, X):
        return self.multiclass_model.decision_function(X)

    def predict(self, X):
        return self.multiclass_model.predict(X)

    def score(self, X, y):
        return self.multiclass_model.score(X, y)


class LinearSVR(_LinearSVC):
    def __init__(
        self,
        C=1,
        eps=0,
        max_iter=1000,
        tol=1e-5,
        cache_size=256,
    ) -> None:
        super().__init__(C, max_iter, tol, cache_size)
        self.eps = eps

    def fit(self, X, y):
        X, z = np.array(X), np.array(y)
        l, self.n_features = X.shape

        y = np.empty(2 * l)
        y[:l], y[l:] = 1., -1.

        p = np.ones(2 * l) * self.eps
        p[:l] -= z
        p[l:] += z

        w = np.zeros(self.n_features)

        if self.cache_size == 0:
            Q = np.matmul(X, X.T)
            Q2 = np.hstack((Q, -Q))
            Q4 = np.vstack((Q2, -Q2))
            solver = Solver(Q4, p, y, self.C, self.tol)
        else:
            solver = SolverWithCache(p, y, self.C, self.tol, self.cache_size)

        def func(i):
            if i < l:
                Qi = np.matmul(X, X[i])
            else:
                Qi = -np.matmul(X, X[i - l])
            return np.hstack((Qi, -Qi))

        for n_iter in range(self.max_iter):
            i, j = solver.working_set_select()
            if i < 0:
                break

            delta_i, delta_j = solver.update(i, j, func)
            w += (delta_i * y[i] * X[i if i < l else i - l] +
                  delta_j * y[j] * X[j if j < l else j - l])
        else:
            print("LinearSVR not coverage with {} iterations".format(
                self.max_iter))

        self.coef_ = (w, solver.calculate_rho())
        return self

    def decision_function(self, X):
        return super().decision_function(X)

    def predict(self, X):
        return self.decision_function(X)

    def score(self, X, y):
        return r2_score(y, self.predict(X))


class _KernelSVC(_LinearSVC):
    def __init__(
        self,
        C=1,
        kernel='rbf',
        degree=3,
        gamma='scale',
        coef0=0,
        max_iter=1000,
        rff=False,
        D=1000,
        tol=1e-5,
        cache_size=256,
    ) -> None:
        super().__init__(C, max_iter, tol, cache_size)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.rff = rff
        self.D = D

    def register_kernel(self, std):
        if type(self.gamma) == str:
            gamma = {
                'scale': 1 / (self.n_features * std),
                'auto': 1 / self.n_features,
            }[self.gamma]
        else:
            gamma = self.gamma

        if self.rff:
            rff = NormalRFF(gamma * 2,
                            self.D).fit(np.ones((1, self.n_features)))
            rbf_func = lambda x, y: rff.transform(x) @ rff.transform(y).T
        else:
            rbf_func = lambda x, y: np.exp(-gamma * (
                (x**2).sum(1).reshape(-1, 1) + (y**2).sum(1) - 2 * x @ y.T))

        degree = self.degree
        coef0 = self.coef0
        return {
            "linear": lambda x, y: np.matmul(x, y.T),
            "poly": lambda x, y: (gamma * np.matmul(x, y.T) + coef0)**degree,
            "rbf": rbf_func,
            "sigmoid": lambda x, y: np.tanh(gamma * np.matmul(x, y.T) + coef0)
        }[self.kernel]

    def fit(self, X, y):
        X, y = np.array(X), np.array(y, dtype=float)
        y[y != 1] = -1
        l, self.n_features = X.shape
        p = -np.ones(l)

        kernel_func = self.register_kernel(X.std())

        if self.cache_size == 0:
            Q = y.reshape(-1, 1) * y * kernel_func(X, X)
            solver = Solver(Q, p, y, self.C, self.tol)
        else:
            solver = SolverWithCache(p, y, self.C, self.tol, self.cache_size)

        def func(i):
            return y * kernel_func(X, X[i:i + 1]).reshape(-1) * y[i]

        for n_iter in range(self.max_iter):
            i, j = solver.working_set_select()
            if i < 0:
                break
            solver.update(i, j, func)
        else:
            print("KernelSVC not coverage with {} iterations".format(
                self.max_iter))

        self.decision_function = lambda x: np.matmul(
            solver.alpha * y,
            kernel_func(X, x),
        ) - solver.calculate_rho()
        return self


class KernelSVC(LinearSVC, _KernelSVC):
    def __init__(
        self,
        C=1,
        kernel='rbf',
        degree=3,
        gamma='scale',
        coef0=0,
        max_iter=1000,
        rff=False,
        D=1000,
        tol=1e-5,
        cache_size=256,
        multiclass="ovr",
        n_jobs=None,
    ) -> None:
        super().__init__(C, max_iter, tol, cache_size)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.rff = rff
        self.D = D
        params = {
            "estimator":
            _KernelSVC(C, kernel, degree, gamma, coef0, max_iter, rff, D, tol,
                       cache_size),
            "n_jobs":
            n_jobs,
        }
        self.multiclass_model = {
            "ovo": OneVsOneClassifier(**params),
            "ovr": OneVsRestClassifier(**params),
        }[multiclass]

    def decision_function(self, X):
        return super().decision_function(X)

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        return super().score(X, y)


class KernelSVR(_KernelSVC):
    def __init__(self,
                 C=1,
                 eps=0,
                 kernel='rbf',
                 degree=3,
                 gamma='scale',
                 coef0=0,
                 max_iter=1000,
                 rff=False,
                 D=1000,
                 tol=1e-5,
                 cache_size=256) -> None:
        super().__init__(C, kernel, degree, gamma, coef0, max_iter, rff, D,
                         tol, cache_size)
        self.eps = eps

    def fit(self, X, y):
        X, z = np.array(X), np.array(y)
        l, self.n_features = X.shape

        y = np.empty(2 * l)
        y[:l], y[l:] = 1., -1.

        p = np.ones(2 * l) * self.eps
        p[:l] -= z
        p[l:] += z

        kernel_func = self.register_kernel(X.std())

        if self.cache_size == 0:
            Q = kernel_func(X, X)
            Q2 = np.hstack((Q, -Q))
            Q4 = np.vstack((Q2, -Q2))
            solver = Solver(Q4, p, y, self.C, self.tol)
        else:
            solver = SolverWithCache(p, y, self.C, self.tol, self.cache_size)

        def func(i):
            if i < l:
                Qi = kernel_func(X, X[i:i + 1]).reshape(-1)
            else:
                Qi = -kernel_func(X, X[i - l:i - l + 1]).reshape(-1)
            return np.hstack((Qi, -Qi))

        for n_iter in range(self.max_iter):
            i, j = solver.working_set_select()
            if i < 0:
                break

            solver.update(i, j, func)
        else:
            print("KernelSVR not coverage with {} iterations".format(
                self.max_iter))

        self.decision_function = lambda x: np.matmul(
            solver.alpha[:l] - solver.alpha[l:],
            kernel_func(X, x),
        ) - solver.calculate_rho()

        return self

    def predict(self, X):
        return self.decision_function(np.array(X))

    def score(self, X, y):
        return r2_score(y, self.predict(X))


class _NuSVC(_KernelSVC):
    def __init__(self,
                 nu=0.5,
                 kernel='rbf',
                 degree=3,
                 gamma='scale',
                 coef0=0,
                 max_iter=1000,
                 rff=False,
                 D=1000,
                 tol=1e-5,
                 cache_size=256) -> None:
        super().__init__(1, kernel, degree, gamma, coef0, max_iter, rff, D,
                         tol, cache_size)
        self.nu = nu

    def fit(self, X, y):
        X, y = np.array(X), np.array(y, dtype=float)
        y[y != 1] = -1
        l, self.n_features = X.shape
        p = np.zeros(l)

        kernel_func = self.register_kernel(X.std())

        def func(i):
            return y * kernel_func(X, X[i:i + 1]).reshape(-1) * y[i]

        if self.cache_size == 0:
            Q = y.reshape(-1, 1) * y * kernel_func(X, X)
            solver = NuSolver(Q, p, y, self.nu * l, self.C, self.tol)
        else:
            solver = NuSolverWithCache(p, y, self.nu * l, self.C, func,
                                       self.tol, self.cache_size)

        for n_iter in range(self.max_iter):
            i, j, Qi, Qj = solver.working_set_select(func)
            if i < 0:
                break
            solver.update(i, j, Qi, Qj)
        else:
            print("NuSVC not coverage with {} iterations".format(
                self.max_iter))

        rho, b = solver.calculate_rho_b()
        self.decision_function = lambda x: np.matmul(
            solver.alpha * y,
            kernel_func(X, x),
        ) / rho + b / rho
        return self

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        return super().score(X, y)


class NuSVC(KernelSVC, _NuSVC):
    def __init__(self,
                 nu=0.5,
                 kernel='rbf',
                 degree=3,
                 gamma='scale',
                 coef0=0,
                 max_iter=1000,
                 rff=False,
                 D=1000,
                 tol=1e-5,
                 cache_size=256,
                 multiclass="ovr",
                 n_jobs=None) -> None:
        super().__init__(1, kernel, degree, gamma, coef0, max_iter, rff, D,
                         tol, cache_size, multiclass, n_jobs)
        self.nu = nu
        params = {
            "estimator":
            _NuSVC(nu, kernel, degree, gamma, coef0, max_iter, rff, D, tol,
                   cache_size),
            "n_jobs":
            n_jobs,
        }
        self.multiclass_model: OneVsOneClassifier = {
            "ovo": OneVsOneClassifier(**params),
            "ovr": OneVsRestClassifier(**params),
        }[multiclass]

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        return super().score(X, y)


class NuSVR(KernelSVR):
    def __init__(self,
                 C=1,
                 nu=0.5,
                 kernel='rbf',
                 degree=3,
                 gamma='scale',
                 coef0=0,
                 max_iter=1000,
                 rff=False,
                 D=1000,
                 tol=1e-5,
                 cache_size=256) -> None:
        super().__init__(C, 0, kernel, degree, gamma, coef0, max_iter, rff, D,
                         tol, cache_size)
        self.nu = nu

    def fit(self, X, y):
        X, z = np.array(X), np.array(y)
        l, self.n_features = X.shape

        y = np.empty(2 * l)
        y[:l], y[l:] = 1, -1

        p = np.empty(2 * l)
        p[:l], p[l:] = -z, z

        kernel_func = self.register_kernel(X.std())

        def func(i):
            if i < l:
                Qi = kernel_func(X, X[i:i + 1]).reshape(-1)
            else:
                Qi = -kernel_func(X, X[i - l:i - l + 1]).reshape(-1)
            return np.hstack((Qi, -Qi))

        if self.cache_size == 0:
            Q = kernel_func(X, X)
            Q2 = np.hstack((Q, -Q))
            Q4 = np.vstack((Q2, -Q2))
            solver = NuSolver(Q4, p, y, self.C * l * self.nu, self.C, self.tol)
        else:
            solver = NuSolverWithCache(p, y, self.C * l * self.nu, self.C,
                                       func, self.tol, self.cache_size)
        

        for n_iter in range(self.max_iter):
            i, j, Qi, Qj = solver.working_set_select(func)
            if i < 0:
                break

            solver.update(i, j, Qi, Qj)
        else:
            print("NuSVR not coverage with {} iterations".format(
                self.max_iter))

        rho, b = solver.calculate_rho_b()
        self.decision_function = lambda x: np.matmul(
            solver.alpha[:l] - solver.alpha[l:],
            kernel_func(X, x),
        ) + b
        return self

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        return super().score(X, y)


class OneClassSVM(_NuSVC):
    def __init__(
        self,
        nu=0.5,
        kernel='rbf',
        degree=3,
        gamma='scale',
        coef0=0,
        max_iter=1000,
        rff=False,
        D=1000,
        tol=1e-5,
        cache_size=256,
    ) -> None:
        super().__init__(nu, kernel, degree, gamma, coef0, max_iter, rff, D,
                         tol, cache_size)

    def fit(self, X):
        X = np.array(X)
        l, self.n_features = X.shape

        kernel_func = self.register_kernel(X.std())
        p = np.zeros(l)
        y = np.ones(l)

        def func(i):
            return kernel_func(X, X[i:i + 1]).reshape(-1)

        # init
        alpha = np.ones(l)
        n = int(self.nu * l)
        for i in range(n):
            alpha[i] = 1
        if n < l:
            alpha[i] = self.nu * l - n
        for i in range(n + 1, l):
            alpha[i] = 0

        if self.cache_size == 0:
            Q = kernel_func(X, X)
            solver = Solver(Q, p, y, 1, self.tol)
            solver.alpha = alpha
            solver.neg_y_grad = -y * (Q @ solver.alpha)

        else:
            solver = SolverWithCache(p, y, 1, self.tol, self.cache_size)
            solver.alpha = alpha
            for i in range(l):
                solver.neg_y_grad[i] -= y[i] * func(i) @ solver.alpha

        for n_iter in range(self.max_iter):
            i, j = solver.working_set_select()
            if i < 0:
                break

            solver.update(i, j, func)
        else:
            print("OneClassSVM not coverage with {} iterations".format(
                self.max_iter))

        rho = solver.calculate_rho()
        self.decision_function = lambda x: np.matmul(
            solver.alpha,
            kernel_func(X, x),
        ) - rho
        return self

    def predict(self, X):
        pred = self.decision_function(X)
        pred[pred <= 0] = -1
        pred[pred > 0] = 1
        return pred

    def score(self, X, y):
        raise NotImplementedError
