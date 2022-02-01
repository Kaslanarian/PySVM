import base_svc
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.base import BaseEstimator

multi_method = {
    "ovo": OneVsOneClassifier,
    "ovr": OneVsRestClassifier,
}


class MultiLinearSVC(BaseEstimator):
    def __init__(self,
                 C=1,
                 max_iter=1000,
                 tol=1e-3,
                 n_jobs=None,
                 method="ovo") -> None:
        super().__init__()
        self.max_iter = max_iter
        self.C = C
        self.tol = tol
        self.n_jobs = n_jobs
        self.method = method

        self.base_model = base_svc.LinearSVC(C, max_iter, tol)

    def fit(self, X, y):
        self.multi_model = multi_method[self.method](
            estimator=self.base_model,
            n_jobs=self.n_jobs,
        )
        self.multi_model.fit(X, y)
        return self

    def predict(self, X):
        return self.multi_model.predict(X)

    def score(self, X, y):
        pred = self.predict(X)
        return (pred == y).mean(0)


class MultiKernelSVC(MultiLinearSVC):
    def __init__(self,
                 C=1,
                 kernel='rbf',
                 degree=3,
                 gamma='scale',
                 coef0=0,
                 max_iter=1000,
                 rff=False,
                 D=10000,
                 tol=1e-3,
                 n_jobs=None,
                 method="ovo") -> None:
        super().__init__(
            C=C,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
            method=method,
        )
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.rff = rff,
        self.D = D,

        self.base_model = base_svc.KernelSVC(
            C,
            kernel,
            degree,
            gamma,
            coef0,
            max_iter,
            rff,
            D,
            tol,
        )


class MultiNuSVC(MultiKernelSVC):
    def __init__(
        self,
        nu=0.5,
        kernel='rbf',
        degree=3,
        gamma='scale',
        coef0=0,
        max_iter=1000,
        rff=False,
        D=10000,
        tol=1e-3,
        n_jobs=None,
        method="ovo",
    ) -> None:
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            max_iter=max_iter,
            rff=rff,
            D=D,
            tol=tol,
            n_jobs=n_jobs,
            method=method,
        )
        self.nu = nu
        self.base_model = base_svc.NuSVC(
            nu,
            kernel,
            degree,
            gamma,
            coef0,
            max_iter,
            rff,
            D,
            tol,
        )


LinearSVC = MultiLinearSVC
KernelSVC = MultiKernelSVC
NuSVC = MultiNuSVC
