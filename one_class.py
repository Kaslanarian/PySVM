from functools import lru_cache
import numpy as np
from rff import RFF
from base_svc import NuSVC


class OneClassSVM(NuSVC):
    def __init__(self,
                 nu=0.5,
                 kernel='rbf',
                 degree=3,
                 gamma='scale',
                 coef0=0,
                 max_iter=1000,
                 rff=False,
                 D=10000,
                 tol=1e-3) -> None:
        super().__init__(
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

    def fit(self, X):
        self.X = np.array(X)
        l, self.n_features = X.shape

        self.kernel_func = self.register_kernel()

        n = int(self.nu * l)
        alpha = np.ones(l)
        if n < l:
            alpha[n] = self.nu * l - n
        for i in range(n + 1, l):
            alpha[i] = 0

        neg_y_grad = np.empty(l)
        for i in range(l):
            neg_y_grad[i] = -self.calculate_kernel(i) @ alpha
        n_iter = 0
        while n_iter < self.max_iter:
            Iup = np.argwhere(alpha < self.C).reshape(-1)
            Ilow = np.argwhere(alpha > 0).reshape(-1)

            i, j = Iup[np.argmax(neg_y_grad[Iup])], Ilow[np.argmin(
                neg_y_grad[Ilow])]

            if neg_y_grad[i] - neg_y_grad[j] < self.tol:
                break

            Qi, Qj = self.calculate_kernel(i), self.calculate_kernel(j)
            old_alpha_i, old_alpha_j = alpha[i], alpha[j]

            quad_coef = Qi[i] + Qj[j] - 2 * Qi[j]
            if quad_coef <= 0:
                quad_coef = 1e-12
            delta = (neg_y_grad[j] - neg_y_grad[i]) / quad_coef
            sum = alpha[i] + alpha[j]
            alpha[i] -= delta
            alpha[j] += delta

            if sum > self.C:
                if alpha[i] > self.C:
                    alpha[i] = self.C
                    alpha[j] = sum - self.C

            else:
                if alpha[j] < 0:
                    alpha[j] = 0
                    alpha[i] = sum

            if sum > self.C:
                if alpha[j] > self.C:
                    alpha[j] = self.C
                    alpha[i] = sum - self.C

            else:
                if alpha[i] < 0:
                    alpha[i] = 0
                    alpha[j] = sum

            delta_i, delta_j = alpha[i] - old_alpha_i, alpha[j] - old_alpha_j
            neg_y_grad -= Qi * delta_i + Qj * delta_j
            n_iter += 1

        if n_iter == self.max_iter:
            print("Not coverage")

        sv = np.logical_and(
            alpha > 0,
            alpha < self.C,
        )
        if sv.sum() > 0:
            rho = -np.average(neg_y_grad[sv])
        else:
            ub_id = alpha == self.C
            lb_id = alpha == 0
            rho = -(neg_y_grad[lb_id].min() + neg_y_grad[ub_id].max()) / 2
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

    @lru_cache(maxsize=64)
    def calculate_kernel(self, i):
        return self.kernel_func(self.X, self.X[i:i + 1]).reshape(-1)
