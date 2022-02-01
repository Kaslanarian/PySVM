from functools import lru_cache
import numpy as np
from sklearn.base import BaseEstimator
from rff import RFF


class LinearSVC(BaseEstimator):
    def __init__(self, C=1, max_iter=1000, tol=1e-3) -> None:
        self.C = C
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self.X, self.y = np.array(X), np.array(y, dtype=float)
        self.y[self.y != 1] = -1
        l, self.n_features = X.shape

        # Solve SMO
        alpha = np.zeros(l)
        w = np.zeros(self.n_features)
        neg_y_grad = np.copy(self.y)  # Calculate -y·▽f(α)
        n_iter = 0
        while n_iter < self.max_iter:
            # wss
            Iup = np.argwhere(
                np.logical_or(
                    np.logical_and(alpha < self.C, self.y == 1),
                    np.logical_and(alpha > 0, self.y == -1),
                )).reshape(-1)
            Ilow = np.argwhere(
                np.logical_or(
                    np.logical_and(alpha < self.C, self.y == -1),
                    np.logical_and(alpha > 0, self.y == 1),
                )).reshape(-1)

            i, j = Iup[np.argmax(neg_y_grad[Iup])], Ilow[np.argmin(
                neg_y_grad[Ilow])]

            if neg_y_grad[i] - neg_y_grad[j] < self.tol:
                break

            # Update
            Qi, Qj = self.__calculate_product(i), self.__calculate_product(j)
            old_alpha_i, old_alpha_j = alpha[i], alpha[j]
            yi, yj = self.y[i], self.y[j]

            quad_coef = Qi[i] + Qj[j] - 2 * yi * yj * Qi[j]
            if quad_coef <= 0:
                quad_coef = 1e-12

            if yi * yj == -1:
                delta = (neg_y_grad[i] * yi + neg_y_grad[j] * yj) / quad_coef
                diff = old_alpha_i - old_alpha_j
                alpha[i] += delta
                alpha[j] += delta

                if diff > 0:
                    if (alpha[j] < 0):
                        alpha[j] = 0
                        alpha[i] = diff

                else:
                    if (alpha[i] < 0):
                        alpha[i] = 0
                        alpha[j] = -diff

                if diff > 0:
                    if (alpha[i] > self.C):
                        alpha[i] = self.C
                        alpha[j] = self.C - diff

                else:
                    if (alpha[j] > self.C):
                        alpha[j] = self.C
                        alpha[i] = self.C + diff

            else:
                delta = (neg_y_grad[j] * yj - neg_y_grad[i] * yi) / quad_coef
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
            neg_y_grad -= self.y * (Qi * delta_i + Qj * delta_j)
            w += delta_i * yi * self.X[i] + delta_j * yj * self.X[j]
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
            ub_id = np.logical_or(
                np.logical_and(alpha == 0, self.y == -1),
                np.logical_and(alpha == self.C, self.y == 1),
            )
            lb_id = np.logical_or(
                np.logical_and(alpha == 0, self.y == 1),
                np.logical_and(alpha == self.C, self.y == -1),
            )
            rho = -(neg_y_grad[lb_id].min() + neg_y_grad[ub_id].max()) / 2
        self.decision_function = lambda x: w @ x.T - rho

        return self

    def predict(self, X):
        X = np.array(X).reshape(-1, self.n_features)  # (l * n_f)
        pred = self.decision_function(X)
        pred[pred > 0] = 1
        pred[pred < 0] = 0
        return pred.astype('int')

    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)

    @lru_cache(maxsize=128)
    def __calculate_product(self, i):
        return self.y * (self.X @ self.X[i]) * self.y[i]


class KernelSVC(LinearSVC):
    def __init__(self,
                 C=1,
                 kernel='rbf',
                 degree=3,
                 gamma='scale',
                 coef0=0,
                 max_iter=1000,
                 rff=False,
                 D=10000,
                 tol=1e-3) -> None:
        super().__init__(C=C, max_iter=max_iter, tol=tol)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.rff = rff
        self.D = D

    def fit(self, X, y):
        self.X, self.y = np.array(X), np.array(y, dtype=float)
        l, self.n_features = X.shape

        self.kernel_func = self.register_kernel()

        # 计算Q
        self.y[self.y != 1] = -1
        alpha = np.zeros(l)
        neg_y_grad = np.copy(self.y)  # Calculate -y·▽f(α)
        n_iter = 0
        while n_iter < self.max_iter:
            # wss
            Iup = np.argwhere(
                np.logical_or(
                    np.logical_and(alpha < self.C, self.y == 1),
                    np.logical_and(alpha > 0, self.y == -1),
                )).reshape(-1)
            Ilow = np.argwhere(
                np.logical_or(
                    np.logical_and(alpha < self.C, self.y == -1),
                    np.logical_and(alpha > 0, self.y == 1),
                )).reshape(-1)

            i, j = Iup[np.argmax(neg_y_grad[Iup])], Ilow[np.argmin(
                neg_y_grad[Ilow])]

            if neg_y_grad[i] - neg_y_grad[j] < self.tol:
                break

            Qi, Qj = self.calculate_kernel(i), self.calculate_kernel(j)
            old_alpha_i, old_alpha_j = alpha[i], alpha[j]
            yi, yj = self.y[i], self.y[j]

            quad_coef = Qi[i] + Qj[j] - 2 * yi * yj * Qi[j]
            if quad_coef <= 0:
                quad_coef = 1e-12

            if yi * yj == -1:
                delta = (neg_y_grad[i] * yi + neg_y_grad[j] * yj) / quad_coef
                diff = old_alpha_i - old_alpha_j
                alpha[i] += delta
                alpha[j] += delta

                if diff > 0:
                    if (alpha[j] < 0):
                        alpha[j] = 0
                        alpha[i] = diff

                else:
                    if (alpha[i] < 0):
                        alpha[i] = 0
                        alpha[j] = -diff

                if diff > 0:
                    if (alpha[i] > self.C):
                        alpha[i] = self.C
                        alpha[j] = self.C - diff

                else:
                    if (alpha[j] > self.C):
                        alpha[j] = self.C
                        alpha[i] = self.C + diff

            else:
                delta = (neg_y_grad[j] * yj - neg_y_grad[i] * yi) / quad_coef
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
            neg_y_grad -= self.y * (Qi * delta_i + Qj * delta_j)
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
            ub_id = np.logical_or(
                np.logical_and(alpha == 0, self.y == -1),
                np.logical_and(alpha == self.C, self.y == 1),
            )
            lb_id = np.logical_or(
                np.logical_and(alpha == 0, self.y == 1),
                np.logical_and(alpha == self.C, self.y == -1),
            )
            rho = -(neg_y_grad[lb_id].min() + neg_y_grad[ub_id].max()) / 2
        self.decision_function = lambda x: alpha * self.y @ self.kernel_func(
            X,
            x,
        ) - rho
        return self

    def register_kernel(self):
        if type(self.gamma) == float:
            gamma = self.gamma
        else:
            gamma = {
                'scale': 1 / (self.n_features * self.X.std()),
                'auto': 1 / self.n_features,
            }[self.gamma]

        # 是否使用随机傅里叶特征
        if self.rff:
            rff = RFF(gamma, self.D).fit(self.X)
            rbf_func = lambda x, y: rff.transform(x) @ rff.transform(y).T
        else:
            rbf_func = lambda x, y: np.exp(-gamma * (
                (x**2).sum(1).reshape(-1, 1) + (y**2).sum(1) - 2 * x @ y.T))

        degree = self.degree
        coef0 = self.coef0
        return {
            "linear": lambda x, y: x @ y.T,
            "poly": lambda x, y: (gamma * x @ y.T + coef0)**degree,
            "rbf": rbf_func,
            "sigmoid": lambda x, y: np.tanh(gamma * (x @ y.T) + coef0)
        }[self.kernel]

    @lru_cache(maxsize=128)
    def calculate_kernel(self, i):
        return self.y * self.kernel_func(
            self.X, self.X[i:i + 1]).reshape(-1) * self.y[i]


class NuSVC(KernelSVC):
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
        )
        self.nu = nu

    def fit(self, X, y):
        self.X, self.y = np.array(X), np.array(y, dtype=float)
        self.y[self.y != 1] = -1
        l, self.n_features = X.shape

        self.kernel_func = self.register_kernel()

        # 计算alpha初始值
        sum_pos = self.nu * l / 2
        sum_neg = self.nu * l / 2
        alpha = np.empty(l)
        grad = np.empty(l)
        neg_y_prod_grad = np.empty(l)

        for i in range(l):
            if self.y[i] == +1:
                alpha[i] = min(1., sum_pos)
                sum_pos -= alpha[i]
            else:
                alpha[i] = min(1., sum_neg)
                sum_neg -= alpha[i]

        for i in range(l):
            grad[i] = self.calculate_kernel(i) @ alpha
            neg_y_prod_grad[i] = -self.y[i] * grad[i]

        n_iter = 0
        while n_iter < self.max_iter:
            # wss
            Iup = np.argwhere(
                np.logical_or(
                    np.logical_and(alpha < 1, self.y == 1),
                    np.logical_and(alpha > 0, self.y == -1),
                )).reshape(-1)
            Ilow = np.argwhere(
                np.logical_or(
                    np.logical_and(alpha < 1, self.y == -1),
                    np.logical_and(alpha > 0, self.y == 1),
                )).reshape(-1)

            Imp = Iup[self.y[Iup] == 1]
            IMp = Ilow[self.y[Ilow] == 1]
            Imn = Iup[self.y[Iup] == -1]
            IMn = Ilow[self.y[Ilow] == -1]

            i_p = Imp[np.argmax(neg_y_prod_grad[Imp])]
            j_p = IMp[np.argmin(neg_y_prod_grad[IMp])]
            i_n = Imn[np.argmax(neg_y_prod_grad[Imn])]
            j_n = IMn[np.argmin(neg_y_prod_grad[IMn])]

            m_p = neg_y_prod_grad[i_p]
            M_p = neg_y_prod_grad[j_p]
            m_n = neg_y_prod_grad[i_n]
            M_n = neg_y_prod_grad[j_n]

            if max(m_p - M_p, m_n - M_n) < self.tol:
                break

            Qip = self.calculate_kernel(i_p)
            Qjp = self.calculate_kernel(j_p)
            Qin = self.calculate_kernel(i_n)
            Qjn = self.calculate_kernel(j_n)

            if self.y[i_p] != self.y[j_p]:
                quad_coef = Qip[i_p] + Qjp[j_p] + 2 * Qip[j_p]
                if quad_coef <= 0:
                    quad_coef = 1e-12
                term_p = -(-grad[i_p] - grad[j_p])**2 / (2 * quad_coef)
            else:
                quad_coef = Qip[i_p] + Qjp[j_p] - 2 * Qip[j_p]
                if quad_coef <= 0:
                    quad_coef = 1e-12
                term_p = -(grad[i_p] - grad[j_p])**2 / (2 * quad_coef)

            if self.y[i_n] != self.y[j_n]:
                quad_coef = Qin[i_n] + Qjn[j_n] + 2 * Qin[j_n]
                if quad_coef <= 0:
                    quad_coef = 1e-12
                term_n = -(-grad[i_n] - grad[j_n])**2 / (2 * quad_coef)
            else:
                quad_coef = Qin[i_n] + Qjn[j_n] - 2 * Qin[j_n]
                if quad_coef <= 0:
                    quad_coef = 1e-12
                term_n = -(grad[i_n] - grad[j_p])**2 / (2 * quad_coef)
            i, j, Qi, Qj = (i_p, j_p, Qip,
                            Qjp) if term_p < term_n else (i_n, j_n, Qin, Qjn)

            old_alpha_i, old_alpha_j = alpha[i], alpha[j]
            quad_coef = Qi[i] + Qj[j] - 2 * Qi[j]
            if quad_coef <= 0:
                quad_coef = 1e-12
            delta = (grad[i] - grad[j]) / quad_coef

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

            detla_grad = (alpha[i] - old_alpha_i) * Qi + (alpha[j] -
                                                          old_alpha_j) * Qj
            grad += detla_grad
            neg_y_prod_grad -= self.y * detla_grad
            n_iter += 1

        if n_iter == self.max_iter:
            print("Not coverage")

        pos_sv = np.logical_and(
            np.logical_and(alpha > 0, alpha < 1),
            self.y == 1,
        )
        if pos_sv.sum() == 0:
            r1 = ((grad[np.logical_and(
                alpha == 1,
                self.y == 1,
            )]).max() + (grad[np.logical_and(
                alpha == 0,
                self.y == 1,
            )])).min() / 2
        else:
            r1 = np.average(grad[pos_sv])

        neg_sv = np.logical_and(
            np.logical_and(alpha > 0, alpha < 1),
            self.y == -1,
        )
        if neg_sv.sum() == 0:
            r2 = (grad[np.logical_and(
                alpha == 1,
                self.y == -1,
            )].max() + grad[np.logical_and(
                alpha == 0,
                self.y == -1,
            )].min()) / 2
        else:
            r2 = np.average(grad[neg_sv])
        rho = (r1 + r2) / 2
        b = (r2 - r1) / 2
        self.decision_function = lambda x: alpha / rho * self.y @ self.kernel_func(
            X, x) + b / rho
        return self