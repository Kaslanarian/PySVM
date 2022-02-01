from functools import lru_cache
from rff import RFF
import numpy as np
from base_svc import LinearSVC, KernelSVC


class LinearSVR(LinearSVC):
    def __init__(self, C=1, max_iter=1000, epsilon=0, tol=0.001) -> None:
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.C = C
        self.tol = tol

    def fit(self, X, y):
        self.X, z = np.array(X), np.array(y)
        self.l, self.n_features = X.shape

        y = np.empty(2 * self.l)
        y[:self.l], y[self.l:] = 1, -1

        p = self.epsilon + np.hstack((-z, z))  # 2l
        alpha = np.zeros(2 * self.l)
        neg_y_grad = -y * p
        n_iter = 0
        while n_iter < self.max_iter:
            Iup = np.argwhere(
                np.logical_or(
                    np.logical_and(alpha < self.C, y == 1),
                    np.logical_and(alpha > 0, y == -1),
                )).reshape(-1)
            Ilow = np.argwhere(
                np.logical_or(
                    np.logical_and(alpha < self.C, y == -1),
                    np.logical_and(alpha > 0, y == 1),
                )).reshape(-1)

            i, j = Iup[np.argmax(neg_y_grad[Iup])], Ilow[np.argmin(
                neg_y_grad[Ilow])]

            if neg_y_grad[i] - neg_y_grad[j] < self.tol:
                break

            Qi, Qj = self.calculate_product(i), self.calculate_product(j)
            old_alpha_i, old_alpha_j = alpha[i], alpha[j]

            quad_coef = Qi[i] + Qj[j] - 2 * y[i] * y[j] * Qi[j]
            if quad_coef <= 0:
                quad_coef = 1e-12

            if y[i] * y[j] == -1:
                delta = (neg_y_grad[i] * y[i] +
                         neg_y_grad[j] * y[j]) / quad_coef
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
                delta = (neg_y_grad[j] * y[j] -
                         neg_y_grad[i] * y[i]) / quad_coef
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
            neg_y_grad -= y * (Qi * delta_i + Qj * delta_j)
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
                np.logical_and(alpha == 0, y == -1),
                np.logical_and(alpha == self.C, y == 1),
            )
            lb_id = np.logical_or(
                np.logical_and(alpha == 0, y == 1),
                np.logical_and(alpha == self.C, y == -1),
            )
            rho = -(neg_y_grad[lb_id].min() + neg_y_grad[ub_id].max()) / 2

        w = (alpha[:self.l] - alpha[self.l:]) @ X
        self.decision_function = lambda x: w @ x.T - rho
        return self

    def predict(self, X):
        return self.decision_function(np.array(X).reshape(-1, self.n_features))

    def score(self, X, y):
        X = np.array(X).reshape(-1, self.n_features)
        y = np.array(y).reshape(-1)
        pred = self.predict(X)
        SS_tot = np.sum((y - y.mean())**2)
        SS_res = np.sum((y - pred)**2)
        return 1 - SS_res / SS_tot

    @lru_cache(maxsize=64)
    def calculate_product(self, i):
        if i < self.l:
            Qi = self.X @ self.X[i]
        else:
            Qi = -self.X @ self.X[i - self.l]
        return np.hstack((Qi, -Qi))


class KernelSVR(LinearSVR, KernelSVC):
    def __init__(self,
                 C=1,
                 epsilon=0,
                 kernel='rbf',
                 degree=3,
                 gamma='scale',
                 coef0=0,
                 max_iter=1000,
                 rff=False,
                 D=10000,
                 tol=1e-3) -> None:
        super().__init__(C, max_iter, epsilon, tol)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.rff = rff
        self.D = D

    def fit(self, X, y):
        self.X, z = np.array(X), np.array(y, dtype=float)
        self.l, self.n_features = self.X.shape

        self.kernel_func = self.register_kernel()

        y = np.empty(2 * self.l)
        y[:self.l], y[self.l:] = 1, -1
        p = self.epsilon + np.hstack((-z, z))  # 2l
        alpha = np.zeros(2 * self.l)

        neg_y_grad = -y * p
        n_iter = 0
        while n_iter < self.max_iter:
            Iup = np.argwhere(
                np.logical_or(
                    np.logical_and(alpha < self.C, y == 1),
                    np.logical_and(alpha > 0, y == -1),
                )).reshape(-1)
            Ilow = np.argwhere(
                np.logical_or(
                    np.logical_and(alpha < self.C, y == -1),
                    np.logical_and(alpha > 0, y == 1),
                )).reshape(-1)

            i, j = Iup[np.argmax(neg_y_grad[Iup])], Ilow[np.argmin(
                neg_y_grad[Ilow])]

            if neg_y_grad[i] - neg_y_grad[j] < self.tol:
                break

            Qi, Qj = self.calculate_kernel(i), self.calculate_kernel(j)
            old_alpha_i, old_alpha_j = alpha[i], alpha[j]

            quad_coef = Qi[i] + Qj[j] - 2 * y[i] * y[j] * Qi[j]
            if quad_coef <= 0:
                quad_coef = 1e-12

            if y[i] * y[j] == -1:
                delta = (neg_y_grad[i] * y[i] +
                         neg_y_grad[j] * y[j]) / quad_coef
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
                delta = (neg_y_grad[j] * y[j] -
                         neg_y_grad[i] * y[i]) / quad_coef
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
            neg_y_grad -= y * (Qi * delta_i + Qj * delta_j)
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
                np.logical_and(alpha == 0, y == -1),
                np.logical_and(alpha == self.C, y == 1),
            )
            lb_id = np.logical_or(
                np.logical_and(alpha == 0, y == 1),
                np.logical_and(alpha == self.C, y == -1),
            )
            rho = -(neg_y_grad[lb_id].min() + neg_y_grad[ub_id].max()) / 2
        alpha_diff = alpha[:self.l] - alpha[self.l:]
        self.decision_function = lambda x: alpha_diff @ self.kernel_func(
            self.X,
            x,
        ) - rho

        return self

    @lru_cache(maxsize=64)
    def calculate_kernel(self, i):
        if i < self.l:
            Qi = self.kernel_func(self.X, self.X[i:i + 1]).reshape(-1)
        else:
            Qi = -self.kernel_func(
                self.X, self.X[i - self.l:i - self.l + 1]).reshape(-1)
        return np.hstack((Qi, -Qi))


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
                 D=10000,
                 tol=0.001) -> None:
        super().__init__(
            C=C,
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
        self.X, z = np.array(X), np.array(y)
        self.l, self.n_features = X.shape

        self.kernel_func = self.register_kernel()

        y = np.empty(2 * self.l)
        y[:self.l], y[self.l:] = 1, -1

        p = self.epsilon + np.hstack((-z, z))  # 2l
        alpha = np.zeros(2 * self.l)

        sum = self.C * self.nu * self.l / 2
        for i in range(self.l):
            alpha[i] = alpha[i + self.l] = min(sum, self.C)
            sum -= alpha[i]

        grad = np.empty(2 * self.l)
        neg_y_prod_grad = np.empty(2 * self.l)
        for i in range(2 * self.l):
            grad[i] = self.calculate_kernel(i) @ alpha + p[i]
            neg_y_prod_grad[i] = -y[i] * grad[i]

        n_iter = 0
        while n_iter < self.max_iter:
            Iup = np.argwhere(
                np.logical_or(
                    np.logical_and(alpha < 1, y == 1),
                    np.logical_and(alpha > 0, y == -1),
                )).reshape(-1)
            Ilow = np.argwhere(
                np.logical_or(
                    np.logical_and(alpha < 1, y == -1),
                    np.logical_and(alpha > 0, y == 1),
                )).reshape(-1)

            Imp = Iup[y[Iup] == 1]
            IMp = Ilow[y[Ilow] == 1]
            Imn = Iup[y[Iup] == -1]
            IMn = Ilow[y[Ilow] == -1]

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

            if y[i_p] != y[j_p]:
                quad_coef = Qip[i_p] + Qjp[j_p] + 2 * Qip[j_p]
                if quad_coef <= 0:
                    quad_coef = 1e-12
                term_p = -(-grad[i_p] - grad[j_p])**2 / (2 * quad_coef)
            else:
                quad_coef = Qip[i_p] + Qjp[j_p] - 2 * Qip[j_p]
                if quad_coef <= 0:
                    quad_coef = 1e-12
                term_p = -(grad[i_p] - grad[j_p])**2 / (2 * quad_coef)

            if y[i_n] != y[j_n]:
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
            neg_y_prod_grad -= y * detla_grad
            n_iter += 1

        if n_iter == self.max_iter:
            print("Not coverage")

        pos_sv = np.logical_and(
            np.logical_and(alpha > 0, alpha < self.C),
            y == 1,
        )
        if pos_sv.sum() == 0:
            r1 = ((grad[np.logical_and(
                alpha == 1,
                y == 1,
            )]).max() + (grad[np.logical_and(
                alpha == 0,
                y == 1,
            )])).min() / 2
        else:
            r1 = np.average(grad[pos_sv])

        neg_sv = np.logical_and(
            np.logical_and(alpha > 0, alpha < self.C),
            y == -1,
        )
        if neg_sv.sum() == 0:
            r2 = (grad[np.logical_and(
                alpha == 1,
                y == -1,
            )].max() + grad[np.logical_and(
                alpha == 0,
                y == -1,
            )].min()) / 2
        else:
            r2 = np.average(grad[neg_sv])
        b = (r2 - r1) / 2
        alpha_diff = alpha[:self.l] - alpha[self.l:]
        self.decision_function = lambda x: alpha_diff @ self.kernel_func(
            self.X,
            x,
        ) + b
        return self
