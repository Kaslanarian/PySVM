import numpy as np
from functools import lru_cache
from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from rff import RFF


class LinearSVC(BaseEstimator):
    def __init__(self, C=1, max_iter=1000, tol=1e-3, verbose=False) -> None:
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y):
        self.X, self.y = np.array(X), np.array(y, dtype=float)
        self.y[self.y != 1] = -1
        l, self.n_features = X.shape

        # 求解SMO
        self.alpha = np.zeros(l)
        self.w = np.zeros(self.n_features)
        self.neg_y_grad = np.copy(self.y)  # 计算 -y·▽f(α)
        self.has_coverage = False
        n_iter = 0
        while self.max_iter == -1 or n_iter < self.max_iter:
            # 选变量
            Iup = np.argwhere(
                np.logical_or(
                    np.logical_and(self.alpha < self.C, self.y == 1),
                    np.logical_and(self.alpha > 0, self.y == -1),
                )).reshape(-1)
            Ilow = np.argwhere(
                np.logical_or(
                    np.logical_and(self.alpha < self.C, self.y == -1),
                    np.logical_and(self.alpha > 0, self.y == 1),
                )).reshape(-1)

            i, j = Iup[np.argmax(self.neg_y_grad[Iup])], Ilow[np.argmin(
                self.neg_y_grad[Ilow])]

            if self.neg_y_grad[i] - self.neg_y_grad[j] < self.tol:
                break
            # 更新
            Qi, Qj = self.calculate_product(i), self.calculate_product(j)

            old_alpha_i, old_alpha_j = self.alpha[i], self.alpha[j]
            yi, yj = self.y[i], self.y[j]

            quad_coef = Qi[i] + Qj[j] - 2 * yi * yj * Qi[j]
            if quad_coef <= 0:
                quad_coef = 1e-12

            if yi * yj == -1:
                delta = (self.neg_y_grad[i] * yi +
                         self.neg_y_grad[j] * yj) / quad_coef
                diff = old_alpha_i - old_alpha_j
                self.alpha[i] += delta
                self.alpha[j] += delta

                if diff > 0:
                    if (self.alpha[j] < 0):
                        self.alpha[j] = 0
                        self.alpha[i] = diff

                else:
                    if (self.alpha[i] < 0):
                        self.alpha[i] = 0
                        self.alpha[j] = -diff

            else:
                delta = (self.neg_y_grad[j] * yj -
                         self.neg_y_grad[i] * yi) / quad_coef
                sum = self.alpha[i] + self.alpha[j]
                self.alpha[i] -= delta
                self.alpha[j] += delta

                if sum > self.C:
                    if self.alpha[i] > self.C:
                        self.alpha[i] = self.C
                        self.alpha[j] = sum - self.C

                else:
                    if self.alpha[j] < 0:
                        self.alpha[j] = 0
                        self.alpha[i] = sum

            delta_i, delta_j = self.alpha[i] - old_alpha_i, self.alpha[
                j] - old_alpha_j
            self.neg_y_grad -= self.y * (Qi * delta_i + Qj * delta_j)
            self.w += delta_i * yi * self.X[i] + delta_j * yj * self.X[j]
            n_iter += 1

        sv = np.logical_and(
            self.alpha > 0,
            self.alpha < self.C,
        )
        if sv.sum() > 0:
            self.rho = -self.neg_y_grad[sv].sum() / sv.sum()
        else:
            ub_id = np.logical_or(
                np.logical_and(self.alpha == 0, self.y == -1),
                np.logical_and(self.alpha == self.C, self.y == 1),
            )
            lb_id = np.logical_or(
                np.logical_and(self.alpha == 0, self.y == 1),
                np.logical_and(self.alpha == self.C, self.y == -1),
            )
            self.rho = -(self.neg_y_grad[lb_id].min() +
                         self.neg_y_grad[ub_id].max()) / 2
        self.b = -self.rho
        self.decision_function = lambda x: self.w @ x.T + self.b

        return self

    def predict(self, X):
        X = np.array(X).reshape(-1, self.n_features)
        pred = self.decision_function(X)
        pred[pred > 0] = 1
        pred[pred < 0] = 0
        return pred.astype('int')

    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)

    @lru_cache(maxsize=256)
    def calculate_product(self, i: int):
        return self.y * (self.X @ self.X[i]) * self.y[i]
