import numpy as np
from functools import lru_cache


class Solver:
    '''
    Solve problem

    min_a  1 / 2 a^T Q a + p^T a
    s.t.   y^T a = 0 
           0 <= a_i <= C.

    without cache mechanism.
    '''
    def __init__(self, Q, p, y, C, tol=1e-5) -> None:
        problem_size = p.shape[0]
        assert problem_size == y.shape[0]
        if Q is not None:
            assert problem_size == Q.shape[0]
            assert problem_size == Q.shape[1]

        self.Q = Q
        self.p = p
        self.y = y
        self.C = C
        self.tol = tol
        self.alpha = np.zeros(problem_size)

        # Calculate -y·▽f(α)
        self.neg_y_grad = -y * p

    def working_set_select(self):
        Iup = np.argwhere(
            np.logical_or(
                np.logical_and(self.alpha < self.C, self.y > 0),
                np.logical_and(self.alpha > 0, self.y < 0),
            )).flatten()
        Ilow = np.argwhere(
            np.logical_or(
                np.logical_and(self.alpha < self.C, self.y < 0),
                np.logical_and(self.alpha > 0, self.y > 0),
            )).flatten()
        i, j = (
            Iup[np.argmax(self.neg_y_grad[Iup])],
            Ilow[np.argmin(self.neg_y_grad[Ilow])],
        )
        if self.neg_y_grad[i] - self.neg_y_grad[j] < self.tol:
            return -1, -1
        return i, j

    def update(self, i, j, func=None):
        Qi, Qj = self.get_Q(i, func), self.get_Q(j, func)
        yi, yj = self.y[i], self.y[j]
        alpha_i, alpha_j = self.alpha[i], self.alpha[j]

        quad_coef = Qi[i] + Qj[j] - 2 * yi * yj * Qi[j]
        if quad_coef <= 0:
            quad_coef = 1e-12

        if yi * yj == -1:
            delta = (self.neg_y_grad[i] * yi +
                     self.neg_y_grad[j] * yj) / quad_coef
            diff = alpha_i - alpha_j
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

            if diff > 0:
                if (self.alpha[i] > self.C):
                    self.alpha[i] = self.C
                    self.alpha[j] = self.C - diff

            else:
                if (self.alpha[j] > self.C):
                    self.alpha[j] = self.C
                    self.alpha[i] = self.C + diff

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

            if sum > self.C:
                if self.alpha[j] > self.C:
                    self.alpha[j] = self.C
                    self.alpha[i] = sum - self.C

            else:
                if self.alpha[i] < 0:
                    self.alpha[i] = 0
                    self.alpha[j] = sum

        delta_i = self.alpha[i] - alpha_i
        delta_j = self.alpha[j] - alpha_j
        self.neg_y_grad -= self.y * (delta_i * Qi + delta_j * Qj)
        return delta_i, delta_j

    def calculate_rho(self):
        sv = np.logical_and(
            self.alpha > 0,
            self.alpha < self.C,
        )
        if sv.sum() > 0:
            rho = -np.average(self.neg_y_grad[sv])
        else:
            ub_id = np.logical_or(
                np.logical_and(self.alpha == 0, self.y < 0),
                np.logical_and(self.alpha == self.C, self.y > 0),
            )
            lb_id = np.logical_or(
                np.logical_and(self.alpha == 0, self.y > 0),
                np.logical_and(self.alpha == self.C, self.y < 0),
            )
            rho = -(self.neg_y_grad[lb_id].min() +
                    self.neg_y_grad[ub_id].max()) / 2
        return rho

    def get_Q(self, i, func=None):
        return self.Q[i]


class SolverWithCache(Solver):
    cache_size = 256

    def __init__(self, p, y, C, tol=1e-5, cache_size=256) -> None:
        super().__init__(None, p, y, C, tol)
        cache_size = cache_size

    def working_set_select(self):
        return super().working_set_select()

    def update(self, i, j, func=None):
        return super().update(i, j, func=func)

    def calculate_rho(self):
        return super().calculate_rho()

    @lru_cache(cache_size)
    def get_Q(self, i, func):
        return func(i)


class NuSolver(Solver):
    '''
    Solve problem

    min_a  1/2 a^T Q a + p^T a\n
    s.t.   y^T a = 0\n
           e^T a = t\n
           0 <= a_i <= C\n
    '''
    def __init__(self, Q, p, y, t, C, tol=1e-5) -> None:
        super().__init__(Q, p, y, C, tol)
        problem_size = p.shape[0]
        assert problem_size == y.shape[0]
        if Q is not None:
            assert problem_size == Q.shape[0]
            assert problem_size == Q.shape[1]

        sum_pos = sum_neg = t / 2
        self.alpha = np.empty(problem_size)

        for i in range(problem_size):
            if self.y[i] == 1:
                self.alpha[i] = min(1., sum_pos)
                sum_pos -= self.alpha[i]
            else:
                self.alpha[i] = min(1., sum_neg)
                sum_neg -= self.alpha[i]

        self.neg_y_grad = -self.y * (np.matmul(Q, self.alpha) + self.p)
        self.QD = np.diag(self.Q)

    def working_set_select(self, func=None):
        Iup = np.argwhere(
            np.logical_or(
                np.logical_and(self.alpha < self.C, self.y > 0),
                np.logical_and(self.alpha > 0, self.y < 0),
            )).flatten()
        Ilow = np.argwhere(
            np.logical_or(
                np.logical_and(self.alpha < self.C, self.y < 0),
                np.logical_and(self.alpha > 0, self.y > 0),
            )).flatten()

        pos_fail, neg_fail = False, False
        try:
            Imp = Iup[self.y[Iup] > 0]
            IMp = Ilow[self.y[Ilow] > 0]
            i_p = Imp[np.argmax(self.neg_y_grad[Imp])]

            grad_diff = self.neg_y_grad[IMp] - self.neg_y_grad[i_p]
            Qip = self.get_Q(i_p, func)
            quad_coef = Qip[i_p] + self.QD[IMp] - 2 * Qip[IMp]
            quad_coef[quad_coef <= 0] = 1e-12
            obj_diff_p = -grad_diff**2 / quad_coef
            argmin = np.argmin(obj_diff_p)
            j_p = IMp[argmin]
            min_p = obj_diff_p[argmin]

            m_p = self.neg_y_grad[i_p]
            M_p = self.neg_y_grad[j_p]
        except:
            pos_fail = True

        try:
            Imn = Iup[self.y[Iup] < 0]
            IMn = Ilow[self.y[Ilow] < 0]
            i_n = Imn[np.argmax(self.neg_y_grad[Imn])]

            grad_diff = self.neg_y_grad[IMn] - self.neg_y_grad[i_n]
            Qin = self.get_Q(i_n, func)
            quad_coef = Qin[i_n] + self.QD[IMn] - 2 * Qin[IMn]
            quad_coef[quad_coef <= 0] = 1e-12
            obj_diff_n = -grad_diff**2 / quad_coef
            argmin = np.argmin(obj_diff_n)
            j_n = IMn[argmin]
            min_n = obj_diff_n[argmin]

            m_n = self.neg_y_grad[i_n]
            M_n = self.neg_y_grad[j_n]
        except:
            neg_fail = True

        if not pos_fail and not neg_fail:
            if max(m_p - M_p, m_n - M_n) < self.tol:
                return -1, -1, -1, -1
            elif min_p < min_n:
                return i_p, j_p, Qip, self.get_Q(j_p, func)
            else:
                return i_n, j_n, Qin, self.get_Q(j_n, func)
        elif pos_fail and not neg_fail:
            return i_n, j_n, Qin, self.get_Q(j_n, func)
        elif not pos_fail and neg_fail:
            return i_p, j_p, Qip, self.get_Q(j_p, func)
        else:
            return -1, -1, -1, -1

    def update(self, i, j, Qi, Qj):
        alpha_i, alpha_j = self.alpha[i], self.alpha[j]
        quad_coef = Qi[i] + Qj[j] - 2 * Qi[j]
        if quad_coef <= 0:
            quad_coef = 1e-12
        delta = self.y[i] * (self.neg_y_grad[j] -
                             self.neg_y_grad[i]) / quad_coef

        sum = alpha_i + alpha_j
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

        if sum > self.C:
            if self.alpha[j] > self.C:
                self.alpha[j] = self.C
                self.alpha[i] = sum - self.C

        else:
            if self.alpha[i] < 0:
                self.alpha[i] = 0
                self.alpha[j] = sum

        delta_i, delta_j = self.alpha[i] - alpha_i, self.alpha[j] - alpha_j
        self.neg_y_grad -= self.y * (delta_i * Qi + delta_j * Qj)

        return delta_i, delta_j

    def calculate_rho_b(self):
        grad = -self.y * self.neg_y_grad
        pos_sv = np.logical_and(
            np.logical_and(self.alpha > 0, self.alpha < 1),
            self.y == 1,
        )
        if pos_sv.sum() == 0:
            try:
                r1 = ((grad[np.logical_and(
                    self.alpha == 1,
                    self.y == 1,
                )]).max() + (grad[np.logical_and(
                    self.alpha == 0,
                    self.y == 1,
                )])).min() / 2
            except:
                r1 = 0
        else:
            r1 = np.average(grad[pos_sv])

        neg_sv = np.logical_and(
            np.logical_and(self.alpha > 0, self.alpha < 1),
            self.y == -1,
        )
        if neg_sv.sum() == 0:
            try:
                r2 = (grad[np.logical_and(
                    self.alpha == 1,
                    self.y == -1,
                )].max() + grad[np.logical_and(
                    self.alpha == 0,
                    self.y == -1,
                )].min()) / 2
            except:
                r2 = 0
        else:
            r2 = np.average(grad[neg_sv])

        rho = (r1 + r2) / 2
        b = (r2 - r1) / 2

        return rho, b


class NuSolverWithCache(NuSolver, SolverWithCache):
    cache_size = 256

    def __init__(self, p, y, t, C, func, tol=1e-5, cache_size=256) -> None:
        self.p = p
        self.y = y
        self.C = C
        self.tol = tol

        cache_size = cache_size
        problem_size = p.shape[0]
        assert problem_size == y.shape[0]

        sum_pos = sum_neg = t / 2
        self.alpha = np.zeros(problem_size)
        self.neg_y_grad = np.zeros(problem_size)

        for i in range(problem_size):
            if self.y[i] == 1:
                self.alpha[i] = min(1., sum_pos)
                sum_pos -= self.alpha[i]
            else:
                self.alpha[i] = min(1., sum_neg)
                sum_neg -= self.alpha[i]

        QD = []
        for i in range(problem_size):
            self.neg_y_grad[i] = self.get_Q(i, func) @ self.alpha + self.p[i]
            QD.append(self.get_Q(i, func)[i])
        self.neg_y_grad *= -self.y
        self.QD = np.array(QD)

    def working_set_select(self, func=None):
        return super().working_set_select(func)

    def update(self, i, j, Qi, Qj):
        return super().update(i, j, Qi, Qj)

    def calculate_rho_b(self):
        return super().calculate_rho_b()

    @lru_cache(cache_size)
    def get_Q(self, i, func):
        return func(i)
