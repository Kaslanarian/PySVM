import numpy as np


class Solver:
    '''
    We want to solve the optimization problem using SMO:
    ```markdown
    min x^T·Q·x / 2 + px
    s.t. y^T·x = delta, 0≤x_i≤C
    ```

    Parameters
    ----------
    l : 向量长度
    Q : 待优化函数的二次项
    p : 待优化函数的一次项
    y : 限制条件中的y向量
    Cp: 正样本的限制C
    Cn: 负样本的限制C
    tol: 容忍度
    max_iter: 最大迭代次数
    '''
    def __init__(
        self,
        l: int,
        Q,
        p,
        y,
        alpha,
        Cp: float,
        Cn: float,
        eps=1e-3,
        max_iter: int = 1000,
    ) -> None:
        self.l = l
        self.Q = Q
        self.p = p
        self.y = y
        self.C = (Cp, Cn)
        self.alpha = alpha
        self.eps = eps
        self.max_iter = max_iter
        self.f = lambda x: x @ self.Q @ x / 2 + self.p @ x

    def solve(self, verbose=False):
        self.grad = self.Q @ self.alpha + self.p  # 梯度：Qa+p
        self.has_coverage = False  # 收敛标志

        n_iter = 0
        while n_iter < self.max_iter and not self.has_coverage:
            i, j = self.select_working_set()
            if self.has_coverage:
                continue

            self.update(i, j)
            n_iter += 1

            if verbose and n_iter % 100 == 99:
                print("{} iters".format(n_iter))
        obj = self.f(self.alpha)
        if verbose:
            print("optimize with {} iterations, objective value {}".format(
                n_iter,
                obj,
            ))
        if n_iter == self.max_iter:
            print("Not coverage, increase the max_iter")

        self.calculate_rho()
        return obj

    def calculate_rho(self):
        "In C-SVC, ε-SVR and one-class SVM ρ=-b"
        sv = np.logical_and(
            self.alpha > 0,
            self.alpha < self.C[0],
        )
        product = self.y * self.grad
        if sv.sum() > 0:
            self.rho = product[sv].sum() / sv.sum()
        else:
            ub_id = np.logical_or(
                np.logical_and(self.alpha == 0, self.y == -1),
                np.logical_and(self.alpha == self.C[0], self.y == 1),
            )
            lb_id = np.logical_or(
                np.logical_and(self.alpha == 0, self.y == 1),
                np.logical_and(self.alpha == self.C[1], self.y == -1),
            )
            self.rho = (product[lb_id].max() + product[ub_id].min()) / 2
        self.b = -self.rho

    def select_working_set(self):
        Iup = np.argwhere(
            np.logical_or(
                np.logical_and(self.alpha < self.C[0], self.y == 1),
                np.logical_and(self.alpha > 0, self.y == -1),
            )).reshape(-1)
        Ilow = np.argwhere(
            np.logical_or(
                np.logical_and(self.alpha < self.C[1], self.y == -1),
                np.logical_and(self.alpha > 0, self.y == 1),
            )).reshape(-1)

        product = -self.y * self.grad
        i, j = Iup[np.argmax(product[Iup])], Ilow[np.argmin(product[Ilow])]

        if product[i] - product[j] < self.eps:
            self.has_coverage = True  # 选不出来违反对，停止迭代
            i, j = None, None
            
        return i, j

    def update(self, i, j):
        alpha = self.alpha
        old_alpha_i, old_alpha_j = alpha[[i, j]]
        C_i, C_j = self.C[int(self.y[i] == -1)], self.C[int(self.y[i] == 1)]

        if self.y[i] != self.y[j]:
            quad_coef = self.Q[i, i] + self.Q[j, j] + 2 * self.Q[i, j]
            if quad_coef <= 0:
                quad_coef = 1e-12
            delta = (-self.grad[i] - self.grad[j]) / quad_coef

            diff = alpha[i] - alpha[j]
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

            if diff > C_i - C_j:
                if (alpha[i] > C_i):
                    alpha[i] = C_i
                    alpha[j] = C_i - diff

            else:
                if alpha[j] > C_j:
                    alpha[j] = C_j
                    alpha[i] = C_j + diff
        else:
            quad_coef = self.Q[i, i] + self.Q[j, j] - 2 * self.Q[i, j]
            if quad_coef <= 0:
                quad_coef = 1e-12
            delta = (self.grad[i] - self.grad[j]) / quad_coef

            sum = alpha[i] + alpha[j]
            alpha[i] -= delta
            alpha[j] += delta

            if sum > C_i:
                if alpha[i] > C_i:
                    alpha[i] = C_i
                    alpha[j] = sum - C_i

            else:
                if alpha[j] < 0:
                    alpha[j] = 0
                    alpha[i] = sum

            if sum > C_j:
                if alpha[j] > C_j:
                    alpha[j] = C_j
                    alpha[i] = sum - C_j

            else:
                if alpha[i] < 0:
                    alpha[i] = 0
                    alpha[j] = sum

        delta_i, delta_j = alpha[i] - old_alpha_i, alpha[j] - old_alpha_j
        self.grad += self.Q[i] * delta_i + self.Q[j] * delta_j
        return delta_i, delta_j

    def get_alpha(self):
        return self.alpha

    def get_rho(self):
        return self.rho

    def get_b(self):
        return self.b


class NuSolver(Solver):
    '''
    We want to solve the optimization problem using SMO:
    ```markdown
    min x^T·Q·x / 2 + px
    s.t. y^T·x = delta_1,
         e^T·x = delta_2, 
         0≤x_i≤C
    ```
    '''
    def __init__(
        self,
        l: int,
        Q,
        p,
        y,
        alpha,
        Cp: float,
        Cn: float,
        eps=1e-3,
        max_iter: int = 1000,
    ) -> None:
        super().__init__(l, Q, p, y, alpha, Cp, Cn, max_iter=max_iter, eps=eps)

    def solve(self, verbose=False):
        self.grad = self.Q @ self.alpha + self.p  # 梯度：Qa+p
        self.has_coverage = False  # 收敛标志
        obj = self.f(self.alpha)

        n_iter = 0
        while n_iter < self.max_iter and not self.has_coverage:
            '''
            Here we always select i, j with same label: y[i] = y[j]
            '''
            i, j = self.select_working_set()
            if self.has_coverage:
                continue

            self.update(i, j)

            obj_new = self.f(self.alpha)
            if abs(obj - obj_new) < 1e-5:
                self.has_coverage = True

            obj = obj_new
            n_iter += 1

            if verbose and n_iter % 100 == 0:
                print("{} iters".format(n_iter))

        if verbose:
            print("optimize with {} iterations, objective value {}".format(
                n_iter,
                obj,
            ))
        if n_iter == self.max_iter:
            print("Not coverage, increase the max_iter")

        self.calculate_rho()
        return obj

    def select_working_set(self):
        Iup = {
            i
            for i in range(self.l)
            if (self.alpha[i] < self.C[0] and self.y[i] == 1) or (
                self.alpha[i] > 0 and self.y[i] == -1)
        }
        Ilow = {
            j
            for j in range(self.l)
            if (self.alpha[j] < self.C[1] and self.y[j] == -1) or (
                self.alpha[j] > 0 and self.y[j] == 1)
        }

        Imp = [i for i in Iup if self.y[i] == 1]
        IMp = [i for i in Ilow if self.y[i] == 1]
        Imn = [i for i in Iup if self.y[i] == -1]
        IMn = [i for i in Ilow if self.y[i] == -1]

        m_p = max([-self.y[i] * self.grad[i] for i in Imp])
        M_p = min([-self.y[i] * self.grad[i] for i in IMp])
        m_n = max([-self.y[i] * self.grad[i] for i in Imn])
        M_n = min([-self.y[i] * self.grad[i] for i in IMn])

        if max(m_p - M_p, m_n - M_n) < self.eps:
            self.has_coverage = True
            return None, None

        for i_p in Imp:
            if -self.y[i_p] * self.grad[i_p] == m_p:
                break
        for j_p in IMp:
            if -self.y[j_p] * self.grad[j_p] == M_p:
                break
        for i_n in Imn:
            if -self.y[i_n] * self.grad[i_n] == m_n:
                break
        for j_n in IMn:
            if -self.y[j_n] * self.grad[j_n] == M_n:
                break

        if self.y[i_p] != self.y[j_p]:
            quad_coef = self.Q[i_p, i_p] + self.Q[j_p,
                                                  j_p] + 2 * self.Q[i_p, j_p]
            if quad_coef <= 0:
                quad_coef = 1e-12
            term_p = -(-self.grad[i_p] - self.grad[j_p])**2 / (2 * quad_coef)
        else:
            quad_coef = self.Q[i_p, i_p] + self.Q[j_p,
                                                  j_p] - 2 * self.Q[i_p, j_p]
            if quad_coef <= 0:
                quad_coef = 1e-12
            term_p = -(self.grad[i_p] - self.grad[j_p])**2 / (2 * quad_coef)

        if self.y[i_n] != self.y[j_n]:
            quad_coef = self.Q[i_n, i_n] + self.Q[j_n,
                                                  j_n] + 2 * self.Q[i_n, j_n]
            if quad_coef <= 0:
                quad_coef = 1e-12
            term_n = -(-self.grad[i_n] - self.grad[j_n])**2 / (2 * quad_coef)
        else:
            quad_coef = self.Q[i_n, i_n] + self.Q[j_n,
                                                  j_n] - 2 * self.Q[i_n, j_n]
            if quad_coef <= 0:
                quad_coef = 1e-12
            term_n = -(self.grad[i_n] - self.grad[j_p])**2 / (2 * quad_coef)

        i, j = (i_p, j_p) if term_p < term_n else (i_n, j_n)
        return i, j

    def update(self, i, j):
        '''
        y[i] = y[j]
        '''
        alpha = self.alpha
        old_alpha_i, old_alpha_j = alpha[[i, j]]
        C = self.C[int(self.y[i] == -1)]

        quad_coef = self.Q[i, i] + self.Q[j, j] - 2 * self.Q[i, j]
        if quad_coef <= 0:
            quad_coef = 1e-12
        delta = (self.grad[i] - self.grad[j]) / quad_coef

        sum = alpha[i] + alpha[j]
        alpha[i] -= delta
        alpha[j] += delta

        if sum > C:
            if alpha[i] > C:
                alpha[i] = C
                alpha[j] = sum - C

        else:
            if alpha[j] < 0:
                alpha[j] = 0
                alpha[i] = sum

        if sum > C:
            if alpha[j] > C:
                alpha[j] = C
                alpha[i] = sum - C

        else:
            if alpha[i] < 0:
                alpha[i] = 0
                alpha[j] = sum

        delta_alpha_i = alpha[i] - old_alpha_i
        delta_alpha_j = alpha[j] - old_alpha_j

        self.grad += self.Q[[i, j]].T @ np.array([
            delta_alpha_i,
            delta_alpha_j,
        ])

    def calculate_rho(self):
        Cp, Cn = self.C
        pos_sv = np.logical_and(
            np.logical_and(
                self.alpha > 0,
                self.alpha < Cp,
            ),
            self.y == 1,
        )
        if pos_sv.sum() == 0:
            r1 = (np.max(self.grad[self.alpha == Cp and self.y == 1]) +
                  np.min(self.grad[self.alpha == 0 and self.y == 1])) / 2
        else:
            r1 = np.sum(self.grad[pos_sv]) / pos_sv.sum()

        neg_sv = np.logical_and(
            np.logical_and(
                self.alpha > 0,
                self.alpha < Cn,
            ),
            self.y == -1,
        )
        if neg_sv.sum() == 0:
            r2 = (np.max(self.grad[self.alpha == Cn and self.y == -1]) +
                  np.min(self.grad[self.alpha == 0 and self.y == -1])) / 2
        else:
            r2 = np.sum(self.grad[neg_sv]) / neg_sv.sum()
        self.rho = (r1 + r2) / 2
        self.b = -(r1 - r2) / 2

    def get_rho(self):
        return self.rho

    def get_b(self):
        return self.b

    def get_alpha(self):
        return super().get_alpha()
