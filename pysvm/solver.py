import numpy as np
from functools import lru_cache


class Solver:
    r'''SMO算法求解器，迭代求解下面的问题:

    .. math:: \min_{\pmb\alpha}\quad&\frac12\pmb\alpha^T\pmb Q\pmb\alpha+\pmb p^T\pmb\alpha\\
        \text{s.t.}\quad&\pmb y^T\pmb\alpha=0\\
        &0\leq\alpha_i\leq C,i=1,\cdots,l
    
    Parameters
    ----------
    Q : numpy.ndarray
        优化问题中的 :math:`\pmb Q` 矩阵；
    p : numpy.ndarray
        优化问题中的 :math:`\pmb p` 向量；
    y : numpy.ndarray
        优化问题中的 :math:`\pmb y` 向量；
    C : float
        优化问题中的 :math:`C` 变量；
    tol : float, default=1e-5
        变量选择的tolerance，默认为1e-5.
    '''
    def __init__(self,
                 Q: np.ndarray,
                 p: np.ndarray,
                 y: np.ndarray,
                 C: float,
                 tol: float = 1e-5) -> None:
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
        r'''工作集选择，这里采用一阶选择:

        .. math:: \pmb{I}_{up}(\pmb\alpha)&=\{t|\alpha_t<C,y_t=1\text{ or }\alpha_t>0,y_t=-1\}\\
                 \pmb{I}_{low}(\pmb\alpha)&=\{t|\alpha_t<C,y_t=-1\text{ or }\alpha_t>0,y_t=1\}\\
                 i&\in\arg\max_{t}\{-y_t\nabla_tf(\pmb\alpha)|t\in\pmb{I}_{up}(\pmb\alpha)\}\\
                 j&\in\arg\max_{t}\{-y_t\nabla_tf(\pmb\alpha)|t\in\pmb{I}_{low}(\pmb\alpha)\}\\
        '''
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

        find_fail = False
        try:
            i = Iup[np.argmax(self.neg_y_grad[Iup])]
            j = Ilow[np.argmin(self.neg_y_grad[Ilow])]
        except:
            find_fail = True

        if find_fail or self.neg_y_grad[i] - self.neg_y_grad[j] < self.tol:
            return -1, -1
        return i, j

    def update(self, i: int, j: int, func=None):
        '''变量更新，在保证变量满足约束的条件下对两变量进行更新

        参考<https://welts.xyz/2021/07/11/libsmo/>.
        '''
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

    def calculate_rho(self) -> float:
        r'''计算偏置项
        
        .. math:: \rho=\dfrac{\sum_{i:0<\alpha_i<C}y_i\nabla_if(\pmb\alpha)}{|\{i\vert0<\alpha_i<C\}|}

        如果不存在满足条件的支持向量，那么

        .. math:: -M(\pmb\alpha)&=\max\{y_i\nabla_if(\pmb\alpha)|\alpha_i=0,y_i=-1\text{ or }\alpha_i=C,y_i=1\}\\
                  -m(\pmb\alpha)&=\max\{y_i\nabla_if(\pmb\alpha)|\alpha_i=0,y_i=1\text{ or }\alpha_i=C,y_i=-1\}\\
                  \rho&=-\dfrac{M(\pmb\alpha)+m(\pmb\alpha)}{2}
        '''
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

    def get_Q(self, i: int, func=None):
        '''获取核矩阵的第i行/列，即
        
        .. math:: [K(\pmb x_1, \pmb x_i),\cdots,K(\pmb x_l, \pmb x_i)]
        '''
        return self.Q[i]


class SolverWithCache(Solver):
    '''带核函数缓存机制的Solver：使用LRU缓存来计算Q矩阵，从而不需要计算Q矩阵，从而带来存储的问题。
    
    Parameters
    ----------
    p : numpy.ndarray
        优化问题中的 :math:`\pmb p` 向量；
    y : numpy.ndarray
        优化问题中的 :math:`\pmb y` 向量；
    C : float
        优化问题中的 :math:`C` 变量；
    tol : float, default=1e-5
        变量选择的tolerance，默认为1e-5.
    cache_size : int, default=256
        LRU缓存数.

    See also
    --------
    Solver
    '''
    cache_size = 256

    def __init__(self,
                 p: np.ndarray,
                 y: np.ndarray,
                 C: float,
                 tol: float = 1e-5,
                 cache_size: int = 256) -> None:
        super().__init__(None, p, y, C, tol)
        cache_size = cache_size

    def working_set_select(self):
        return super().working_set_select()

    def update(self, i: int, j: int, func=None):
        return super().update(i, j, func=func)

    def calculate_rho(self):
        return super().calculate_rho()

    @lru_cache(cache_size)
    def get_Q(self, i, func):
        return func(i)


class NuSolver(Solver):
    r'''SMO算法求解器，迭代求解下面的问题

    .. math:: \min_{\pmb\alpha}\quad&\frac12\pmb\alpha^T\pmb Q\pmb\alpha+\pmb p^T\pmb\alpha\\
        \text{s.t.}\quad&\pmb y^T\pmb\alpha=0\\
        &\pmb e^T\pmb\alpha=t\\
        &0\leq\alpha_i\leq C,i=1,\cdots,l
    
    Parameters
    ----------
    Q : numpy.ndarray
        优化问题中的 :math:`\pmb Q` 矩阵；
    p : numpy.ndarray
        优化问题中的 :math:`\pmb p` 向量；
    y : numpy.ndarray
        优化问题中的 :math:`\pmb y` 向量；
    t : float
        优化问题中的 :math:`t` 变量；
    C : float
        优化问题中的 :math:`C` 变量；
    tol : float, default=1e-5
        变量选择的tolerance，默认为1e-5.
    '''
    def __init__(self,
                 Q: np.ndarray,
                 p: np.ndarray,
                 y: np.ndarray,
                 t: float,
                 C: float,
                 tol: float = 1e-5) -> None:
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
        r'''采用一阶信息进行变量选择，选出两对变量，然后比较估计下降量，选出能让目标函数下降更大的变量对

        存在只能选出一对的情况，此时我们不比较，直接返回这对变量；
        如果两对都选不到，停止迭代。
        '''
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
            j_p = IMp[np.argmin(self.neg_y_grad[IMp])]
        except:
            pos_fail = True

        try:
            Imn = Iup[self.y[Iup] < 0]
            IMn = Ilow[self.y[Ilow] < 0]
            i_n = Imn[np.argmax(self.neg_y_grad[Imn])]
            j_n = IMn[np.argmin(self.neg_y_grad[IMn])]
        except:
            neg_fail = True

        if pos_fail and neg_fail:
            return -1, -1, -1, -1
        elif pos_fail:
            return i_n, j_n, self.get_Q(i_n, func), self.get_Q(j_n, func)
        elif neg_fail:
            return i_p, j_p, self.get_Q(i_p, func), self.get_Q(j_p, func)
        else:
            grad_diff_p = self.neg_y_grad[i_p] - self.neg_y_grad[j_p]
            Q_ip = self.get_Q(i_p, func)
            quad_coef = self.QD[i_p] + self.QD[j_p] - 2 * Q_ip[j_p]
            if quad_coef <= 0:
                quad_coef = 1e-12
            obj_diff_p = -grad_diff_p**2 / quad_coef

            grad_diff_n = self.neg_y_grad[i_n] - self.neg_y_grad[j_n]
            Q_in = self.get_Q(i_n, func)
            quad_coef = self.QD[i_n] + self.QD[j_n] - 2 * Q_in[j_n]
            if quad_coef <= 0:
                quad_coef = 1e-12
            obj_diff_n = -grad_diff_n**2 / quad_coef
            if obj_diff_p < obj_diff_n:
                return i_p, j_p, Q_ip, self.get_Q(j_p, func)
            return i_n, j_n, Q_in, self.get_Q(j_n, func)

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
    '''带核函数缓存机制的NuSolver：使用LRU缓存来计算Q矩阵，从而不需要计算Q矩阵，从而带来存储的问题。
    
    Parameters
    ----------
    Q : numpy.ndarray
        优化问题中的 :math:`\pmb Q` 矩阵；
    p : numpy.ndarray
        优化问题中的 :math:`\pmb p` 向量；
    y : numpy.ndarray
        优化问题中的 :math:`\pmb y` 向量；
    t : float
        优化问题中的 :math:`t` 变量；
    C : float
        优化问题中的 :math:`C` 变量；
    tol : float, default=1e-5
        变量选择的tolerance，默认为1e-5.

    See also
    --------
    NuSolver
    '''
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
