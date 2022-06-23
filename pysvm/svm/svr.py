import numpy as np
from sklearn.metrics import r2_score

from .svc import BiLinearSVC, BiKernelSVC
from ..solver import Solver, SolverWithCache, NuSolver, NuSolverWithCache


class LinearSVR(BiLinearSVC):
    r'''线性SVM回归(SVR)

    原对偶问题

    .. math:: \min_{\pmb{\alpha},\pmb{\alpha}^*}\quad&\dfrac12(\pmb{\alpha}-\pmb{\alpha}^*)^\top Q(\pmb{\alpha}-\pmb{\alpha}^*)+\varepsilon\sum_{i=1}^l(\alpha_i+\alpha_i^*)+\sum_{i=1}^l z_i({\alpha}_i-{\alpha}_i^*)\\
            \text{s.t.}\quad&\pmb e^\top(\pmb{\alpha}-\pmb{\alpha}^*)=0\\
            &0\leqslant\alpha_i,\alpha^*_i\leqslant C,i=1,\cdots ,l

    我们将其变成单变量优化问题，然后使用SMO求解，参考https://welts.xyz/2021/09/16/svr/。得到决策边界

    .. math:: f(\pmb x)=\sum_{i=1}^l(-\alpha_i+\alpha_i^*)\pmb x_i^T\pmb x-\rho

    Parameters
    ----------
    C : float, default=1
        SVM的正则化参数，默认为1；
    eps : float, default=0
        :math:`\varepsilon`-hinge损失的参数；
    max_iter : int, default=1000
        SMO算法迭代次数，默认1000；
    tol : float, default=1e-5
        SMO算法的容忍度参数，默认1e-5；
    cache_size : int, default=256
        lru缓存大小，默认256，如果为0则不使用缓存，计算Q矩阵然后求解.
    '''
    def __init__(self,
                 C: float = 1.,
                 eps: float = 0.,
                 max_iter: int = 1000,
                 tol: float = 1e-5,
                 cache_size: int = 256) -> None:
        super().__init__(C, max_iter, tol, cache_size)
        self.eps = eps

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''训练模型

        Parameters
        ----------
        X : np.ndarray
            训练集特征;
        y : np.array
            训练集target

        Return
        ------
        self : LinearSVR
        '''
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
            Q2 = np.r_[Q, -Q]
            solver = Solver(np.c_[Q2, -Q2], p, y, self.C, self.tol)
        else:
            solver = SolverWithCache(p, y, self.C, self.tol, self.cache_size)

        def func(i):
            if i < l:
                Qi = np.matmul(X, X[i])
            else:
                Qi = -np.matmul(X, X[i - l])
            return np.r_[Qi, -Qi]

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

    def decision_function(self, X: np.ndarray):
        return super().decision_function(X)

    def predict(self, X: np.ndarray):
        '''预测函数，输出预测值'''
        return self.decision_function(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        '''评估函数，给定特征和标签，输出r2系数'''
        return r2_score(y, self.predict(X))


class KernelSVR(BiKernelSVC):
    r'''核支持向量回归
    
    Parameters
    ----------
    C : float, default=1
        SVM的正则化参数，默认为1；
    eps : float, default=0
        :math:`\varepsilon`-hinge损失的参数；
    kernel : {"linear", "poly", "rbf", "sigmoid"}, default="rbf"
        核函数，默认径向基函数(RBF)；
    degree : float, default=3
        多项式核的次数，默认3；
    gamma : {"scale", "auto", float}, default="scale"
        rbf、ploy和sigmoid核的参数 :math:`\gamma`，如果用'scale'，那么就是1 / (n_features * X.var())，如果用'auto'，那么就是1 / n_features；
    coef0 : float, default=0.
        核函数中的独立项。它只在"poly"和"sigmoid"中有意义；
    max_iter : int, default=1000
        SMO算法迭代次数，默认1000；
    rff : bool, default=False
        是否采用随机傅里叶特征，默认为False；
    D : int, default=1000
        随机傅里叶特征的采样次数，默认为1000；
    tol : float, default=1e-5
        SMO算法的容忍度参数，默认1e-5；
    cache_size : int, default=256
        lru缓存大小，默认256，如果为0则不使用缓存，计算Q矩阵然后求解.
    '''
    def __init__(self,
                 C: int = 1.,
                 eps: float = 0.,
                 kernel: str = 'rbf',
                 degree: float = 3,
                 gamma: float = 'scale',
                 coef0: float = 0.,
                 max_iter: int = 1000,
                 rff: bool = False,
                 D: int = 1000,
                 tol: float = 1e-5,
                 cache_size: int = 256) -> None:
        super().__init__(C, kernel, degree, gamma, coef0, max_iter, rff, D,
                         tol, cache_size)
        self.eps = eps

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, z = np.array(X), np.array(y)
        l, self.n_features = X.shape

        y = np.empty(2 * l)
        y[:l], y[l:] = 1., -1.

        p = np.ones(2 * l) * self.eps
        p[:l] -= z
        p[l:] += z

        kernel_func = self.register_kernel(X.std())

        if self.cache_size == 0:
            Q = np.matmul(X, X.T)
            Q2 = np.r_[Q, -Q]
            solver = Solver(np.c_[Q2, -Q2], p, y, self.C, self.tol)
        else:
            solver = SolverWithCache(p, y, self.C, self.tol, self.cache_size)

        def func(i):
            if i < l:
                Qi = kernel_func(X, X[i:i + 1]).flatten()
            else:
                Qi = -kernel_func(X, X[i - l:i - l + 1]).flatten()
            return np.r_[Qi, -Qi]

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

    def predict(self, X: np.ndarray):
        '''预测函数，输出预测值'''
        return self.decision_function(np.array(X))

    def score(self, X: np.ndarray, y: np.ndarray):
        '''评估函数，给定特征和标签，输出r2系数'''
        return r2_score(y, self.predict(X))


class NuSVR(KernelSVR):
    r'''NuSVM回归

    对偶问题求解

    .. math:: \min_{\pmb{\alpha},\pmb{\alpha}^*}\quad&\dfrac12(\pmb{\alpha}-\pmb{\alpha}^*)^\top Q(\pmb{\alpha}-\pmb{\alpha}^*)+\pmb z^\top({\pmb\alpha}-{\pmb\alpha}^*)\\
            \text{s.t.}\quad&\pmb e^\top(\pmb{\alpha}-\pmb{\alpha}^*)=0,\pmb e^\top(\pmb\alpha+\pmb\alpha_i^*)\leqslant C\nu\\
            &0\leqslant\alpha_i,\alpha^*_i\leqslant C/l,i=1,\cdots ,l

    处理方式和LinearSVR中的类似.

    Parameters
    ----------
    C : float, default=1
        SVM的正则化参数，默认为1；
    nu : float, default=0.5
        NuSVM的参数，控制支持向量的数量；
    kernel : {"linear", "poly", "rbf", "sigmoid"}, default="rbf"
        核函数，默认径向基函数(RBF)；
    degree : float, default=3
        多项式核的次数，默认3；
    gamma : {"scale", "auto", float}, default="scale"
        rbf、ploy和sigmoid核的参数 :math:`\gamma`，如果用'scale'，那么就是1 / (n_features * X.var())，如果用'auto'，那么就是1 / n_features；
    coef0 : float, default=0.
        核函数中的独立项。它只在"poly"和"sigmoid"中有意义；
    max_iter : int, default=1000
        SMO算法迭代次数，默认1000；
    rff : bool, default=False
        是否采用随机傅里叶特征，默认为False；
    D : int, default=1000
        随机傅里叶特征的采样次数，默认为1000；
    tol : float, default=1e-5
        SMO算法的容忍度参数，默认1e-5；
    cache_size : int, default=256
        lru缓存大小，默认256，如果为0则不使用缓存，计算Q矩阵然后求解.
    '''
    def __init__(self,
                 C: float = 1.,
                 nu: float = 0.5,
                 kernel: str = 'rbf',
                 degree: float = 3,
                 gamma: float = 'scale',
                 coef0: float = 0.,
                 max_iter: int = 1000,
                 rff: bool = False,
                 D: int = 1000,
                 tol: float = 1e-5,
                 cache_size: int = 256) -> None:
        super().__init__(C, 0, kernel, degree, gamma, coef0, max_iter, rff, D,
                         tol, cache_size)
        self.nu = nu

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, z = np.array(X), np.array(y)
        l, self.n_features = X.shape

        y = np.empty(2 * l)
        y[:l], y[l:] = 1, -1

        p = np.empty(2 * l)
        p[:l], p[l:] = -z, z

        kernel_func = self.register_kernel(X.std())

        def func(i):
            if i < l:
                Qi = kernel_func(X, X[i:i + 1]).flatten()
            else:
                Qi = -kernel_func(X, X[i - l:i - l + 1]).flatten()
            return np.r_[Qi, -Qi]

        if self.cache_size == 0:
            Q = kernel_func(X, X)
            Q2 = np.r_[Q, -Q]
            solver = NuSolver(np.c_[Q2, -Q2], p, y, self.C * l * self.nu, self.C, self.tol)
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

    def predict(self, X: np.ndarray):
        return super().predict(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        return super().score(X, y)
