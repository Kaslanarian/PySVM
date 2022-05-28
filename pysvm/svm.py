import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from .rff import NormalRFF
from .solver import Solver, SolverWithCache, NuSolver, NuSolverWithCache


class BiLinearSVC(BaseEstimator):
    r'''二分类线性SVM，该类被多分类LinearSVC继承，所以不需要使用它。
    
    通过求解对偶问题

    .. math:: \min_{\pmb\alpha}\quad&\dfrac12\pmb\alpha^\top Q\pmb\alpha-\pmb{e}^\top\pmb{\alpha}\\
        \text{s.t.}\quad& \pmb{y}^\top\pmb\alpha=0,\\
        &0\leqslant\alpha_i\leqslant C,i=1,\cdots ,l

    得到决策边界

    .. math:: f(\pmb x)=\sum_{i=1}^ly_i\alpha_i\pmb x_i^T\pmb x-\rho

    Parameters
    ----------
    C : float, default=1
        SVM的正则化参数，默认为1；
    max_iter : int, default=1000
        SMO算法迭代次数，默认1000；
    tol : float, default=1e-5
        SMO算法的容忍度参数，默认1e-5；
    cache_size : int, default=256
        lru缓存大小，默认256，如果为0则不使用缓存，计算Q矩阵然后求解.
    '''
    def __init__(self,
                 C: float = 1.,
                 max_iter: int = 1000,
                 tol: float = 1e-5,
                 cache_size: int = 256) -> None:
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.cache_size = cache_size

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''训练模型

        Parameters
        ----------
        X : np.ndarray
            训练集特征;
        y : np.array
            训练集标签，建议0为负标签，1为正标签.
        '''
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
            return y * np.matmul(X, X[i]) * y[i]

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

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        '''决策函数，输出预测值'''
        return np.matmul(self.coef_[0], np.array(X).T) - self.coef_[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''预测函数，输出预测标签(0-1)'''
        return (self.decision_function(np.array(X)) >= 0).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        '''评估函数，给定特征和标签，输出正确率'''
        return accuracy_score(y, self.predict(X))


class LinearSVC(BiLinearSVC):
    r'''多分类线性SVM，使用sklearn的multiclass模块实现了多分类。

    Parameters
    ----------
    C : float, default=1
        SVM的正则化参数，默认为1；
    max_iter : int, default=1000
        SMO算法迭代次数，默认1000；
    tol : float, default=1e-5
        SMO算法的容忍度参数，默认1e-5；
    cache_size : int, default=256
        lru缓存大小，默认256，如果为0则不使用缓存，计算Q矩阵然后求解；
    multiclass : {"ovr", "ovo"}, default="ovr"
        多分类策略，ovr(一对多)或ovo(一对一)，默认ovr；
    n_jobs : int, default=None
        是否采用多核，使用多少CPU并行，默认不采用。
    '''
    def __init__(self,
                 C: float = 1.,
                 max_iter: int = 1000,
                 tol: float = 1e-5,
                 cache_size: int = 256,
                 multiclass: str = "ovr",
                 n_jobs=None) -> None:
        super().__init__(C, max_iter, tol, cache_size)
        self.multiclass = multiclass
        self.n_jobs = n_jobs
        params = {
            "estimator": BiLinearSVC(C, max_iter, tol, cache_size),
            "n_jobs": n_jobs,
        }
        self.multiclass_model: OneVsOneClassifier = {
            "ovo": OneVsOneClassifier(**params),
            "ovr": OneVsRestClassifier(**params),
        }[multiclass]

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''训练模型

        Parameters
        ----------
        X : np.ndarray
            训练集特征;
        y : np.array
            训练集标签，建议0为负标签，1为正标签.

        Return
        ------
        self : LinearSVC
        '''
        self.multiclass_model.fit(X, y)
        return self

    def decision_function(self, X: np.ndarray):
        return self.multiclass_model.decision_function(X)

    def predict(self, X: np.ndarray):
        return self.multiclass_model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        return self.multiclass_model.score(X, y)


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

    def decision_function(self, X: np.ndarray):
        return super().decision_function(X)

    def predict(self, X: np.ndarray):
        '''预测函数，输出预测值'''
        return self.decision_function(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        '''评估函数，给定特征和标签，输出r2系数'''
        return r2_score(y, self.predict(X))


class BiKernelSVC(BiLinearSVC):
    r'''二分类核SVM，该类被多分类KernelSVC继承，所以不需要使用它。优化问题与BiLinearSVC相同，只是Q矩阵定义不同。

    此时的决策边界

    .. math:: f(\pmb x)=\sum_{i=1}^ly_i\alpha_i K(\pmb x_i, \pmb x)-\rho

    Parameters
    ----------
    C : float, default=1
        SVM的正则化参数，默认为1；
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
                 kernel: str = 'rbf',
                 degree: float = 3,
                 gamma: str = 'scale',
                 coef0: float = 0,
                 max_iter: int = 1000,
                 rff: bool = False,
                 D: int = 1000,
                 tol: float = 1e-5,
                 cache_size: int = 256) -> None:
        super().__init__(C, max_iter, tol, cache_size)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.rff = rff
        self.D = D

    def register_kernel(self, std: float):
        '''注册核函数
        
        Parameters
        ----------
        std : 输入数据的标准差，用于rbf='scale'的情况
        '''
        if type(self.gamma) == str:
            gamma = {
                'scale': 1 / (self.n_features * std),
                'auto': 1 / self.n_features,
            }[self.gamma]
        else:
            gamma = self.gamma

        if self.rff:
            rff = NormalRFF(gamma, self.D).fit(np.ones((1, self.n_features)))
            rbf_func = lambda x, y: np.matmul(rff.transform(x),
                                              rff.transform(y).T)
        else:
            rbf_func = lambda x, y: np.exp(-gamma * (
                (x**2).sum(1, keepdims=True) +
                (y**2).sum(1) - 2 * np.matmul(x, y.T)))

        degree = self.degree
        coef0 = self.coef0
        return {
            "linear": lambda x, y: np.matmul(x, y.T),
            "poly": lambda x, y: (gamma * np.matmul(x, y.T) + coef0)**degree,
            "rbf": rbf_func,
            "sigmoid": lambda x, y: np.tanh(gamma * np.matmul(x, y.T) + coef0)
        }[self.kernel]

    def fit(self, X: np.ndarray, y: np.ndarray):
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return super().score(X, y)


class KernelSVC(LinearSVC, BiKernelSVC):
    r'''多分类核SVM。

    Parameters
    ----------
    C : float, default=1
        SVM的正则化参数，默认为1；
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
    multiclass : {"ovr", "ovo"}, default="ovr"
        多分类策略，ovr(一对多)或ovo(一对一)，默认ovr；
    n_jobs : int, default=None
        是否采用多核，使用多少CPU并行，默认不采用。
    '''
    def __init__(self,
                 C: float = 1.,
                 kernel: str = 'rbf',
                 degree: float = 3,
                 gamma: float = 'scale',
                 coef0: float = 0.,
                 max_iter: int = 1000,
                 rff: bool = False,
                 D: int = 1000,
                 tol: float = 1e-5,
                 cache_size: int = 256,
                 multiclass: str = "ovr",
                 n_jobs: int = None) -> None:
        super().__init__(C, max_iter, tol, cache_size)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.rff = rff
        self.D = D
        params = {
            "estimator":
            BiKernelSVC(C, kernel, degree, gamma, coef0, max_iter, rff, D, tol,
                        cache_size),
            "n_jobs":
            n_jobs,
        }
        self.multiclass_model = {
            "ovo": OneVsOneClassifier(**params),
            "ovr": OneVsRestClassifier(**params),
        }[multiclass]

    def fit(self, X: np.ndarray, y: np.ndarray):
        return super().fit(X, y)

    def decision_function(self, X: np.ndarray):
        return super().decision_function(X)

    def predict(self, X: np.ndarray):
        return super().predict(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        return super().score(X, y)


class KernelSVR(BiKernelSVC):
    '''核支持向量回归
    
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

    def predict(self, X: np.ndarray):
        '''预测函数，输出预测值'''
        return self.decision_function(np.array(X))

    def score(self, X: np.ndarray, y: np.ndarray):
        '''评估函数，给定特征和标签，输出r2系数'''
        return r2_score(y, self.predict(X))


class BiNuSVC(BiKernelSVC):
    r'''二分类NuSVM，通过参数 :math:`\nu`来控制支持向量的数量。
    
    通过求解对偶问题

    .. math:: \min_{\pmb\alpha}\quad&\dfrac12\pmb\alpha^\top Q\pmb\alpha\\
            \text{s.t.}\quad&0\leqslant\alpha_i\leqslant\frac{1}{l},,i=1,\cdots,l\\
            &\pmb{e}^\top\pmb\alpha\geqslant \nu,\pmb y^\top\pmb{\alpha}=0
    
    得到决策边界

    .. math:: f(\pmb x)=\sum_{i=1}^ly_i\alpha_i\pmb K(\pmb x_i,\pmb x)-\rho

    Parameters
    ----------
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
        super().__init__(1, kernel, degree, gamma, coef0, max_iter, rff, D,
                         tol, cache_size)
        self.nu = nu

    def fit(self, X: np.ndarray, y: np.ndarray):
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

    def predict(self, X: np.ndarray):
        return super().predict(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        return super().score(X, y)


class NuSVC(KernelSVC, BiNuSVC):
    '''多分类NuSVM
    
    Parameters
    ----------
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
    multiclass : {"ovr", "ovo"}, default="ovr"
        多分类策略，ovr(一对多)或ovo(一对一)，默认ovr；
    n_jobs : int, default=None
        是否采用多核，使用多少CPU并行，默认不采用。
    '''
    def __init__(self,
                 nu: float = 0.5,
                 kernel: str = 'rbf',
                 degree: float = 3,
                 gamma: float = 'scale',
                 coef0: float = 0.,
                 max_iter: int = 1000,
                 rff: bool = False,
                 D: int = 1000,
                 tol: float = 1e-5,
                 cache_size: int = 256,
                 multiclass: str = "ovr",
                 n_jobs: int = None) -> None:
        super().__init__(1, kernel, degree, gamma, coef0, max_iter, rff, D,
                         tol, cache_size, multiclass, n_jobs)
        self.nu = nu
        params = {
            "estimator":
            BiNuSVC(nu, kernel, degree, gamma, coef0, max_iter, rff, D, tol,
                    cache_size),
            "n_jobs":
            n_jobs,
        }
        self.multiclass_model: OneVsOneClassifier = {
            "ovo": OneVsOneClassifier(**params),
            "ovr": OneVsRestClassifier(**params),
        }[multiclass]

    def fit(self, X: np.ndarray, y: np.ndarray):
        return super().fit(X, y)

    def predict(self, X: np.ndarray):
        return super().predict(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        return super().score(X, y)


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

    def predict(self, X: np.ndarray):
        return super().predict(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        return super().score(X, y)


class OneClassSVM(BiNuSVC):
    r'''OneClassSVM(OC_SVM)，单类SVM，用于异常检测
    
    求解对偶问题

    .. math:: \min_{\pmb\alpha}\quad&\dfrac{1}{2}\pmb\alpha^\top Q\pmb\alpha\\
                \text{s.t.}\quad&0\le\alpha_i\le1/(\nu l),i=1,\cdots l\\
                &\pmb e^\top\alpha=1

    得到判别式

    .. math:: f(\pmb x)=\text{sgn}(\sum_{i=1}^ly_i\alpha_i\pmb K(\pmb x_i,\pmb x)-\rho)

    Parameters
    ----------
    nu : float, default=0.5
        控制支持向量的数量的参数；
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
        super().__init__(nu, kernel, degree, gamma, coef0, max_iter, rff, D,
                         tol, cache_size)

    def fit(self, X: np.ndarray):
        '''训练函数，注意到OC_SVM是无监督学习，所以输入无标签
        
        Parameters
        ----------
        X : np.ndarray
            训练特征数据
        '''
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
            solver.neg_y_grad = -y * np.matmul(Q, solver.alpha)

        else:
            solver = SolverWithCache(p, y, 1, self.tol, self.cache_size)
            solver.alpha = alpha
            for i in range(l):
                solver.neg_y_grad[i] -= y[i] * np.matmul(func(i), solver.alpha)

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

    def predict(self, X: np.ndarray):
        '''判别数据是否异常，正常为1，异常为-1'''
        pred = np.sign(self.decision_function(X))
        pred[pred == 0] = -1
        return pred

    def score(self, X, y):
        '''无监督问题不存在评估函数，因此调用该函数会引发异常'''
        raise NotImplementedError
