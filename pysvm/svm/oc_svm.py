import numpy as np

from .svc import BiNuSVC
from ..solver import Solver, SolverWithCache


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
            return kernel_func(X, X[i:i + 1]).flatten()

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
