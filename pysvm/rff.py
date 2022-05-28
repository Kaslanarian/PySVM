import numpy as np
from sklearn.base import TransformerMixin


class NormalRFF(TransformerMixin):
    '''随机傅里叶特征逼近RBF核函数'''
    def __init__(self, gamma=1, D=1000) -> None:
        super().__init__()
        self.gamma = gamma
        self.D = D

    def fit(self, X: np.ndarray):
        self.n_features = np.array(X).shape[1]
        self.w = np.sqrt(self.gamma * 2) * np.random.randn(
            self.D, self.n_features)
        self.b = np.random.uniform(0, 2 * np.pi, self.D)
        return self

    def transform(self, X: np.ndarray):
        return np.sqrt(2 / self.D) * np.cos(np.matmul(X, self.w.T) + self.b)

    def fit_transform(self, X: np.ndarray):
        return self.fit(X).transform(X)
