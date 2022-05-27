import numpy as np
from sklearn.base import TransformerMixin


class NormalRFF(TransformerMixin):
    def __init__(self, gamma=1, D=10000) -> None:
        super().__init__()
        self.gamma = gamma
        self.D = D

    def fit(self, X):
        self.n_features = np.array(X).shape[1]
        self.w = np.sqrt(self.gamma * 2) * np.random.normal(size=(
            self.D,
            self.n_features,
        ))
        self.b = 2 * np.pi * np.random.rand(self.D)
        return self

    def transform(self, X):
        return np.sqrt(2 / self.D) * np.cos(np.array(X) @ self.w.T + self.b)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
