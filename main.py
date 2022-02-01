from svr import *
from svc import MultiLinearSVC, MultiKernelSVC, MultiNuSVC
from time import time

if __name__ == "__main__":
    from sklearn.datasets import *
    from sklearn.preprocessing import StandardScaler
    from time import time

    # X, y = make_classification(n_samples=5000)
    clf = NuSVR

    # X, y = load_iris(return_X_y=True)
    # X = StandardScaler().fit(X).transform(X)
    # print(clf().fit(X, y).score(X, y))

    # X, y = load_breast_cancer(return_X_y=True)
    # X = StandardScaler().fit(X).transform(X)
    # print(clf().fit(X, y).score(X, y))

    # X, y = load_wine(return_X_y=True)
    # X = StandardScaler().fit(X).transform(X)
    # print(clf().fit(X, y).score(X, y))

    # X, y = load_digits(return_X_y=True)
    # X = StandardScaler().fit(X).transform(X)
    # print(clf().fit(X, y).score(X, y))

    X, y = load_boston(return_X_y=True)
    X = StandardScaler().fit(X).transform(X)
    print(clf(C=100).fit(X, y).score(X, y))

    X, y = load_diabetes(return_X_y=True)
    X = StandardScaler().fit(X).transform(X)
    print(clf(C=100).fit(X, y).score(X, y))