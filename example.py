import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from svc import LinearSVC, KernelSVC, NuSVC
from svr import LinearSVR, KernelSVR, NuSVR
from one_class import OneClassSVM

RANDOM_STATE = 2021


def visual_svc_test():
    X, y = make_classification(
        n_features=2,
        n_classes=3,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        random_state=RANDOM_STATE,
    )
    for y_ in np.unique(y):
        data = X[y == y_]
        plt.scatter(data[:, 0], data[:, 1], label="%d" % y_)

    plt.legend()
    plt.show()

    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        train_size=0.7,
        random_state=RANDOM_STATE,
    )
    print(("{}SVC's perf : {:.2f}%\n" * 3).format(
        "Linear",
        LinearSVC().fit(train_X, train_y).score(test_X, test_y) * 100,
        "Kernel",
        KernelSVC().fit(train_X, train_y).score(test_X, test_y) * 100,
        "    Nu",
        NuSVC().fit(train_X, train_y).score(test_X, test_y) * 100,
    ))


def dataset_svc_test(dataset="breast_cancer"):
    X, y = {
        "iris": load_iris,
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "digits": load_digits,
    }[dataset](return_X_y=True)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        train_size=0.7,
        random_state=RANDOM_STATE,
    )
    print(("dataset : load_{}\n" + "{}SVC's perf : {:.2f}%\n" * 3).format(
        dataset,
        "Linear",
        LinearSVC(n_jobs=4).fit(train_X, train_y).score(test_X, test_y) * 100,
        "Kernel",
        KernelSVC(n_jobs=4).fit(train_X, train_y).score(test_X, test_y) * 100,
        "    Nu",
        NuSVC(n_jobs=4).fit(train_X, train_y).score(test_X, test_y) * 100,
    ))


def dataset_svc_rff_test(dataset="digits"):
    X, y = {
        "iris": load_iris,
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "digits": load_digits,
    }[dataset](return_X_y=True)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        train_size=0.7,
        random_state=RANDOM_STATE,
    )
    print("dataset : load_{}".format(dataset))
    print("RBF-KernelSVC's perf : {:.2f}%, RFF-KernelSVC's perf : {:.2f}%".
          format(
              KernelSVC(n_jobs=4).fit(train_X, train_y).score(test_X, test_y) *
              100,
              KernelSVC(n_jobs=4, rff=True).fit(train_X, train_y).score(
                  test_X, test_y) * 100,
          ))
    print("RBF-NuSVC's perf : {:.2f}%, RFF-NuSVC's perf : {:.2f}%\n".format(
        NuSVC(n_jobs=4).fit(train_X, train_y).score(test_X, test_y) * 100,
        NuSVC(n_jobs=4, rff=True).fit(train_X, train_y).score(test_X, test_y) *
        100,
    ))


def visual_svr_test():
    X, y = make_regression(n_features=1,
                           noise=3,
                           n_samples=50,
                           random_state=RANDOM_STATE)
    plt.scatter(X.reshape(-1), y, label="linear_data")
    plt.scatter(X.reshape(-1), 0.01 * y**2, label="squared_data")

    model = LinearSVR(C=10)
    model.fit(X, y)
    test_x = np.linspace(X.min(0), X.max(0), 2)
    pred = model.predict(test_x)
    plt.plot(test_x, pred, label="LinearSVC", color='red')

    model = KernelSVR(C=100, kernel='poly', degree=2)
    model.fit(X, 0.01 * y**2)
    test_x = np.linspace(X.min(0), X.max(0), 100)
    pred = model.predict(test_x)
    plt.plot(test_x, pred, label="KernelSVC", color='yellowgreen')

    plt.legend()
    plt.show()


def dataset_svr_test(dataset="boston"):
    X, y = {
        "boston": load_boston,
        "diabetes": load_diabetes,
    }[dataset](return_X_y=True)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        train_size=0.7,
        random_state=RANDOM_STATE,
    )
    print(("dataset : load_{}\n" + "{}SVR's perf : {}\n" * 3).format(
        dataset,
        "Linear",
        LinearSVR().fit(train_X, train_y).score(test_X, test_y),
        "Kernel",
        KernelSVR(C=100).fit(train_X, train_y).score(test_X, test_y),
        "    Nu",
        NuSVR(C=100).fit(train_X, train_y).score(test_X, test_y),
    ))


def visual_one_class_test():
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

    X = 0.3 * np.random.randn(100, 2)
    X_train = np.r_[X + 2, X - 2]

    X = 0.3 * np.random.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]

    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

    clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)

    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("Novelty Detection")
    plt.contourf(xx,
                 yy,
                 Z,
                 levels=np.linspace(Z.min(), 0, 7),
                 cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

    s = 40
    b1 = plt.scatter(X_train[:, 0],
                     X_train[:, 1],
                     c='white',
                     s=s,
                     edgecolors='k')
    b2 = plt.scatter(X_test[:, 0],
                     X_test[:, 1],
                     c='blueviolet',
                     s=s,
                     edgecolors='k')
    c = plt.scatter(X_outliers[:, 0],
                    X_outliers[:, 1],
                    c='gold',
                    s=s,
                    edgecolors='k')

    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))

    plt.legend([a.collections[0], b1, b2, c], [
        "learned frontier", "training data", "test regular data",
        "test abnormal data"
    ],
               loc="upper left",
               prop=FontProperties(size=11))
    plt.xlabel(
        "error train: %d/200 ;   errors novel regular: %d/40 ;   errors novel abnormal: %d/40"
        % (n_error_train, n_error_test, n_error_outliers))
    plt.show()


dataset_svr_test()