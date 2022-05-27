import numpy as np
from pysvm import LinearSVC, KernelSVC, NuSVC
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)

score = []
for load_dataset in [load_iris, load_wine, load_breast_cancer, load_digits]:
    X, y = load_dataset(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y)
    stder = StandardScaler().fit(train_X)
    train_X = stder.transform(train_X)
    test_X = stder.transform(test_X)

    for model in [LinearSVC, KernelSVC, NuSVC]:
        score.append(
            model(
                n_jobs=6,
                max_iter=2000,
            ).fit(train_X, train_y).score(test_X, test_y))

score = np.array(score).reshape(-1, 4)
print(score)
