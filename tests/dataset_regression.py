import numpy as np
from pysvm import LinearSVR, KernelSVR, NuSVR
from sklearn.datasets import load_boston, load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(2022)

score = np.zeros((3, 2))
for i, load_dataset in enumerate([load_boston, load_diabetes]):
    X, y = load_dataset(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y)
    stder = StandardScaler().fit(train_X)
    train_X = stder.transform(train_X)
    test_X = stder.transform(test_X)

    for j, model in enumerate([LinearSVR, KernelSVR, NuSVR]):
        score[j, i] = model(max_iter=1000).fit(train_X,
                                               train_y).score(test_X, test_y)

print(score)
