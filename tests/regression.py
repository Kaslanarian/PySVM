from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
from pysvm import LinearSVR, KernelSVR

try:
    import seaborn as sns
    sns.set()
except:
    pass

RANDOM_STATE = 2022

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
plt.plot(test_x, pred, label="LinearSVR", color='red')

model = KernelSVR(kernel='poly', degree=2, C=10)
model.fit(X, 0.01 * y**2)
test_x = np.linspace(X.min(0), X.max(0), 100)
pred = model.predict(test_x)
plt.plot(test_x, pred, label="KernelSVR", color='yellowgreen')

plt.legend()
plt.savefig("../src/regression.png")