from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
from pysvm import LinearSVR, KernelSVR, NuSVR

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
plt.scatter(X.reshape(-1), y - 100, label="linear_data")
plt.scatter(X.reshape(-1), 0.01 * y**2, label="squared_data")
plt.scatter(X.reshape(-1), 100 * np.sin(0.01 * y) + 100, label="sin_data")

model = LinearSVR(C=10)
model.fit(X, y - 100)
test_x = np.linspace(X.min(0), X.max(0), 2)
pred = model.predict(test_x)
plt.plot(test_x, pred, label="LinearSVR", color='red')

model = NuSVR(kernel='poly', degree=2, C=10)
model.fit(X, 0.01 * y**2)
test_x = np.linspace(X.min(0), X.max(0), 100)
pred = model.predict(test_x)
plt.plot(test_x,
         pred,
         label="NuSVR(kernel=poly, degree=2)",
         color='yellowgreen')

model = KernelSVR(C=10, gamma=0.25)
model.fit(X, 100 * np.sin(0.01 * y) + 100)
test_x = np.linspace(X.min(0), X.max(0), 100)
pred = model.predict(test_x)
plt.plot(test_x,
         pred,
         label="KernelSVR(kernel=rbf, gamma=0.25)",
         color='orange')

plt.legend()
plt.title("Regression visualization")
plt.savefig("../src/visual_regression.png")