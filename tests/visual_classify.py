from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
from pysvm import LinearSVC, KernelSVC

try:
    import seaborn as sns
    sns.set()
except:
    pass

RANDOM_STATE = 2022

X, y = make_classification(
    n_samples=250,
    n_classes=2,
    n_features=2,
    n_redundant=0,
    random_state=RANDOM_STATE,
)

x_min = np.min(X[:, 0])
x_max = np.max(X[:, 0])
y_min = np.min(X[:, 1])
y_max = np.max(X[:, 1])

plot_x = np.linspace(x_min - 1, x_max + 1, 1001)
plot_y = np.linspace(x_min - 1, x_max + 1, 1001)
xx, yy = np.meshgrid(plot_x, plot_y)

clf = LinearSVC().fit(X, y)
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap = plt.cm.coolwarm

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.contourf(xx, yy, Z, cmap=cmap)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('SVC with linear kernel(Decision function view)')

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.subplot(2, 2, 2)
plt.contourf(xx, yy, Z, cmap=cmap)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('SVC with linear kernel(0-1 view)')

clf = KernelSVC().fit(X, y)
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(2, 2, 3)
plt.contourf(xx, yy, Z, cmap=cmap)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('SVC with Gaussian kernel(Decision function view)')

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.subplot(2, 2, 4)
plt.contourf(xx, yy, Z, cmap=cmap)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('SVC with Gaussian kernel(0-1 view)')

plt.savefig("../src/visual_classify.png")
