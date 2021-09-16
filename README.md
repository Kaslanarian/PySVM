# Support Vector Machine

Python(numpy)实现SMO算法，也就是

![opt](src/formula.png)

的优化算法（[Solver类](./solver.py)），从而实现支持向量机分类与回归。

## 支持向量分类

目前只支持支持向量机的二分类

### 线性支持分类

```python
from sklearn.datasets import *
from svc import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
y[y != 0] = 1 # 转化为二分类数据
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7)

model = LinearSVC(C=10)
model.fit(train_X, train_y)
pred = model.predict(test_X)
print(accuracy_score(test_y, pred))
```

输出100\%。

### 核SVM分类

KernelSVC类实现了带核函数的支持向量分类（支持线性核、多项式核、Sigmoid核以及rbf核）：

```python
from sklearn.datasets import *
from svc import KernelSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X = (X - X.mean(0)) / X.std(0)
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7)

model = KernelSVC() # 默认RBF核
model.fit(train_X, train_y)
pred = model.predict(test_X)
print(accuracy_score(test_y, pred))
```

准确率97\%+.

## 支持向量回归

### 线性支持向量回归

将$\alpha$和$\alpha^\star$整合成一个变量，实现支持向量回归：

```python
from svr import LinearSVR
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = load_boston(return_X_y=True)
X = (X - X.mean(0)) / X.std(0)
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7)
model = LinearSVR(max_iter=2000, epsilon=1, C=1)
model.fit(train_X, train_y)
pred = model.predict(test_X)
print(mean_squared_error(test_y, pred))
```

输出测试误差为24.49，与线性模型拟合的平均效果近似（[性能baseline参考](https://welts.xyz/2021/09/07/baseline/)），验证了我们的模型实现正确。

### 核函数支持向量回归

类似的，我们实现了支持多种核函数的SVR：

```python
from svr import LinearSVR, KernelSVR
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = load_boston(return_X_y=True)
X = (X - X.mean(0)) / X.std(0)

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7)

model = KernelSVR(max_iter=2000, epsilon=0.01, C=5, kernel='rbf')
model.fit(train_X, train_y)
pred = model.predict(test_X)
print(mean_squared_error(test_y, pred))
```

当设置为上面的参数时，我们将测试误差控制在10-20内，相比于线性支持向量机性能更好。
