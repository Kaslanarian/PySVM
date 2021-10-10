# PySVM

对标sklearn中的SVM模块

- [x] LinearSVC
- [x] SVC
- [x] NuSVC
- [x] LinearSVR
- [x] SVR
- [x] NuSVR
- [x] OneClassSVM

Python(numpy)实现SMO算法，也就是

![opt](src/formula.png)

和

![opt2](src/nu-formula.png)

的优化算法（[Solver类](./solver.py)和[NuSolver类](./solver.py)），从而实现支持向量机分类、回归以及异常检测。

我们在`example.py`中通过测试函数介绍了这些模型的用法，并将测试结果打印或可视化出来：

### 自制数据集的分类

![1](src/1.png)

分类结果(默认参数，并没有调参)

```python
LinearSVC's perf : 83.33%
KernelSVC's perf : 83.33%
    NuSVC's perf : 83.33%
```

### 对sklearn自带数据集分类

以digits数据集为例：

```python
LinearSVC's perf : 97.59%
KernelSVC's perf : 97.78%
    NuSVC's perf : 97.78%
```

### 自制数据集进行回归

分别用线性核和二次多项式核对两种数据进行回归：

![2](src/2.png)

### 对sklearn自带数据集回归

以boston数据集为例：

```python
dataset : load_boston
LinearSVR's perf : 21.32973364364534
KernelSVR's perf : 18.90831067654022
    NuSVR's perf : 179.48448849079972
```

### 自制数据集对OneClassSVM测试

自制带异常点的数据集，用OneClassSVM进行异常检测，并可视化：

![3](src/3.png)
