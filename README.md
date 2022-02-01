# PySVM

实现LIBSVM中的SVM算法，对标sklearn中的SVM模块

- [x] LinearSVC
- [x] SVC
- [x] NuSVC
- [x] LinearSVR
- [x] SVR
- [x] NuSVR
- [x] OneClassSVM

2021.11.05 : 加入了高斯核函数的RFF方法。
2022.01.27 : 通过向量化运算对算法进行提速，加入性能对比。
2022.01.30 : 删除Solver类，设计针对特定问题的SMO算法。 
2022.02.01 : 修改SVR算法中的错误

## 主要算法

Python(numpy)实现SMO算法，也就是

<img src="src/formula.png" alt="opt" style="zoom:67%;" />

和

<img src="src/nu-formula.png" alt="opt2" style="zoom: 55%;" />

的优化算法，从而实现支持向量机分类、回归以及异常检测。

## 与sklearn的性能对比

`sklearn.svm`模块中支持多种SVM模型，其中线性模型是由LIBLINEAR实现，也就是LinearSVC和LinearSVR模型，其它模型都是由LIBSVM实现，是我们比较的重点。比较指标为模型性能与耗时，实验数据为`sklearn`内置数据集。**我们只比较训练准确率和训练集上的r2**。

### 分类问题

|                 模型\数据集 |  Iris  | Breast Cancer |  Wine  | Digits |
| --------------------------: | :----: | :-----------: | :----: | :----: |
|       PySVM.LinearSVC(C=10) | 97.33% |    97.53%     |  100%  | 96.88% |
| sklearn.svm.LInearSVC(C=10) | 96.67% |    98.95%     |  100%  | 99.67% |
|       PySVM.KernelSVC(C=10) | 98.67% |    99.47%     |  100%  |  100%  |
|       sklearn.svm.SVC(C=10) | 98.67% |    99.12%     |  100%  |  100%  |
|          PySVM.NuSVC(ν=0.5) | 97.33% |    95.08%     | 94.38% | 96.93% |
|    sklearn.svm.NuSVC(ν=0.5) | 96.67% |    94.55%     | 98.88% | 96.67% |

上表是我们设计的模型与sklearn中SVM模型在实验数据集上的表现，PySVM.*表示我们设计的模型，相同模型的参数相同。下面比较模型计算时间(独立重复运行10次的平均时间，单位:s)，考虑到我们的算法是Python实现，所以显然无法和C++实现的LIBSVM和LIBLINEAR媲美，我们的目标是尽可能加速模型。

|                 模型\数据集 | Iris  | Breast Cancer | Wine  | Digits |
| --------------------------: | :---: | :-----------: | :---: | :----: |
|       PySVM.LinearSVC(C=10) | 0.030 |     0.083     | 0.026 | 1.466  |
| sklearn.svm.LInearSVC(C=10) | 0.007 |     0.008     | 0.002 | 0.191  |
|       PySVM.KernelSVC(C=10) | 0.050 |     0.080     | 0.043 | 2.385  |
|       sklearn.svm.SVC(C=10) | 0.002 |     0.015     | 0.003 | 0.799  |
|          PySVM.NuSVC(ν=0.5) | 0.021 |     0.043     | 0.024 | 1.913  |
|    sklearn.svm.NuSVC(ν=0.5) | 0.004 |     0.032     | 0.005 | 0.628  |

## SVM效果及可视化

我们在`example.py`中通过测试函数介绍了这些模型的用法，并将测试结果打印或可视化出来：

### 自制数据集的分类

<img src="src/1.png" alt="1" style="zoom:67%;" />

分类结果(默认参数，并没有调参)

```python
LinearSVC's perf : 83.33%
KernelSVC's perf : 83.33%
    NuSVC's perf : 83.33%
```

### 对sklearn自带数据集分类

以breast_cancer数据集为例：

```python
LinearSVC's perf : 96.49%
KernelSVC's perf : 97.08%
    NuSVC's perf : 94.74%
```

### 基于随机傅里叶特征(RFF)的SVC

以breast_cancer数据集为例：

```python
RBF-KernelSVC's perf : 97.08%, RFF-KernelSVC's perf : 95.91%
RBF-NuSVC's perf : 94.74%, RFF-NuSVC's perf : 94.74%
```

### 自制数据集进行回归

分别用线性核和二次多项式核对两种数据进行回归：

<img src="src/2.png" alt="2" style="zoom:67%;" />

### 对sklearn自带数据集回归

以boston数据集为例：

```python
dataset : load_boston
LinearSVR's perf : 0.6718188093886592
KernelSVR's perf : 0.8587363109159876
    NuSVR's perf : 0.7779371361462216
```

### 自制数据集对OneClassSVM测试

自制带异常点的数据集，用OneClassSVM进行异常检测，并可视化：

<img src="src/3.png" alt="3" style="zoom:67%;" />
