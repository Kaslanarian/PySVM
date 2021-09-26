import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from one_class import OneClassSVM

# 背景坐标点阵
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

# 生成训练数据
X = 0.3 * np.random.randn(100, 2)  # 100个正常数据，shape=(100,2),[0,1)之间
X_train = np.r_[X + 2, X - 2]  # 向左侧平移2得到一组数据，向右侧平移2得到一组数据，两组数据串联，

# 生成测试数据
X = 0.3 * np.random.randn(20, 2)  # 20个异常数据
X_test = np.r_[X + 2, X - 2]  # 向左侧平移2得到一组数据，向右侧平移2得到一组数据，两组数据串联，

# 生成20个异常数据，
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# 训练模型
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)

#----》判断数据是在超平面内还是超平面外，返回+1或-1，正号是超平面内，负号是在超平面外
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

#----》计算样本到超平面的距离，含正负号，正好表示在超平面内，负号表示在超平面外
y_deci_train = clf.decision_function(X_train)
y_deci_test = clf.decision_function(X_test)
y_deci_outliers = clf.decision_function(X_outliers)

# 统计预测错误的个数
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# 计算网格数据到超平面的距离，含正负号
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  # ravel表示数组拉直
Z = Z.reshape(xx.shape)
"""
绘图
"""
plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7),
             cmap=plt.cm.PuBu)  #绘制异常区域的轮廓， 把异常区域划分为7个层次
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                colors='darkred')  # 绘制轮廓，SVM的边界点（到边界距离为0的点
plt.contourf(xx, yy, Z, levels=[0, Z.max()],
             colors='palevioletred')  # 绘制正常样本的区域，使用带有填充的轮廓

s = 40  # 样本点的尺寸大小
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s,
                 edgecolors='k')  # 绘制训练样本，填充白色，边缘”k“色
b2 = plt.scatter(X_test[:, 0],
                 X_test[:, 1],
                 c='blueviolet',
                 s=s,
                 edgecolors='k')  # 绘制测试样本--正常样本，填充蓝色，边缘”k“色
c = plt.scatter(X_outliers[:, 0],
                X_outliers[:, 1],
                c='gold',
                s=s,
                edgecolors='k')  # 绘制测试样本--异常样本，填充金色，边缘”k“色

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))

# 集中添加图注
plt.legend([a.collections[0], b1, b2, c], [
    "learned frontier", "training data", "test regular data",
    "test abnormal data"
],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "error train: %d/200 ;   errors novel regular: %d/40 ;   errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outliers))
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np
from one_class import OneClassSVM
import seaborn as sns

sns.set()
X_train = np.random.randn(1000)
X_train[:10] = 0.001 * np.random.randn(10) + 10
sns.histplot(X_train)
# %%
X_test = [-10, 0, 10]
model = OneClassSVM()
model.fit(X_train.reshape(-1, 1))
# %%
model.predict(X_test)