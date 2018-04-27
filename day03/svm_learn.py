"""
支持向量机（SVM）是一种非常强大的机器学习模型，能够进行线性、非线性分类、回归问题，还能检测异常值。SVM特别适用于复杂但小型或中型的数据集的分类。
"""
"""
SVM对特征之间的尺度比较敏感，因此要先对特征进行缩放（如标准化（StandardScaler）） 
"""
#coding:utf-8
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
iris=datasets.load_iris()
X=iris["data"][:,(2,3)]
y=(iris["target"]==2).astype(np.float64)
print(y)
svm_clf=Pipeline(
    (
        ("scaler",StandardScaler()),
        ("liner_scv",LinearSVC(C=1,loss="hinge")),

    )
)
svm_clf.fit(X,y)

preed=svm_clf.predict([[5.5,1.7]])
print(preed)
"""
使用SVC类，参数为（kernel=“linear”，C=1）,
"""
#非线性SVM分类   器  Nonlinear SVM Classification   Polynomial Kernel（多项式核）



