from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import  Pipeline
from sklearn.svm import SVC
from sklearn.datasets import make_moons

(X,y)=make_moons(200,noise=0.2)



rbf_kernel_svm_clf = Pipeline((
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
))
rbf_kernel_svm_clf.fit(X, y)

import numpy as np
import matplotlib.pyplot as plt
xx, yy = np.meshgrid(np.arange(-2,3,0.01), np.arange(-1,2,0.01))
y_new=rbf_kernel_svm_clf.predict(np.c_[xx.ravel(),yy.ravel()])
plt.contourf(xx, yy, y_new.reshape(xx.shape),cmap="PuBu")
plt.scatter(X[:,0],X[:,1],marker="o",c=y)

"""
决策边界范围很小，如果γγ比较大，会使得决策线变窄，变得不规则。相反，小的γγ使决策线变宽，边平滑。所以γγ就像一个正则化参数：如果你的模型过拟合，可以适当减少它，如果它欠拟合，可以增加它（类似于C超参数）。"""