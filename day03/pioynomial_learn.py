from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#构造球型数据集
(X,y)=make_moons(200,noise=0.2)
#使用SVC类中的多项式核训练
poly_kernel_svm_clf = Pipeline((
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
))
#其中参数coef0为高degree特征相比低degree特征对模型的影响程度。参数degree为选择多项式特征的维度，参数C为松弛因子。
poly_kernel_svm_clf.fit(X, y)
import numpy as np
import matplotlib.pyplot as plt
xx, yy = np.meshgrid(np.arange(-2,3,0.01), np.arange(-1,2,0.01))
y_new=poly_kernel_svm_clf.predict(np.c_[xx.ravel(),yy.ravel()])
plt.contourf(xx, yy, y_new.reshape(xx.shape),cmap="PuBu")   #其中前两个参数x和y为两个等长一维数组，第三个参数z为二维数组（表示平面点xi,yi映射的函数值）。
plt.scatter(X[:,0],X[:,1],marker="o",c=y)
plt.show()

