"""
主成分分析（PCA）是用的最出名的降维技术，它通过确定最接近数据的超平面，然后将数据投射(project)到该超平面上。
"""
import numpy as np
x1=np.random.normal(0,1,100)
x2=x1*1+np.random.rand(100)

x=np.c_[x1,x2]

x_centered=x-x.mean(axis=0)

U,s,V=np.linalg.svd(x_centered)
c1=V.T[:,0]
c2=V.T[:,1]
#得到主成分以后就能将数据降维，假设降到d维空间，则用数据矩阵与前d个主成分形成的矩阵相乘，得到降维后的矩阵
d=1
Wd = V.T[:, :d]
XD = x_centered.dot(Wd)


#Scikit-learn提供了PCA类，n_components控制维数
from sklearn.decomposition import PCA
pca=PCA(n_components=1)
XD=pca.fit_transform(x)

"""
fit以后可以通过components_变量来输出主成分，还可以通过explained_variance_ratio_来查看每个主成分占方差的比率。
"""
#查看主成分
pca.components_.T
#显示PCA主成分比率
print("主成分方差比率为：")
print(pca.explained_variance_ratio_)
#选择合理的维数
"""
合理的选择维数而不是随机选择一个维数，我们可以通过设置一个合适的方差比率（如95%），计算需要多少个主成分的方差比率和能达到这个比率，就选择该维度
"""
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) #累加
d = np.argmax(cumsum >= 0.95) + 1
"""
得到维度d后再次设置n_components进行PCA降维。当然还有更加简便的方法，直接设置n_components_=0.95，那么Scikit-learn能直接作上述的操作。
"""

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(x)
