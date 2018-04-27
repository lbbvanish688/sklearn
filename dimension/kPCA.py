"""
SVM中提到了核技巧，即通过数学方法达到增加特征类似的功能来实现非线性分类。类似的技巧还能用在PCA上，使得可以实现复杂的非线性投影降维，称为kPCA。该算法善于保持聚类后的集群(clusters)后投影，有时展开数据接近于扭曲的流形。
"""
#生成Swiss roll数据
from sklearn.datasets import make_swiss_roll
data=make_swiss_roll(n_samples=1000, noise=0.0, random_state=None)
X=data[0]
y=data[1]
print(X.shape,y.shape)