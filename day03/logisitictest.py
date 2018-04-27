import numpy as np
from sklearn import datasets

iris=datasets.load_iris()
X=iris["data"][:,3]
y=(iris["target"]==2).astype(np.int)

#训练Logistic 回归模型
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X,y)

