"""
和AdaBoost类似，Gradient Boosting也是逐个训练学习器，尝试纠正前面学习器的错误。不同的是，AdaBoost纠正错误的方法是更加关注前面学习器分错的样本，Gradient Boosting（适合回归任务）纠正错误的方法是拟合前面学习器的残差（预测值减真实值）。

"""

import numpy.random as rnd
from sklearn.tree import DecisionTreeRegressor

X=rnd.rand(200,1)-1
y=3*X**2+0.05*rnd.randn(200,1)

#训练第一个模型
tree_reg1=DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X,y)

#根据生一个模型的残差训练第二个模型
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X, y2)

##再根据上一个模型的残差训练第三个模型
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, y3)
#预测
X_new=0.5
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))


#sklearn自带的gradientBoostingRegresson
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)

"""
参数learning_rate表示每个学习器的贡献程度，如果设置learning_rate比较低，则需要较多的学习器拟合训练数据，但是通常会得到更好的效果。下图展示较低学习率的训练结果，左图学习器过少，欠拟合；右图学习器过多，过拟合。

"""