"""
Boosting是将弱学习器集成为强学习器的方法，主要思想是按顺序训练学习器，以尝试修改之前的学习器。Boosting的方法有许多，最为有名的方法为AdaBoost（Adaptive Boosting）和Gradient Boosting。
"""
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
(X,y)=make_moons(1000,noise=0.5)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


"""
一个新的学习器会更关注之前学习器分类错误的训练样本。因此新的学习器会越来越多地关注困难的例子。这种技术成为AdaBoost。
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
                             DecisionTreeClassifier(max_depth=1), n_estimators=200,
                             algorithm="SAMME.R", learning_rate=0.5
                             )
ada_clf.fit(X_train, y_train)

"""
和AdaBoost类似，Gradient Boosting也是逐个训练学习器，尝试纠正前面学习器的错误。不同的是，AdaBoost纠正错误的方法是更加关注前面学习器分错的样本，Gradient Boosting（适合回归任务）纠正错误的方法是拟合前面学习器的残差（预测值减真实值）。
"""
