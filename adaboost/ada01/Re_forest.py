#产生moon数据并分开训练测试集
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
(X,y)=make_moons(1000,noise=0.5)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

"""
随机森林算法是以决策树算法为基础，通过bagging算法采样训练样本，再抽样特征，3者组合成的算法。
"""
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)


"""
随机森林算法每个分类器是从随机抽样部分特征，然后选择最优特征来划分。如果在此基础上使用随机的阈值分割这个最优特征，而不是最优的阈值，这就是Extra-Trees（Extremely Randomized Trees）。这会再次增加偏差，减少方差。一般来说，Extra-Trees训练速度优于随机森林，因为寻找最优的阈值比随机阈值耗时。对应scikit-learn的类为ExtraTreesClassifier（ExtraTreesRegressor），参数与随机森林相同。

  Extra-tree和随机森林哪个更好不好比较，只能通过交叉验证两种算法都实验一次才能知道结果。
"""


"""
判断决策树的特征重要性
在Scikit-learn可以通过feature_importance获得特征的重要程度
"""
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)