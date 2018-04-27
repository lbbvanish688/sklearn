"""
决策树，它能够处理回归和分类问题，甚至是多输出问题，能够拟合复杂的数据（容易过拟合），而且它是集成算法：随机森林（Random forest）的基础，下面开始介绍决策树Scikit-learn的用法，以及参数的选择及算法的局限性。
"""
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
proba=tree_clf.predict_proba([[5, 1.5]])
print(proba)
type=tree_clf.predict([[5, 1.5]])
print(type)


from sklearn.tree import export_graphviz
export_graphviz(
tree_clf,
out_file=image_path("iris_tree.dot"),
feature_names=iris.feature_names[2:],
class_names=iris.target_names,
rounded=True,
filled=True
)
"""
因此为了防止对训练数据过拟合，需要增加一些参数来限制。最一般的设置应该设置最大深度（max_depth）；DecisionTreeClassifier类还有一些其他参数用来防止过拟合，节点被分开的最小样本数（min_samples_split）；叶子节点的最小样本数（min_samples_leaf）；和min_samples_leaf有点像，不过这个是分开节点变为叶子的最小比例，（min_weight_fraction_leaf）；叶子节点的最大样本数（max_leaf_nodes）；在每个节点分开时评估的最大特征数（max_features）。增加min_*，减小max_*都能正则化算法。 """