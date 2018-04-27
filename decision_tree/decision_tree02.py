from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=2)
#tree_reg.fit(X, y)

"""
决策树回归与决策树分类相似，不同的是决策树分类叶子节点最终预测的是类别，而决策树回归叶子节点最终预测的是一个值。
"""
