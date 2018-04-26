""""查看训练集的特征图像信息以及特征之间的相关性"""

train_housing = strat_train_set.copy()
train_housing.plot(kind="scatter", x="longitude", y="latitude")