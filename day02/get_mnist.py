import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original',data_home="./datasets")

X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

import numpy as np
shuffle_index = np.random.permutation(60000) #打乱顺序
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#二分类
"""
假设现在分类是否为数字5，则分类两类（是5或不是5），训练一个SGD分类器（该分类器对大规模的数据处理较快）。
"""
#划分数据
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
#训练模型
from sklearn.linear_model import SGDClassifier    #随机梯度下降线性分类器

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
#交叉验证
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print(y_train_pred)
#查全率，查准率
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

from sklearn.metrics import f1_score   #F1指标   考虑到两个部分
f1_score(y_train_5, y_train_pred)



from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

some_digit = X[36000]    #随便去一张

predict1=sgd_clf.predict([some_digit])
print(predict1)
some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)

from sklearn.model_selection import cross_val_score    #交叉验证
accuracy=cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print(accuracy)

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred=cross_val_predict(sgd_clf,X_train,y_train,cv=3)
conf_mx=confusion_matrix(y_train,y_train_pred)
print(conf_mx)
"""plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
"""
#生成左边的噪声图
import numpy.random as rnd
noise1 = rnd.randint(0, 100, (len(X_train), 784))
noise2 = rnd.randint(0, 100, (len(X_test), 784))
X_train_mod = X_train + noise1
X_test_mod = X_test + noise2
y_train_mod = X_train
y_test_mod = X_test
import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.imshow(X_train_mod[36000].reshape(28,28),cmap=plt.cm.gray)
plt.subplot(1,2,2)
plt.imshow(X_train[36000].reshape(28,28),cmap=plt.cm.gray)
#去燥
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_train_mod[36000]])
plt.imshow(clean_digit.reshape(28,28),cmap=plt.cm.gray)
plt.show()

