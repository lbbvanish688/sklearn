"""
一个集成Logistic回归，SVM分类，Random forest分类的投票分类器，实验数据由moon产生，由于是硬投票，所以voting要设置为hard
"""
#产生moon数据并分开训练测试集
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
(X,y)=make_moons(1000,noise=0.5)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


#构造模型和集成模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
voting_clf = VotingClassifier(
estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
voting='hard'
)
voting_clf.fit(X_train, y_train)

#训练并预测
from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

"""
另一种方法是对每个分类器使用相同的算法，但是要在训练集的不同随机子集上进行训练。如果抽样时有放回，称为Bagging；当抽样没有放回，称为Pasting
"""
"""
选择决策树分类器作为训练算法；n_estimators表示产生分类器的数目；max_samples为每个分类器分得的样本数；bootstrap=True表示使用bagging算法，否则为pasting算法；n_jobs表示使用CPU核的数目，-1代表把能用的都用上。
"""


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf = BaggingClassifier(
DecisionTreeClassifier(), n_estimators=500,
max_samples=100, bootstrap=True, n_jobs=-1
)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print(bag_clf.__class__.__name__, accuracy_score(y_test, y_pred))