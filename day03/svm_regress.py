"""
SVM除了能进行分类任务以外还能做回归任务。与SVM分类任务尽量让点在margin以外，而SVM回归则是尽量让点在margin以内通过参数εε控制margin的大小，εε越大，margin越大，否则越小。
"""

from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5)
#svm_reg.fit(X, y)

# SVM非线性回归问题与分类问题类似，通过设置核来实现。
from sklearn.svm import SVR
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
#svm_poly_reg.fit(X, y)

"""
#修改SVC类使得有predict_proba()方法，并软投票
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)
voting_clf = VotingClassifier(
estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
voting='soft'
)
voting_clf.fit(X_train, y_train)

#训练并预测
from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
"""


"""

"""
