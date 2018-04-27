import numpy as np
X=2*np.random.rand(100,1)
y=4+3*X+np.random.randn(100,1)

#线性模型


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)
print(lin_reg.intercept_)
print(lin_reg.coef_)

eta=0.1
n_iterations=1000
m=100
theta=np.random.randn(2,1)
X_b=np.c_[np.ones((100,1)),X]

"""
a = np.array([[1, 2, 3],[4,5,6]])
b=np.array([[1,3,5],[2,4,6]])

c=np.c_[a,b]   #cow
print(c)
d=np.r_[a,b]  #row
print(d)

"""
for iteration in range(n_iterations):
    gradients = 2/float(m) * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

#随机梯度下降
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
sgd_reg.intercept_, sgd_reg.coef_

