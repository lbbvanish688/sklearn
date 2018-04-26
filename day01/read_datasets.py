import pandas as pd
import os
import get_datasets
import matplotlib.pyplot as plt
def load_housing_datas(housing_path=get_datasets.HOUSING_PATH ):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
housing = load_housing_datas()
print(type(housing))
print(housing.head())
#housing.info() info()可以查看每个特征的元素总个数，因此可以查看某个特征是否存在缺失值。还可以查看数据的类型以及内存占用情况。
#housing["ocean_proximity"].value_counts()value_counts()统计特征中每个元素的总个数
#housing.describe()describe()可以看实数特征的最大值、最小值、平均值、方差、总个数、25%，50%，75%小值。
#import matplotlib.pyplot as plt    同过hist()生成直方图，能够查看实数特征元素的分布情况。
#housing.hist(bins=50, figsize=(20,15))
#plt.show()
"""自定义的训练测试分离"""
import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")
'''
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
'''
"""
为了解决方案一的问题，采用每个样本的识别码（可以是ID，可以是行号）来决定是否放入测试集，例如计算识别码的hash值，取hash值得最后一个字节（0~255），如果该值小于一个数（20% * 256）则放入测试集。这样，这20%的数据不会包含训练过的样本。具体代码如下：
"""
"""scikit提供的训练测试分离__随机采样
"""
"""
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
"""
"""
————分层采样
继续从真实数据来看，假设专家告诉你median_income 是用于预测median housing price一个很重要的特征，则你想把median_income作为划分的准则来观察不同的median_income对median housing price的影响。但是可以看到median_income是连续实数值。所以需要把median_income变为类别属性。

根据之前显示的图标表，除以1.5界分为5类，除了以后大于5的归为5，下面图片可以上述说过的hist()函数画出来看看，对比一下原来的median_income的分布，看是否相差较大，如果较大，则界需要调整。
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
接下来就可以根据上面分号层的”income_cat”使用StratifiedShuffleSplit函数作分层采样，其中n_splits为分为几组样本（如果需要交叉验证，则n_splits可以取大于1，生成多组样本），其他参数和之前相似。"""

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

""""查看训练集的特征图像信息以及特征之间的相关性"""
print(type(strat_train_set))
#查看训练集的特征图像信息以及特征之间的相关性
#strat_train_set.plot(kind="scatter", x="longitude", y="latitude")
#plt.show()
corr_matrix = strat_train_set.corr()
print(type(corr_matrix))
print(corr_matrix)  #线性相关性
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#准备数据  预处理
train_housing = strat_train_set.drop("median_house_value", axis=1)
train_housing_labels = strat_train_set["median_house_value"].copy()
print("标签：")
print(train_housing_labels.shape)
print("训练数据集：")
print(train_housing.shape)


#数据清洗
#train_housing.dropna(subset=["total_bedrooms"]) # option 1
#train_housing.drop("total_bedrooms", axis=1) # option 2
#median = train_housing["total_bedrooms"].median()
#train_housing["total_bedrooms"].fillna(median) # option 3

#scikit对数据的清洗
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
housing_num = train_housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
#也可以将numpy格式的转换为pd格式
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
#文本属性的处理
"""
由于文本属性不能作median等操作，所以需要将文本特征编码为实数特征，
对应Scikit-Learn中的类为LabelEncoder，
通过调用LabelEncoder类，再使用fit_transform()方法自动将文本特征编码"""
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = train_housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(housing_cat_encoded)     #标签
print(encoder.classes_)
"""上述是针对文本作为标签
下面是将文本作为特征值
如果用于特征，则这种数字编码不适用，应该采用one hot编码（形式可以看下面的图），对应Scikit-Learn中的类为OneHotEncoder

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
#encoder = LabelBinarizer(sparse_output=True)
housing_cat_1hot = encoder.fit_transform(housing_cat)
"""
#由于Scikit-Learn没有处理Pandas数据的DataFrame，因此需要自己自定义一个如下
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

#自定义类由于Scikit-Learn中的函数中提供的Transformer方法并不一定适用于真实情形，所以有时候需要自定义一个Transformer，与Scikit-Learn能够做到“无缝结合”，比如pineline（以后会说到）。定义类时需要加入基础类：BaseEstimator（必须），以及TransformerMixin（用于自动生成fit_transformer()方法）。下面是一个例子：用于增加组合特征的Trainsformer
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

"""
 Pipeline是由（name(名字)，Estimator(类)）对组成，但最后一个必须为transformer，这是因为要形成fit_transform()方法
"""

""""""
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
class LabelBinarizer_new(TransformerMixin, BaseEstimator):
    def fit(self, X, y = 0):
        self.encoder = None
        return self
    def transform(self, X, y = 0):
        if(self.encoder is None):
            self.encoder = LabelBinarizer();
            result = self.encoder.fit_transform(X)
        else:
            result = self.encoder.transform(X)
        return result



"""数据预处理，文本信息转换，特征值缩放"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#特征值缩放
from sklearn.pipeline import FeatureUnion
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
('selector', DataFrameSelector(num_attribs)),
('imputer', Imputer(strategy="median")),
('attribs_adder', CombinedAttributesAdder()),
('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
('selector', DataFrameSelector(cat_attribs)),
('label_binarizer', LabelBinarizer_new()),
])
full_pipeline = FeatureUnion(transformer_list=[
("num_pipeline", num_pipeline),
("cat_pipeline", cat_pipeline),
])
housing_prepared = full_pipeline.fit_transform(train_housing)



#第一个模型 线性回归模型   多元线性回归

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, train_housing_labels)

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(train_housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)   #均方误差
print(lin_rmse)


#决策树模型（DecisionTreeRegressor）
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, train_housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(train_housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

#交叉验证
from sklearn.model_selection import cross_val_score
tree_scores = cross_val_score(tree_reg, housing_prepared, train_housing_labels,
scoring="neg_mean_squared_error", cv=10)
lin_scores = cross_val_score(lin_reg, housing_prepared, train_housing_labels,
scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
lin_rmse_scores = np.sqrt(-lin_scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
display_scores(lin_rmse_scores)

from sklearn.externals import joblib
joblib.dump(tree_reg, "my_model.pkl") #保存模型
# and later...
my_model_loaded = joblib.load("my_model.pkl") #加载模型
print(type(my_model_loaded))    #加载的模型还是原来的对象模型

#模型调参
"""
scikit-learn中提供函数GridSearchCV用于网格搜索调参，网格搜索就是通过自己对模型需要调整的几个参数设定一些可行值，然后Grid Search会排列组合这些参数值，每一种情况都去训练一个模型，经过交叉验证今后输出结果。下面为随机森林回归模型（RandomForestRegression）的一个Grid Search的例子。
"""
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, train_housing_labels)

# 输出最好的参数
print(grid_search.best_params_)

"""

随机搜索（Randomized Search）
由于上面的网格搜索搜索空间太大，而机器计算能力不足，则可以通过给参数设定一定的范围，在范围内使用随机搜索选择参数，随机搜索的好处是能在更大的范围内进行搜索，并且可以通过设定迭代次数n_iter，根据机器的计算能力来确定参数组合的个数，是下面给出一个随机搜索的例子。
"""
from sklearn.model_selection import RandomizedSearchCV
param_ran={'n_estimators':range(30,50),'max_features': range(3,8)}
forest_reg = RandomForestRegressor()
random_search = RandomizedSearchCV(forest_reg,param_ran,cv=5,scoring='neg_mean_squared_error',n_iter=10)
random_search.fit(housing_prepared, train_housing_labels)


#分析最好的模型每个特征的重要性
"""
假设现在调参以后得到最好的参数模型，然后可以查看每个特征对预测结果的贡献程度，根据贡献程度，可以删减减少一些不必要的特征。
"""
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


#测试集上进行评估它的泛化能力
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


