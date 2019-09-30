import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import logistic
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from tpot import TPOTClassifier

#读入数据
train_data=pd.read_csv('MyDrive/Colab Notebooks/train.csv')
test_data=pd.read_csv('MyDrive/Colab Notebooks/test.csv')

#删除全部为空的行
train_data=train_data.dropna(axis=0,how='all')
test_data=test_data.dropna(axis=0,how='all')

#查看前5
print(train_data.head())

#查看数据基本情况
print(train_data.info())
print(test_data.info())

#弥补age缺失值 用均值
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)

#弥补fare缺失值 用均值
train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)


#弥补Embarked缺失值，用众数 因为是string 不能用mode
print(train_data['Embarked'].value_counts())
train_data['Embarked'].fillna('S',inplace=True)
test_data['Embarked'].fillna('S',inplace=True)


# #不用onehot
# train_data['Sex']=train_data['Sex'].replace(['male','female'],[0,1])
# test_data['Sex']=test_data['Sex'].replace(['male','female'],[0,1])
# train_data['Embarked']=train_data['Embarked'].replace(['S','Q','C'],[0,1,2])
# test_data['Embarked']=test_data['Embarked'].replace(['S','Q','C'],[0,1,2])

#定义特征值
features=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
#定义训练和测试数据
train_features=train_data[features]
train_y= train_data['Survived']
test_features=test_data[features]

#将xonehot化
dv=DictVectorizer(sparse=False)
train_dv=dv.fit_transform(train_features.to_dict(orient='record'))
test_dv=dv.transform(test_features.to_dict(orient='record'))


# #训练模型 CART
TP=TPOTClassifier()
TP.fit(train_dv,train_y)
pred=TP.predict(test_dv)


# for n,p in zip(test_data['Name'],pred):
#     print('{}是{}'.format(n,p))

# 得到决策树准确率(基于训练集)
acc_decision_tree = round(TP.score(train_dv, train_y), 6)
print(u'score准确率为 %.4lf' % acc_decision_tree)
