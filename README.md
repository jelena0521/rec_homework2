# 仅用于学习日记，不可用于任何商业用途
#对Titanic数据进行生存率预测  dt lr  tpot
#1、读取数据 pd.read_csv
#2、补缺失值  方法：删除整行、补均值、众数等
#train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
#3、定义train_x train_y
#4、特征中可能含有string 统一onehot化
#dv=DictVectorizer()
#train_dv=dv.fit_transform(train_features.to_dict(orient='record'))
#test_dv=dv.transform(test_features.to_dict(orient='record'))
#5、训练
#clf.fit  clf.predict  clf.score
#如果用逻辑回归 比决策树多一步数据标准化 ss=StandardScaler()
#用TP=TPOTClassifier()  这是一种automl 在众多机器学习算法中选择最适合这个数据集的算法和参数，缺点是很慢
