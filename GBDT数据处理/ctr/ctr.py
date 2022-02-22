# -*- coding: utf-8 -*-
"""
Created on 2019-04-02 20:07:41

简介：gbdt + lr ，ctr点击率预测
        因为电脑内存不够，训练集选择总数据前8000个，测试集选择总数据后 2000 个

@author: 杨昊 yh4241@foxmail.com
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import preprocessing
# from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import itertools
import matplotlib.pyplot as plt
import graphviz
from sklearn.metrics import confusion_matrix
from pdpbox import pdp, get_dataset, info_plots
import pydotplus
import networkx
import json

##========================= load data ========================================

df = pd.read_csv(r"ctr_data.csv")
# df_test = pd.read_csv(r"D:\Kaggle_CTR\test.csv")


cols = ['C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']#1.412
# cols = ['C1','banner_pos', 'site_domain',  'site_id', 'site_category','app_id', 'app_category', 'app_domain','device_type',  'device_conn_type', 'C14', 'C15','C16']#1.29

cols_all = ['id']
cols_all.extend(cols)
# print(df.head(10))

y = df['click']
y_train = y.iloc[:-2000]  # training label
y_test = y.iloc[-2000:]  # testing label

X = df[cols_all[1:]]  # training dataset

# label encode
lbl = preprocessing.LabelEncoder()
X['site_domain'] = lbl.fit_transform(X['site_domain'].astype(str))  # 将提示的包含错误数据类型这一列进行转换
X['site_id'] = lbl.fit_transform(X['site_id'].astype(str))
X['site_category'] = lbl.fit_transform(X['site_category'].astype(str))
X['app_id'] = lbl.fit_transform(X['app_id'].astype(str))
X['app_category'] = lbl.fit_transform(X['app_category'].astype(str))
X['app_domain'] = lbl.fit_transform(X['app_domain'].astype(str))
X['device_id'] = lbl.fit_transform(X['device_id'].astype(str))
X['device_ip'] = lbl.fit_transform(X['device_ip'].astype(str))
X['device_model'] = lbl.fit_transform(X['device_model'].astype(str))

X_train = X.iloc[:-2000]
X_test = X.iloc[-2000:]  # testing dataset

##=========================== gbdt -lightgbm =================================

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# number of leaves,will be used in feature transformation
num_leaf = 64

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_train)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict and get data on leaves, training data


##================= convert raw data to sparse-concatenate-new-data ==========

##====================== 训练集转换

print('Writing transformed training data')
y_pred_train = gbm.predict(X_train, pred_leaf=True)

transformed_training_matrix = np.zeros([len(y_pred_train), len(y_pred_train[0]) * num_leaf],
                                       dtype=np.int64)  # N * num_tress * num_leafs
for i in range(0, len(y_pred_train)):
    temp = np.arange(len(y_pred_train[0])) * num_leaf + np.array(y_pred_train[i])
    transformed_training_matrix[i][temp] += 1

##===================== 测试集转换
print('Writing transformed testing data')
y_pred_test = gbm.predict(X_test, pred_leaf=True)
print("y_pred_test:", y_pred_test)
# 评估模型
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred_test) ** 0.5) # 计算真实值和预测值之间的均方根误差

transformed_testing_matrix = np.zeros([len(y_pred_test), len(y_pred_test[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred_test)):
    temp = np.arange(len(y_pred_test[0])) * num_leaf + np.array(y_pred_test[i])
    transformed_testing_matrix[i][temp] += 1

# print('画出特征重要度...')
ax = lgb.plot_importance(gbm, max_num_features=30)
plt.show()
print('Feature importances:', list(gbm.feature_importance()))# 结果包含该特征在模型中使用的次数。

print('画出部分依赖图...')
# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=gbm, dataset=X_train, model_features=cols, feature='C14')
# plot it
pdp.pdp_plot(pdp_goals, 'C14')
plt.show()

# print('用graphviz画出第1颗树...')
# graph = lgb.create_tree_digraph(gbm, tree_index=0, name='Tree1', show_info=['split_gain', 'internal_value', 'leaf_value'])
# graph.render(view=True)

print('Plotting split value histogram...')
ax = lgb.plot_split_value_histogram(gbm, feature='C14', bins='auto')
plt.show()

print('导出决策树的pdf图像到本地')#这里需要安装graphviz应用程序和python安装包
graph = lgb.create_tree_digraph(gbm, tree_index=0, name='Tree0')
graph.render(view=True)

dotplus = pydotplus.graph_from_dot_data(graph.source)
# if graph doesn't have multiedges, use dotplus.set_strict(true)
nx_graph = networkx.nx_pydot.from_pydot(dotplus)

data = networkx.readwrite.json_graph.tree_data(nx_graph, root='split0')
# data is just a dictionary. Dump results in proper json format.
json.dump(data, open('/Users/clarashar/Downloads/experiment/gbdt.json', 'w'))

##=================================  LR ======================================
lm = LogisticRegression(penalty='l2', C=0.05)  # logestic model construction
lm.fit(transformed_training_matrix, y_train)  # fitting the data
y_pred_lr_test = lm.predict_proba(transformed_testing_matrix)  # Give the probabilty on each label

##===============================  metric ====================================
NE = (-1) / len(y_pred_lr_test) * sum(
    ((1 + y_test) / 2 * np.log(y_pred_lr_test[:, 1]) + (1 - y_test) / 2 * np.log(1 - y_pred_lr_test[:, 1])))
print("Normalized Cross Entropy " + str(NE))