# coding: utf-8
# pylint: disable = invalid-name, C0111
#~/opt/anaconda3/bin/conda install pandas

from __future__ import division

import itertools
import json
import pandas as pd
import pickle
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
import matplotlib.pyplot as plt
import graphviz
import pydotplus
import networkx

from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMRegressor
from pdpbox import pdp, get_dataset, info_plots

pd.set_option("display.max_rows", 100)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def convert_str(value):
    return str(value)


def convert_float(value):
    return np.float(value)


# load or create your dataset
print('Load data...')
# 数学成绩
math_df = pd.read_pickle("data/math_df.pkl")
math_df = math_df.reset_index()

# 刷卡记录月统计
print("*************刷卡月记录开始*************")
hotwater_count_df_09 = pd.read_pickle("data/hotwater_count_df.pkl")
library_count_df_09 = pd.read_pickle("data/library_count_df.pkl")
# columns名字与学期月数一致，方便组合两个年级
hotwater_count_df_09.columns = np.arange(1, 35)
library_count_df_09.columns = np.arange(1, 35)

hotwater_count_df_10 = pd.read_pickle("data/hotwater_count_df_10.pkl").drop(["2010-07", "2010-08"],axis=1)
library_count_df_10 = pd.read_pickle("data/library_count_df_10.pkl").drop(["2010-07", "2010-08"],axis=1)

# columns名字与学期月数一致，方便组合两个年级
hotwater_count_df_10.columns = np.arange(1, 35)
library_count_df_10.columns = np.arange(1, 35)

# 将两个年级的刷卡记录组装
hotwater_count_df = pd.concat([hotwater_count_df_09, hotwater_count_df_10])
library_count_df = pd.concat([library_count_df_09, library_count_df_10])
# print("hotwater_count_df","\n",hotwater_count_df)
# print("library_count_df","\n",library_count_df)

print("*************刷卡月记录组装完毕*************")

# 真实熵
print("*************AE组装开始*************")
# reset_index:把学号从df中提取出来
shower_ae_df_09 = pd.read_pickle("data/shower_ae_df.pkl").reset_index()
meal_ae_df_09 = pd.read_pickle("data/meal_ae_df.pkl").reset_index()

shower_ae_df_10 = pd.read_pickle("data/shower_ae_df_10.pkl").reset_index()
meal_ae_df_10 = pd.read_pickle("data/meal_ae_df_10.pkl").reset_index()

meal_ae_df = pd.concat([meal_ae_df_09, meal_ae_df_09])
shower_ae_df= pd.concat([shower_ae_df_09, shower_ae_df_10])
# print("meal_ae_df","\n",meal_ae_df)
# print("shower_ae_df","\n",shower_ae_df)
print("*************AE组装完毕*************")
# Prediction
food_prediction_df_09 = pd.read_pickle("data/prediction_ae_countList/prediction_food09.pkl")
food_prediction_df_10 = pd.read_pickle("data/prediction_ae_countList/prediction_food10.pkl")
shower_prediction_df_09 = pd.read_pickle("data/prediction_ae_countList/prediction_shower09.pkl")
shower_prediction_df_10 = pd.read_pickle("data/prediction_ae_countList/prediction_shower10.pkl")


prediction_df_09 = food_prediction_df_09.join(shower_prediction_df_09, how="outer", lsuffix="_fd", rsuffix="_shwr")
prediction_df_10 = food_prediction_df_10.join(shower_prediction_df_10, how="outer", lsuffix="_fd", rsuffix="_shwr")

prediction_df = pd.concat([prediction_df_09, prediction_df_10])

# 总成绩
print("*************总成绩组装开始*************")

total_df_09 = pd.read_pickle("data/total_df.pkl")
# total_df_09['label'] = total_df_09['2011-2012_2'].map(lambda x:( x>= 60 and False ) or (x<60 and True))
total_df_10 = pd.read_pickle("data/total_df_10.pkl")
# total_df_10['label'] = total_df_10['2012-2013_2'].map(lambda x:( x>= 60 and False ) or (x<60 and True))


total_df_09.columns = [1, 2, 3, 4, 5, 6, "sid"]
total_df_10.columns = [1, 2, 3, 4, 5, 6, "sid"]


def score_to_flag(x):
    if x >= 60:
        return 0
    else:
        return 1


total_df_09['label'] = total_df_09[6].map(score_to_flag)
total_df_10['label'] = total_df_10[6].map(score_to_flag)

total_df = pd.concat([total_df_09, total_df_10])
# print("total_df","\n",total_df)
print("*************总成绩组装完毕*************")


# 数据预处理
# label： 成绩布尔化，及格/不及格

# feature：
# 第一学期每月刷卡次数
# print("hotwater_count_df\n",hotwater_count_df)
sems1 = np.arange(1,13)
# food_count_feature_df = food_count_df[sems1]
hotwater_count_feature_df = hotwater_count_df[sems1].reset_index()
library_count_feature_df = library_count_df[sems1].reset_index()
# shower_count_feature_df = shower_count_df[sems1]
# print("hotwater_count_feature_df\n",hotwater_count_feature_df)

# 所有的学号转化为str
library_count_feature_df['index'] = library_count_feature_df['index'].apply(convert_str)
hotwater_count_feature_df['index'] = hotwater_count_feature_df['index'].apply(convert_str)
shower_ae_df[0] = shower_ae_df[0].apply(convert_str)
meal_ae_df[0] = meal_ae_df[0].apply(convert_str)
math_df["index"] = math_df["index"].apply(convert_str)
total_df["sid"] = total_df["sid"].apply(convert_str)

# print("math_df\n",math_df)
math_df = math_df[["cal1_f", "linear_f", "linear_m", "cal1_m", "index"]]
math_df["cal1_f"] = math_df["cal1_f"].apply(convert_float)
math_df["linear_f"] = math_df["linear_f"].apply(convert_float)
math_df["linear_m"] = math_df["linear_m"].apply(convert_float)
math_df["cal1_m"] = math_df["cal1_m"].apply(convert_float)
# feature-label_df

res = total_df.join(library_count_feature_df.set_index("index"), on="sid", how="outer",sort=True,rsuffix="_lib")
res = res.join(hotwater_count_feature_df.set_index("index"),on="sid", how="left", sort=True,rsuffix="_hw")
res = res.join(shower_ae_df.set_index(0), on="sid", how="left", sort=True,rsuffix="_shwr")
res = res.join(math_df.set_index("index"), on="sid", sort=True, how="left", rsuffix="_mth")
# res = res.join(prediction_df, on="sid", sort=True, how="left")

# 宇宙大拼接之后，没有清除掉NAN值之前的dataset
print("本次清理NAN前总数******************: ", res.shape[0])

# 不加上之后prediction后，所有的的feature
all_feature_2 = ['1_score', '2_score', '3_score', '4_score', '5_score', '6_score', 'sid', 'label', '1_lib', '2_lib', '3_lib',
       '4_lib', '5_lib', '6_lib', '7_lib', '8_lib', '9_lib', '10_lib', '11_lib', '12_lib', '1_hw', '2_hw',
       '3_hw', '4_hw', '5_hw', '6_hw', '7_hw', '8_hw', '9_hw', '10_hw', '11_hw', '12_hw',
       '1_shwr', '2_shwr', '3_shwr', '4_shwr', '5_shwr', '6_shwr',
       'cal1_f', 'linear_f', 'linear_m', 'cal1_m']

# 不加上之后prediction后，需要去掉的feature
drop_feature_2 = ['cal1_f', 'linear_f', 'linear_m', 'cal1_m', '3_score', '4_score', '5_score', '6_score',
                  '1_shwr', '2_shwr', '3_shwr', '4_shwr', '5_shwr', '6_shwr'
                  ]

# 选择本次需要使用的feature
# drop_label = ["label","sid",'1_score','2_score','3_score','4_score','5_score','6_score']
all_feature = all_feature_2
drop_feature = drop_feature_2

selected_feature = np.setdiff1d(all_feature, drop_feature)
res.columns = all_feature

# 去掉不用的feature
res = res.drop(drop_feature, axis=1)

res = res.fillna(0)
print("all_label+++++++++++++\n", all_feature)

res =res[~res.isin([np.nan, np.inf, -np.inf]).any(1)]
print("res*********************res.shape\n", res.shape)
print("组装完成...")
a = res[selected_feature].drop(["label"], axis=1)
# 归一化
# max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
# a[['10_hw', '10_lib', '11_hw', '11_lib', '12_hw', '12_lib', '1_hw',
#        '1_lib', '2_hw', '2_lib','3_hw',
#        '3_lib', '4_hw', '4_lib', '5_hw', '5_lib', '6_hw', '6_lib', '7_hw',
#        '7_lib', '8_hw', '8_lib', '9_hw', '9_lib']] = a[['10_hw', '10_lib', '11_hw', '11_lib', '12_hw', '12_lib', '1_hw',
#        '1_lib', '2_hw', '2_lib','3_hw',
#        '3_lib', '4_hw', '4_lib', '5_hw', '5_lib', '6_hw', '6_lib', '7_hw',
#        '7_lib', '8_hw', '8_lib', '9_hw', '9_lib']].apply(max_min_scaler)
# a["sid"] = a["sid"].apply(lambda x: str(x))
X_train, X_test, y_train, y_test = train_test_split(a, res["label"], test_size=0.2, random_state=19931028)
save_result = pd.DataFrame()
save_result["sid"] = X_test["sid"]
save_result["real"] = y_test
X_train.to_json("X_train.json", orient="records")
X_test.to_json("X_test.json", orient="records")

X_train.drop(["sid"], inplace=True, axis=1)
X_test.drop(["sid"], inplace=True, axis=1)
# train = res.sample(frac=0.8,random_state=5,replace=False, axis=0)
# test = res[~res.index.isin(train.index)]
print("res*********************train.shape\n", X_train.columns)

# X_train = train.drop(drop_label,axis = 1)
# X_test = test.drop(drop_label,axis = 1)
# print("装载 feature label...")
# y_train = train["label"].map(lambda x: bool(x))
# y_test = test["label"].map(lambda x: bool(x))
#
print("本次预测使用的标签******************: ", selected_feature)# sid和label在预测中已经被drop了
feature_names = list(selected_feature)# 用于画部分依赖图
del feature_names[-1]
del feature_names[-1]

print(feature_names)

print("本次样本总数******************: ", res.shape[0])
# train  = res[:]
# test = res[3001:]
# _y_train = train['label']  # training label
# y_test = test['label']  # testing label
# _X_train = train.loc[:, feature]   # training dataset
# X_test = test.loc[:, feature]  # testing dataset

# X_train = feature.sample(frac=0.65,random_state=9,replace=False, axis=0)
# X_test = feature[~feature.index.isin(X_train.index)]
# y_train = label[label.index.isin(X_train.index)]
# y_test = label[~label.index.isin(y_train.index)]


num = 0
m = 0
what = 0
for i in y_train.to_list():
    if i == True:
        num = num+1
    elif i == False:
        m = m+1
    else:
        what += 1

print("原始train：正例======", num, "反例=======", m, "ratio:", num/(num+m), "what:", what)
# sm = SMOTE(random_state=42, sampling_strategy= 0.9)
# X_train, y_train = sm.fit_resample(X_train, y_train)

num = 0
m = 0
for i in y_train.to_list():
    if i == True:
        num = num + 1
    if i == False:
        m = m + 1
print("train：正例======", num, "反例=======", m, "ratio:", num/(num+m))
num = 0
m = 0
for i in y_test.to_list():
    if i == True:
        num = num+1
    if i == False:
        m = m+1
print("test：正例======",num,"test=======", m, "ratio:", num/(num+m))

#create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

evals_result = {}  # to record eval results for plotting

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss','auc'},
    'num_leaves': 63,#63
    # 设置max_depth防止过拟合
    'num_trees': 100,#100
    'learning_rate': 0.1,#0.01
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# number of leaves,will be used in feature transformation
num_leaf = 63

print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                evals_result=evals_result,
                num_boost_round=100,
                # verbose_eval=False,#不打印
                valid_sets=lgb_train)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict and get data on leaves, training data
y_pred = gbm.predict(X_train, pred_leaf=True)
print(np.array(y_pred).shape)

# print('在训练过程中绘图...')
# ax = lgb.plot_metric(evals_result, metric='binary_logloss')
# plt.show()
# ax = lgb.plot_metric(evals_result, metric='auc')
# plt.show()

print('画出特征重要度...')
ax = lgb.plot_importance(gbm, max_num_features=30)
plt.show()

print('画出部分依赖图...')
# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=gbm, dataset=X_train, model_features=feature_names, feature='4_lib')
# plot it
pdp.pdp_plot(pdp_goals, '4_lib')
plt.show()

# print('Plotting split value histogram...')
# ax = lgb.plot_split_value_histogram(gbm, feature='4_lib', bins='auto')
# plt.show()

print('导出决策树的pdf图像到本地')#这里需要安装graphviz应用程序和python安装包
graph = lgb.create_tree_digraph(gbm, tree_index=0, name='Tree1', show_info=['split_gain','leaf_count','leaf_weight','data_percentage', 'leaf_value'])
graph.render(view=True)

graph = lgb.create_tree_digraph(gbm, tree_index=1, name='Tree2', show_info=['split_gain','leaf_count','leaf_weight','data_percentage', 'leaf_value'])
graph.render(view=True)

dotplus = pydotplus.graph_from_dot_data(graph.source)
# if graph doesn't have multiedges, use dotplus.set_strict(true)
nx_graph = networkx.nx_pydot.from_pydot(dotplus)

data = networkx.readwrite.json_graph.tree_data(nx_graph, root='split0')
# data is just a dictionary. Dump results in proper json format.
json.dump(data, open('/Users/vis/Documents/DataVisLab/教育数据/GBDT+LR/Tree2.json', 'w'))#导出决策路径

# feature transformation and write result
print('Writing transformed training data')
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)

for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
    transformed_training_matrix[i][temp] += 1

# for i in range(0,len(y_pred)):
#	 for j in range(0,len(y_pred[i])):
#	 	transformed_training_matrix[i][j * num_leaf + y_pred[i][j]-1] = 1

# predict and get data on leaves, testing data
y_pred = gbm.predict(X_test, pred_leaf=True)#y_pred是一个1968*100的二维数组，gbdt的结果值
# print("y_pred:", y_pred)
# [[0  0  0...  0  0  0]
#  [10 59 49... 2  2  10]
#  [25 24 22... 0  62 22]...

# feature transformation and write result
print('Writing transformed testing data')
transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
    transformed_testing_matrix[i][temp] += 1

#for i in range(0,len(y_pred)):
#	for j in range(0,len(y_pred[i])):
#		transformed_testing_matrix[i][j * num_leaf + y_pred[i][j]-1] = 1

print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))# 结果包含该特征在模型中使用的次数。
# print('Feature importances:', list(gbm.feature_importance("gain")))# 结果包含使用该功能的分割的总增益。
# 模型评估
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# Logestic Regression Start
print("Logestic Regression Start")

# load or create your dataset
print('Load data...')

# c = np.array([1,0.5,0.1,0.05,0.01,0.005,0.001])
# for t in range(0,len(c)):
lm = LogisticRegression(penalty='l2', C=0.001) # logestic model construction
lm.fit(transformed_training_matrix, np.ravel(y_train))  # fitting the data

# 逻辑回归的预测标签
y_pred_label = lm.predict(transformed_testing_matrix)    # For testing data
save_result["predict"] = y_pred_label
# y_pred_est = lm.predict_proba(transformed_training_matrix)   # Give the probabilty on each label
# 逻辑回归的预测标签的概率
y_pred_est = lm.predict_proba(transformed_testing_matrix)
# print("transformed_testing_matrix：", transformed_testing_matrix)#one-hot:1968*6300(63*100,100个1)
# Give the probabilty on each label

# 存储probability
probList = []
print('number of testing data is ' + str(len(y_pred_label)))
for i in range(len(y_pred_est)):
    probList.append(y_pred_est[i][0])
probNarray = np.array(probList)
save_result["prob"] = probNarray
# 评估方式1：
# calculate predict accuracy
#     num = 0
#     for i in range(0,len(y_pred_label)):
#         if y_test[i] == y_pred_label[i]:
#             if y_train[i] == y_pred_label[i]:
#                 num += 1
#     print("*******************")
#     print(y_test)
#     print(y_pred_label)
#     print('penalty parameter is '+ str(c[t]))
#     print("prediction accuracy is " + str((num)/len(y_pred_label)))
#
#     df_prob = pd.DataFrame(
#         y_pred_est,
#         columns=['Death', 'Survived'])
#     fpr, tpr, thresholds = roc_curve(
#         y_test, df_prob['Survived'])
#     # find the area under the curve (auc) for the
#     # ROC
#     roc_auc = auc(fpr, tpr)
#     plt.title(
#         'Receiver Operating Characteristic Curve')
#     plt.plot(fpr, tpr, 'black',
#              label='AUC = %0.2f' % roc_auc)
#     plt.legend(loc='lower right')
#     plt.plot([0, 1], [0, 1], 'r--')
#     plt.xlim([-0.1, 1.1])
#     plt.ylim([-0.1, 1.1])
#     plt.ylabel('True Positive Rate (TPR)')
#     plt.xlabel('False Positive Rate (FPR)')
#     plt.show()
#
# 	# Calculate the Normalized Cross-Entropy
# 	# for testing data
NE = (-1) / len(y_pred_est) * sum(((1+y_test)/2 * np.log(y_pred_est[:,1]) +  (1-y_test)/2 * np.log(1 - y_pred_est[:,1])))
print("Normalized Cross Entropy " + str(NE))

#评估方式2：

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.figure(figsize=(12, 8))
j = 1
for i in thresholds:
    # print("y_pred_est",y_pred_est)
    y_test_predictions_high_recall = y_pred_est[:, 1] > i
    plt.subplot(3, 3, j)
    j += 1

    cnf_matrix = confusion_matrix(y_test, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)
    # print("thresholds: ", i)
    # print('Accurance: ',
    #        (cnf_matrix[0, 0] + cnf_matrix[1, 1])/(cnf_matrix[1, 0] + cnf_matrix[1, 1]+ cnf_matrix[0, 1]+ cnf_matrix[0, 0]))
    recall = cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])
    # print('Recall: ',recall)

    precision = cnf_matrix[1, 1] / (cnf_matrix[0, 1] + cnf_matrix[1, 1])
    # print('Precision:  ', precision)

    print('F1-score:',
          2 * (precision * recall) / (precision + recall)
          )

    # 画出混淆矩阵
    class_names = [0, 1]
    # plot_confusion_matrix是一个自定义的绘制混淆矩阵图表的函数
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %s' % i)
plt.show()
save_result.to_json("result1.json", orient="records")# 关键

# mongoimport --db eduRedar --jsonArray --collection model_results --file /Users/vis/Documents/DataVisLab/教育数据/GBDT+LR/model_results.json