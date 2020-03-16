# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:29:04 2020

@author: lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, accuracy_score, auc

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei',style='darkgrid') 

loan_LC=pd.read_csv('LC.csv')
loan_LC = loan_LC.set_index('ListingId')
loan_LC['借款成功日期'] = pd.to_datetime(loan_LC['借款成功日期'])

loan_LP = pd.read_csv('LP.csv')
loan_LP['到期日期'] = pd.to_datetime(loan_LP['到期日期'])
loan_LP['还款日期'] = pd.to_datetime(loan_LP['还款日期'].replace('\\N', '2017-1-31'))
loan_LP['recorddate'] = datetime.datetime(2017, 2, 22)
loan_LP['逾期天数'] = (loan_LP['还款日期'] - loan_LP['到期日期'])/np.timedelta64(1, 'D')
loan_LP['逾期天数'] = np.where(loan_LP['逾期天数'] < 0, 0, loan_LP['逾期天数'])

# 从未结清贷款中提取已还期数
loan_unsettled = loan_LP[(loan_LP['还款状态'] != 3) & (loan_LP['剩余利息'] > 0)].groupby('ListingId')['期数'].min().reset_index()
loan_unsettled['已还期数'] = loan_unsettled['期数'] - 1
del loan_unsettled['期数']
loan_unsettled = loan_unsettled.set_index('ListingId')
loan1 = pd.concat([loan_LC, loan_unsettled], axis=1, join='outer')
loan1.loc[loan1['已还期数'].isnull(), '已还期数'] = loan1[loan1['已还期数'].isnull()]['借款期限']
# 本笔贷款历史还款情况汇总
loan_LP['逾期天数'].replace(0, np.nan, inplace=True)
loan_delay = loan_LP.groupby('ListingId')['逾期天数'].count()
loan_delay.name = '本笔已逾期次数'
loan_LP['逾期天数'].replace(np.nan, 0, inplace=True)
loan2 = pd.concat([loan1, loan_delay], axis=1, join='outer').fillna(0)

group_loan1 = loan_LP.groupby('ListingId').agg({'剩余本金': 'sum', '剩余利息': 'sum', '还款状态': 'max'})
group_loan2 = loan_LP.drop(columns=loan_LP.columns[1: 10]).groupby('ListingId').max()
group_loan1.rename(columns={'剩余本金': '剩余未还本金', '剩余利息': '剩余未还利息'}, inplace=True)
group_loan = pd.concat([group_loan1, group_loan2], axis=1)
loan = pd.concat([loan2, group_loan], axis=1, join='outer')
loan['历史逾期还款占比'] = (100 * loan['历史逾期还款期数']/(loan['历史逾期还款期数'] + loan['历史正常还款期数'])).round(2).fillna(0)
loan['年龄段'] = pd.cut(loan['年龄'], bins=[15, 20, 25, 30, 35, 40, 45, 50, 60])
loan['借款期限段'] = pd.cut(loan['借款期限'], bins=[3*i for i in range(9)])
loan['target'] = np.where((loan['逾期天数'] > 60) & (loan['剩余未还利息'] > 0), 1, 0)
loan['借款成功日期'] =(loan['借款成功日期']-pd.to_datetime('2015-01-01'))/ (np.timedelta64(1,'D'))

# 缺失值筛查
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_pct = 100 * mis_val/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_pct], axis=1)
    mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'}, inplace=True)
    mis_val_table = mis_val_table[mis_val_table.iloc[:, 1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    return mis_val_table
missing_loan = missing_values_table(loan)
missing_loan

le = LabelEncoder()
for col in loan:
    if loan[col].dtype == 'object':
        if len(list(loan[col].unique())) <= 2:
            le.fit(loan[col])
            loan[col] = le.transform(loan[col])
loan = pd.get_dummies(loan)

# 本次计算现将贷后数据加入计算，后期会剔除贷后特征再建模。
corrs = loan.corr()
corrs['target'].sort_values()

# 核实目标相关性较高的特征之间是否存在共线性
ext_loan = loan[['target', '本笔已逾期次数', '逾期天数', '剩余未还本金', '剩余未还利息']]
plt.figure(figsize=(8, 6))
sns.heatmap(ext_loan.corr(), annot=True)

""" 
target数据来源于逾期天数，故相关性较高，需剔除
本笔已逾期次数与逾期天数非强相关，剩余未还本金与剩余未还利息与逾期天数以及已逾期次数相关性不高，故保留
""" 
del loan['逾期天数']
loan_copy = loan.copy()
del loan_copy['历史逾期还款期数']
del loan_copy['淘宝认证']
del loan_copy['借款金额']
del loan_copy['历史逾期还款占比']
##############################################LR模型
X, y = loan_copy.loc[:, loan_copy.columns != 'target'], loan_copy['target']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred_prob = logreg.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()


"""
常规LR数据准确率较高，精度也较高，召回率偏低，有一定程度的误拒。参数上C=100表现最好。l1惩罚因子更有效。
信贷不平衡较严重，出错代价较大，同时市场竞争剧烈，获客也较为困难。下面尝试将分类权重调整为均衡。
修改后跑出来的数据显示回收率提高较多，显示误拒的情况减少，但精度降低较多，显示更多的坏客户未识别。
整体AUC评分虽少量提升，但对贷款而言，精度更重要，模型应该选常规LR。
"""
def modelfit(alg, df, performCV=True, printFeatureImportance=True, cv_folds=5):
    X, y = df.loc[:, df.columns != 'target'], df['target']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    alg.fit(x_train, y_train)
    
    df_prediction = alg.predict(x_test)
    df_predprob = alg.predict_proba(x_test)[:, 1]
    
    if performCV:
        cv_score = cross_val_score(alg, x_train, y_train, cv=cv_folds, scoring='roc_auc')
    
    print('\nModel Report')
    print('Accuracy : %.4g' % accuracy_score(y_test, df_prediction))
    print('Precision : %.4g' % precision_score(y_test, df_prediction))
    print('Recall : %.4g' % recall_score(y_test, df_prediction))
    print('AUC Score (train) : %f' % roc_auc_score(y_test, df_predprob))
    
    if performCV:
        print('CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g' % (np.mean(cv_score), 
                np.std(cv_score), np.min(cv_score), np.max(cv_score)))
    
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, x_train.columns).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importances Score')
LR1 = LogisticRegression(C=100, penalty='l1', class_weight='balanced')
modelfit(LR1, loan_copy, printFeatureImportance=False)
print('----')
LR2 = LogisticRegression(C=100, penalty='l2', class_weight='balanced')
modelfit(LR2, loan_copy, printFeatureImportance=False)
############################################RF#####################
# 样本量相对于特征数较多，随机森林模型的泛化能力优于逻辑回归模型。
'''rf = RandomForestClassifier(random_state=4)
modelfit(rf, loan_copy)

########################################################
# GBDT模型的精度比RF稍差，正确率稍高，召回率优势明显，整体表现优于RF模型，泛化能力更是好于LR模型。
modelfit(GradientBoostingClassifier(random_state=10), loan_copy)

# GBDT模型调参。先确定n_estimators，20-80，若小于20，需调低learning_rate, 若大于80，需调高，其他正常。
from sklearn.ensemble import GradientBoostingClassifier
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.2, min_samples_split=1000, 
                                    min_samples_leaf=50, max_depth=8, 
                                    max_features='sqrt', subsample=0.8, random_state=10), 
                        param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch1.fit(x_train, y_train)
gsearch1.best_params_, gsearch1.best_score_
"""
再调整max_depth和min_samples_split，节点数和最小样本数均与现有样本量有关。后者一般是样本量的0.5%-1%。
因运算花费时间较长，故步长应设置的大些，后逐步微调。前次以最高值14和1500为最优值，本次以该值为起点再次运算。
""" 
param_test2 = {'max_depth': range(14, 20, 2), 'min_samples_split': range(1500, 2101, 200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators=80, learning_rate=0.2, min_samples_leaf=50,  
                                    max_features='sqrt', subsample=0.8, random_state=10), 
                        param_grid=param_test2, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch2.fit(x_train, y_train)
gsearch2.best_params_, gsearch2.best_score_
'''









