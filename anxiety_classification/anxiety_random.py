#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   anxiety_random.py
@Time    :   2023/06/16 15:20:40
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   焦虑分类模型,使用随机森林
'''
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from feature_extract import data_dir
from imblearn.over_sampling import RandomOverSampler

# 随机数种子
random_num = 15

# 加载示例数据集
data = pd.read_csv(f'{data_dir}/dataset_processed.csv')

# 划分训练集和测试集
X, y = data.drop(['subjectkey', 'eventname', 'label'], axis=1), data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_num)

# 随机过采样
# ros = RandomOverSampler(random_state=random_num)
# X_train, y_train = ros.fit_resample(X_train, y_train)

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=random_num)

# 训练模型
rf_classifier.fit(X_train, y_train)

# 预测测试集
y_pred = rf_classifier.predict(X_test)

# 计算准确率，F1值和AUC值
accuracy, f1, auc = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')

