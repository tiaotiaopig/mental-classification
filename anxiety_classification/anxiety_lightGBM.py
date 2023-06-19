#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   anxiety_lightGBM.py
@Time    :   2023/06/16 12:54:38
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   焦虑分类模型,使用lightGBM
'''
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from feature_extract import data_dir
# 读取数据并准备特征和标签
data = pd.read_csv(f'{data_dir}/dataset_processed.csv')
features = data.drop(['subjectkey', 'eventname', 'label'], axis=1)
labels = data['label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=15)

# 随机过采样
# ros = RandomOverSampler(random_state=15)
# X_train, y_train = ros.fit_resample(X_train, y_train)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# 定义训练参数和模型配置
params = {
    'objective': 'binary',  # 二分类任务
    'metric': 'binary_logloss',  # 评价指标为二分类的对数损失
    'boosting_type': 'gbdt',  # 使用梯度提升树
    'num_leaves': 31,  # 叶子节点数目
    'learning_rate': 0.10,  # 学习率
    'feature_fraction': 0.9,  # 使用特征的比例
    'bagging_fraction': 0.8,  # 使用数据的比例
    'bagging_freq': 5,  # 每5轮迭代执行bagging
    'verbose': 0,  # 显示训练输出信息
    'force_row_wise': True,  # 按行遍历数据
}

model = lgb.train(params, train_data, valid_sets=[train_data, test_data], num_boost_round=100, callbacks=[lgb.early_stopping(10)])

y_pred = model.predict(X_test)
y_pred_binary = [round(pred) for pred in y_pred]  # 将概率转换为二分类标签

accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)