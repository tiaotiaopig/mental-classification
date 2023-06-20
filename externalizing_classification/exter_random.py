#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   exter_random.py
@Time    :   2023/06/19 20:47:53
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   内向型和外向型分类模型,使用随机森林
'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data_process import processed_dir
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV

# 随机数种子
random_num = 15

# 加载示例数据集
data = pd.read_csv(f'{processed_dir}/dataset_im.csv')

# 划分训练集和测试集
X, y = data.drop(['subjectkey', 'eventname', 'label'], axis=1), data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_num)

# 随机过采样
# ros = RandomOverSampler(random_state=random_num)
# X_train, y_train = ros.fit_resample(X_train, y_train)

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(random_state=random_num)

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# 训练模型
# 使用网格搜索进行参数调整
grid_search = GridSearchCV(rf_classifier, param_grid,verbose=1, scoring='accuracy', n_jobs=-1, cv=5)
grid_search.fit(X_train, y_train)

# 输出最优参数和最优模型

print("最优参数:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# 预测测试集
y_pred = best_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')