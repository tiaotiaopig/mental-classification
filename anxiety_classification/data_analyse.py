#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   data_analyse.py
@Time    :   2023/06/16 20:16:51
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   对数据进行分析
'''

import pandas as pd
from feature_extract import data_dir

labels = pd.read_csv(f'{data_dir}/extract/label.csv', sep=',', low_memory=False, encoding='utf-8')
# 焦虑的比例
print(labels['label'].value_counts(normalize=True))