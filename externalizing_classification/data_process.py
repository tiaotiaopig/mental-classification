#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   data_process.py
@Time    :   2023/06/18 19:51:32
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   外向性分类数据处理
'''

from anxiety_classification.feature_extract import *

if __name__ == "__main__":
   
    smri = pd.read_csv(f'{data_dir}/raw/mrirscor02.txt', sep='\t', skiprows=[1], low_memory=False, encoding='utf-16')
    list_col = smri.columns.to_list()
    pattern = re.compile(r'rsfmri_cor_ngd_.+')
    list_col = list(filter(lambda x: pattern.match(x), list_col))
    json.dump(list_col, open(f'tmp.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    