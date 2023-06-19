#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   data_merge.py
@Time    :   2023/05/11 18:26:46
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   数据合并
'''
import os

import pandas as pd
import multiprocessing as mp

def txt_to_csv(file_path: str) -> set:
    file_name = file_path.split('/')[-1].split('.')[0]
    df = pd.read_csv(file_path, sep='\t', low_memory=False, encoding='utf-8')
    print(df.columns, df.index)
    return set(df['eventname'].to_list())
    # df.to_csv(f'data/csv2/{file_name}.csv', index=False)
    
def merge(merged_df: pd.DataFrame) -> pd.DataFrame:
    base_df = pd.DataFrame()
    
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(base_dir, 'data/raw')
    file_path_list = [os.path.join(raw_data_dir, file_name) for file_name in filter(lambda x: x.startswith('abcd_'), os.listdir(raw_data_dir))]
    
    res_set = set()
    for file_path in file_path_list:
        res_set.update(txt_to_csv(file_path))
        print(res_set)
    print(res_set)
    # 多进程加速
    # with mp.Pool(mp.cpu_count() // 4) as pool:
    #     async_csv = pool.map_async(txt_to_csv, file_path_list)
    #     async_csv.get()
        
    # res_set = set()
    # for e in async_csv:
    #     res_set.update(e)
    # print(res_set)