#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   data_transfer.py
@Time    :   2023/05/10 12:24:22
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   数据转储
'''
import os
import pandas as pd

def data_preprocess(file_path: str):
    """
    数据预处理
    """
    file_name = file_path.split('/')[-1].split('.')[0]
    csv_data = pd.read_csv(file_path, sep='\t', skiprows=[1], parse_dates=['interview_date'], low_memory=False, encoding='utf-8')
    
    csv_data.to_csv(f'data/csv/{file_name}.csv', index=False)
    
def create_table_from_txt(file_path: str) -> None:
    """
    根据文本文件创建表sql
    """
    file_name = file_path.split('/')[-1].split('.')[0]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        col_name = f.readline().strip().split('\t')
        col_desc = f.readline().strip().split('\t')
    
    sql_head = f'DROP TABLE IF EXISTS `{file_name}`;\nCREATE TABLE IF NOT EXISTS `{file_name}` (\n'
    sql_body = []
    
    names = [name.strip('\"') for name in col_name]
    descs = [desc.replace('\"', '') for desc in col_desc]
    
    # print('a')
    # 前三个数据是整型
    for name, desc in zip(names[:3], descs[:3]):    
        sql_body.append(f'`{name}` int default null COMMENT \"{desc}\"\n')
        
    # 四五位数据是文本
    for name, desc in zip(names[3:5], descs[3:5]):
        sql_body.append(f'`{name}` varchar(50) default null COMMENT \"{desc}\"\n')
    
    # 六位数据是日期
    for name, desc in zip(names[5:6], descs[5:6]):
        sql_body.append(f'`{name}` date default null COMMENT \"{desc}\"\n')
        
    # 七位数据是整型
    for name, desc in zip(names[6:7], descs[6:7]):
        sql_body.append(f'`{name}` int default null COMMENT \"{desc}\"\n')
        
    # 八位数据是文本
    for name, desc in zip(names[7:8], descs[7:8]):
        sql_body.append(f'`{name}` char(1) default null COMMENT \"{desc}\"\n')
        
    # 九位数据是文本
    for name, desc in zip(names[8:9], descs[8:9]):
        sql_body.append(f'`{name}` varchar(50) default null COMMENT \"{desc}\"\n')
        
    # 其余是浮点型
    for name, desc in zip(names[9:-1], descs[9:-1]):
        sql_body.append(f'`{name}` float default null COMMENT \"{desc}\"\n')
        
    # 最后一位是文本
    for name, desc in zip(names[-1:], descs[-1:]):
        sql_body.append(f'`{name}` varchar(255) default null COMMENT \"{desc}\"\n')
        
    # for name, desc in zip(col_name, col_desc):
    #     name = name.strip('\"')
    #     sql_body.append(f'`{name}` varchar(100) COMMENT \"{desc}\"\n')
    
    sql_body = ','.join(sql_body)
    
    sql_tail = ') ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;\n\n'
    
    with open(f'data/create_table.sql', 'a+', encoding='utf-8') as f:
        f.write(sql_head + sql_body + sql_tail)
        
    
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(base_dir, 'data/raw')
    file_path_list = [os.path.join(raw_data_dir, file_name) for file_name in os.listdir(raw_data_dir)]
    
    # 多进程加速
    # with mp.Pool(mp.cpu_count() // 2) as pool:
    #     async_csv = pool.map_async(data_preprocess, file_path_list)
    #     async_sql = pool.map_async(create_table_from_txt, file_path_list)
        
    #     async_csv.get()
    #     async_sql.get()
    for file_name in os.listdir(raw_data_dir):
        file_path = os.path.join(raw_data_dir, file_name)
        # 转成csv
        # data_preprocess(file_path)
        # 生成建表语句
        create_table_from_txt(file_path)