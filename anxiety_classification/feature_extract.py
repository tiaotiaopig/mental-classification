#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   feature_extract.py
@Time    :   2023/06/12 15:37:28
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   从多个表中提取特征
'''
import re
import os
import json
import pandas as pd
from typing import List

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{base_dir}/../data'

def fea_extract(file_path: str, fea_name: List) -> pd.DataFrame:
    '''
    按照特征名称，从表中提取特征
    '''
    id_fea = ['subjectkey', 'eventname']
    file_name = file_path.split('/')[-1].split('.')[0]
    df = pd.read_csv(file_path, sep='\t',skiprows=[1], low_memory=False, encoding='utf-8')
    df = df[id_fea + fea_name]
    df.to_csv(f'{data_dir}/extract/{file_name}.csv', index=False)
    return df

def sample_label() -> pd.DataFrame:
    '''
        abcd_ksad501和abcd_ksad01有1就焦虑，全为0就正常
    '''
    df_ksad501 = pd.read_csv(f'{data_dir}/extract/abcd_ksad501.csv', sep=',', low_memory=False, encoding='utf-8')
    df_ksad01 = pd.read_csv(f'{data_dir}/extract/abcd_ksad01.csv', sep=',', low_memory=False, encoding='utf-8')
    
    df_ksad501['label'] = df_ksad501.apply(lambda x: (x == 1).any(), axis=1)
    df_ksad01['label'] = df_ksad01.apply(lambda x: (x == 1).any(), axis=1)
    
    df_ksad01.to_csv(f'{data_dir}/validate/ksad01.csv', index=False)
    df_ksad501.to_csv(f'{data_dir}/validate/ksad501.csv', index=False)
    
    df_ksad501 = df_ksad501[df_ksad501['eventname'] == '2_year_follow_up_y_arm_1'][['subjectkey', 'label']]
    df_ksad01 = df_ksad01[df_ksad01['eventname'] == '2_year_follow_up_y_arm_1'][['subjectkey', 'label']]
    
    df_merge = pd.merge(df_ksad501, df_ksad01, on=['subjectkey'], suffixes=('_ksad501', '_ksad01'))
    df_merge['label'] = df_merge.apply(lambda x: 1 if x['label_ksad501'] or x['label_ksad01'] else 0, axis=1)
    
    df_merge.to_csv(f'{data_dir}/extract/label.csv', index=False)
    
    return df_merge[['subjectkey', 'label']]
    
def sample_filter_KSADS5() -> pd.DataFrame:
    '''
        Excluded participants with any of KSADS-5 anxiety diagnosis.
        主要是abcd_ksad501和abcd_ksad01有1就排除(基线)
    '''
    df_ksad501 = pd.read_csv(f'{data_dir}/extract/abcd_ksad501.csv', sep=',', low_memory=False, encoding='utf-8')
    df_ksad01 = pd.read_csv(f'{data_dir}/extract/abcd_ksad01.csv', sep=',', low_memory=False, encoding='utf-8')
    
    # 执行过滤操作
    df_ksad501 = df_ksad501[
        (df_ksad501['eventname'] == 'baseline_year_1_arm_1') & 
        (~(df_ksad501 == 1.0).any(axis=1))
        ][['subjectkey', 'eventname']]
        
    df_ksad01 = df_ksad01[
        (df_ksad01['eventname'] == 'baseline_year_1_arm_1') & 
        (~(df_ksad01 == 1.0).any(axis=1))
        ][['subjectkey', 'eventname']]
    
    # 合并两个表，只保留subjectkey和eventname(并的关系)
    return pd.merge(df_ksad501, df_ksad01, on=['subjectkey', 'eventname'])

def sample_filter_QC() -> pd.DataFrame:
    '''
        Excluded participants without neuroimaging data or did not pass QC.
    '''
    # 保留影像质量好的
    df_abcd_mrfindings02 = pd.read_csv(f'{data_dir}/raw/abcd_mrfindings02.txt', sep='\t', skiprows=[1], low_memory=False, encoding='utf-8')
    
    df_abcd_mrfindings02 = df_abcd_mrfindings02[
        (df_abcd_mrfindings02['eventname'] == 'baseline_year_1_arm_1') & 
        (df_abcd_mrfindings02['mrif_score'].isin([1, 2]))
        ][['subjectkey', 'eventname']]
    
    # 排除轻度脑损伤
    df_abcd_tbi01 = pd.read_csv(f'{data_dir}/raw/abcd_tbi01.txt', sep='\t', skiprows=[1], low_memory=False, encoding='utf-8')
    
    df_abcd_tbi01 = df_abcd_tbi01[
        (df_abcd_tbi01['eventname'] == 'baseline_year_1_arm_1') &
        (df_abcd_tbi01['tbi_ss_worst_overall'].isin([1, 2]))
    ][['subjectkey', 'eventname']]
    
    # df_abcd_lsstbi01 = pd.read_csv(f'{base_dir}/data/raw/abcd_lsstbi01.txt', sep='\t', skiprows=[1], low_memory=False, encoding='utf-8')
    
    # df_abcd_lsstbi01 = df_abcd_lsstbi01[
    #     (df_abcd_lsstbi01['eventname'] == 'baseline_year_1_arm_1') &
    #     ~(df_abcd_lsstbi01['tbi_ss_worst_overall_l'].isin([1, 2]))
    # ][['subjectkey', 'eventname']]
    
    # 排除既往存在意识丧失
    df_abcd_mx01 = pd.read_csv(f'{data_dir}/raw/abcd_mx01.txt', sep='\t', skiprows=[1], low_memory=False, encoding='utf-8')
    
    df_abcd_mx01 = df_abcd_mx01[
        (df_abcd_mx01['eventname'] == 'baseline_year_1_arm_1') &
        (df_abcd_mx01['medhx_6j'] != 1) & (df_abcd_mx01['medhx_6p'] != 1)
    ][['subjectkey', 'eventname']]
    
    # 排除智力小于70的个体
    df_abcd_tbss01 = pd.read_csv(f'{data_dir}/raw/abcd_tbss01.txt', sep='\t', skiprows=[1], low_memory=False, encoding='utf-8')
    
    df_abcd_tbss01 = df_abcd_tbss01[
        (df_abcd_tbss01['eventname'] == 'baseline_year_1_arm_1') &
        (df_abcd_tbss01['nihtbx_totalcomp_agecorrected'] >= 70)
    ][['subjectkey', 'eventname']]
    
    df_merge_1 = pd.merge(df_abcd_mrfindings02, df_abcd_tbi01, on=['subjectkey', 'eventname'])
    df_merge_2 = pd.merge(df_abcd_tbss01, df_abcd_mx01, on=['subjectkey', 'eventname'])
    return pd.merge(df_merge_1, df_merge_2, on=['subjectkey', 'eventname'])

def test() -> None:
    abcd_drsip201 = pd.read_csv(f'{data_dir}/raw/mrirscor02.txt', sep='\t', skiprows=[1], low_memory=False, encoding='utf-16')
    list_col = abcd_drsip201.columns.to_list()
    pattern = re.compile(r'rsfmri_cor_ngd_.+')
    list_col = list(filter(lambda x: pattern.match(x), list_col))
    json.dump(list_col, open(f'{data_dir}/rsfmri_cor_ngd.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    test()
    
    # 1. 特征提取
    df_list = []
    file_feature = json.load(open(f'{data_dir}/feature.json', 'r', encoding='utf-8'))
    for file_name, file_fea in file_feature.items():
        df_sub = fea_extract(f'{data_dir}/raw/{file_name}', file_fea)
        # 排除标签量表的信息
        if file_name not in ['abcd_ksad501.txt', 'abcd_ksad01.txt']:
            df_list.append(df_sub)
    
    # 2. 异常数据排除
    df_base = sample_filter_KSADS5()
    
    df_base2 = sample_filter_QC()
    
    df_base = pd.merge(df_base, df_base2, on=['subjectkey', 'eventname'])
    
    df_base.to_csv(f'{data_dir}/extract/merge_base.csv', index=False)
    
    # 3. 量表合并
    for df_sub in df_list:
        df_base = pd.merge(df_base, df_sub, how='left', on=['subjectkey', 'eventname'])
          
    # 4. 汇聚量表，一些列要合并
    merge_cols = ["devhx_14a3_p", "devhx_14b3_p", "devhx_14c3_p", "devhx_14d3_p",
        "devhx_14e3_p", "devhx_14f3_p", "devhx_14g3_p", "devhx_14h3_p"]
    df_base['devhx_14a3_ocpr'] = df_base["devhx_14a3_p"] + df_base["devhx_14b3_p"] + df_base["devhx_14c3_p"] + df_base["devhx_14d3_p"] + df_base["devhx_14e3_p"] + df_base["devhx_14f3_p"] + df_base["devhx_14g3_p"] + df_base["devhx_14h3_p"]
    df_base = df_base.drop(merge_cols, axis=1)
    
    merge_cols = ['devhx_10a3_p', 'devhx_10b3_p', 'devhx_10c3_p', 'devhx_10d3_p', 'devhx_10e3_p', 'devhx_10f3_p', 'devhx_10g3_p', 'devhx_10h3_p', 'devhx_10i3_p', 'devhx_10j3_p', 'devhx_10k3_p', 'devhx_10l3_p', 'devhx_10m3_p']
    df_base['devhx_14a3_mmcdp'] = df_base['devhx_10a3_p'] + df_base['devhx_10b3_p'] + df_base['devhx_10c3_p'] + df_base['devhx_10d3_p'] + df_base['devhx_10e3_p'] + df_base['devhx_10f3_p'] + df_base['devhx_10g3_p'] + df_base['devhx_10h3_p'] + df_base['devhx_10i3_p'] + df_base['devhx_10j3_p'] + df_base['devhx_10k3_p'] + df_base['devhx_10l3_p'] + df_base['devhx_10m3_p']
    df_base = df_base.drop(merge_cols, axis=1)
    
    merge_cols = ['devhx_9_tobacco', 'devhx_9_alcohol', 'devhx_9_marijuana', 'devhx_9_coc_crack', 'devhx_9_her_morph', 'devhx_9_oxycont']
    df_base['devhx_9_maudp'] = df_base['devhx_9_tobacco'] + df_base['devhx_9_alcohol'] + df_base['devhx_9_marijuana'] + df_base['devhx_9_coc_crack'] + df_base['devhx_9_her_morph'] + df_base['devhx_9_oxycont']
    df_base = df_base.drop(merge_cols, axis=1)
    
    merge_cols = ['demo_fam_exp1_v2', 'demo_fam_exp2_v2', 'demo_fam_exp3_v2', 'demo_fam_exp4_v2', 'demo_fam_exp5_v2', 'demo_fam_exp6_v2', 'demo_fam_exp7_v2']
    df_base['demo_fam_ffapr'] = df_base['demo_fam_exp1_v2'] + df_base['demo_fam_exp2_v2'] + df_base['demo_fam_exp3_v2'] + df_base['demo_fam_exp4_v2'] + df_base['demo_fam_exp5_v2'] + df_base['demo_fam_exp6_v2'] + df_base['demo_fam_exp7_v2']
    df_base = df_base.drop(merge_cols, axis=1)
    
    # 3. 缺失值比例超过10%,删去
    df_base = df_base.dropna(thresh=int(df_base.shape[1] * 0.9))
    df_base.to_csv(f'{data_dir}/extract/merge.csv', index=False)
    
    # 4. 标签提取
    df_label = sample_label()
    
    # 5. 合并标签
    df_dataset = pd.merge(df_base, df_label, on=['subjectkey'])
    df_dataset.to_csv(f'{data_dir}/extract/dataset.csv', index=False)
    
    # 缺失值填充
    df_dataset = df_dataset.fillna(df_dataset.mode().iloc[0])
    
    # 对类别特征进行One-Hot编码
    category_features = json.load(open(f'{base_dir}/../data/settings.json', 'r'))['category_features']
    df_dataset = pd.get_dummies(df_dataset, columns=category_features)
    
    # 保存数据集
    df_dataset.to_csv(f'{data_dir}/dataset_processed.csv', index=False)