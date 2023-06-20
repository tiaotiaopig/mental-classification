#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   data_process.py
@Time    :   2023/06/18 19:51:32
@Author  :   LiFeng
@Contact :   2807229316@qq.com
@Desc    :   外向性分类数据处理
'''

import os
import json
import pandas as pd
from typing import List
from anxiety_classification.feature_extract import sample_filter_QC

base_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = f'{base_dir}/../data/raw'

data_dir, conf_dir, processed_dir = f'{base_dir}/data', f'{base_dir}/conf', f'{base_dir}/processed'


def fea_extract(fea_path: str) -> None:
    """
    从原始txt文件中提取特征,并保存到csv文件中
    """
    id_fea = ['subjectkey', 'eventname']
    file_feature = json.load(open(f'{conf_dir}/feature.json', 'r', encoding='utf-8'))
    for file_name, fea_name in file_feature.items():
        encoding = 'utf-16' if file_name in ('abcd_betnet02.txt', 'mrirscor02.txt') else 'utf-8'
        df = pd.read_csv(f'{raw_dir}/{file_name}', sep='\t',skiprows=[1], low_memory=False, encoding=encoding)
        
        df = df[id_fea + fea_name]
        df.to_csv(f'{data_dir}/{file_name.split(".")[0]}.csv', index=False, encoding='utf-8')
  
def label(limit: int=65) -> pd.DataFrame:
    '''
        获取内向型、外向型和正常标签
        正常0,externalizing 1,internalizing 2
    '''
    def get_label(x: pd.Series) -> int:
        
        if x['cbcl_scr_syn_external_t'] < limit and x['cbcl_scr_syn_internal_t'] < limit:
            return 0
        return 1 if x['cbcl_scr_syn_external_t'] > x['cbcl_scr_syn_internal_t'] else 2  
    
    df = pd.read_csv(f'{data_dir}/abcd_cbcls01.csv', sep=',', low_memory=False, encoding='utf-8')
    df = df[df['eventname'] == '2_year_follow_up_y_arm_1']
    df['label'] = df.apply(get_label, axis=1)
    df =  df[['subjectkey', 'label']]
    df.to_csv(f'{processed_dir}/label.csv', index=False)
    return df

def exclude_exter_inter() -> pd.DataFrame:
    '''
        排除基线时外向性和内向性
    '''
    df = pd.read_csv(f'{data_dir}/abcd_cbcls01.csv', sep=',', low_memory=False, encoding='utf-8')
    df = df[(df['eventname'] == 'baseline_year_1_arm_1') & (df['cbcl_scr_syn_external_t'] < 65) & (df['cbcl_scr_syn_internal_t'] < 65)]
    return df[['subjectkey', 'eventname']]
     
def image_exclusion() -> pd.DataFrame:
    '''
        abcd_imgincl01（ baseline: imgincl_t1w_include=0; imgincl_t2w_include=0; imgincl_dmri_include=0; imgincl_rsfmri_include=0）
    '''
    df = pd.read_csv(f'{data_dir}/abcd_imgincl01.csv', sep=',', low_memory=False, encoding='utf-8')
    df = df[(df['eventname'] == 'baseline_year_1_arm_1') & (df['imgincl_t1w_include'] != 0) & (df['imgincl_t2w_include'] != 0) & (df['imgincl_dmri_include'] != 0) & (df['imgincl_rsfmri_include'] != 0)]
    return df[['subjectkey', 'eventname']]

if __name__ == "__main__":
    
    setting = json.load(open(f'{conf_dir}/setting.json', 'r', encoding='utf-8'))
    # 1. 特征提取
    # fea_extract(f'{conf_dir}/feature.json')
    # 2. 影像排除,基线内外排除
    df_base = pd.merge(exclude_exter_inter(), image_exclusion(), on=['subjectkey', 'eventname'], how='inner')
    df_base.to_csv(f'{processed_dir}/base_baseline.csv', index=False)
    
    # 未通过QC的样本排除
    df_base = pd.merge(df_base, sample_filter_QC(), on=['subjectkey', 'eventname'], how='inner')
    df_base.to_csv(f'{processed_dir}/base_exclude.csv', index=False)
    
    # 合并多维度特征
    for name in setting['feature_names']:
        df_sub = pd.read_csv(f'{data_dir}/{name}.csv', sep=',', low_memory=False, encoding='utf-8')
        df_base = pd.merge(df_base, df_sub, on=['subjectkey', 'eventname'], how='left')
    # abcd_stq01单独合并
    df_sub = pd.read_csv(f'{data_dir}/abcd_stq01.csv', sep=',', low_memory=False, encoding='utf-8')
    df_sub = df_sub[df_sub['eventname'] == '3_year_follow_up_y_arm_1']
    df_sub['screen_time'] = ((df_sub['screentime_wkdy_typical_hr'] * 60 + df_sub['screentime_wkdy_typical_min']) * 5  + (df_sub['screentime_wknd_typical_hr'] * 60 + df_sub['screentime_wknd_t_min']) * 2) / 7
    df_base = pd.merge(df_base, df_sub[['subjectkey', 'screen_time']], on=['subjectkey'], how='left')
    df_base.to_csv(f'{processed_dir}/base_multi.csv', index=False)
    
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
    
    merge_cols = ['famhx_ss_parent_alc_p', 'famhx_ss_parent_dg_p', 'famhx_ss_parent_dprs_p', 'famhx_ss_parent_ma_p', 'famhx_ss_parent_vs_p', 'famhx_ss_parent_trb_p', 'famhx_ss_parent_nrv_p', 'famhx_ss_parent_hspd_p', 'famhx_ss_parent_scd_p']
    df_base['famhx_ss_ppdpr'] = df_base['famhx_ss_parent_alc_p'] + df_base['famhx_ss_parent_dg_p'] + df_base['famhx_ss_parent_dprs_p'] + df_base['famhx_ss_parent_ma_p'] + df_base['famhx_ss_parent_vs_p'] + df_base['famhx_ss_parent_trb_p'] + df_base['famhx_ss_parent_nrv_p'] + df_base['famhx_ss_parent_hspd_p'] + df_base['famhx_ss_parent_scd_p']
    df_base = df_base.drop(merge_cols, axis=1)
    
    # 合并影像的数据
    for name in setting['image_names']:
        df_sub = pd.read_csv(f'{data_dir}/{name}.csv', sep=',', low_memory=False, encoding='utf-8')
        df_base = pd.merge(df_base, df_sub, on=['subjectkey', 'eventname'], how='left')
        
    df_base.to_csv(f'{processed_dir}/base_image.csv', index=False)
    
    # 3. 缺失值比例超过10%的特征删除
    df_base = df_base.dropna(thresh=int(df_base.shape[1] * 0.9))
    df_base.to_csv(f'{processed_dir}/base_10.csv', index=False)
    
    # 4. 合并标签
    df_base = pd.merge(label(), df_base, on=['subjectkey'], how='left')
    
    df_base.to_csv(f'{processed_dir}/base_before_fill.csv', index=False)
    # 5. 缺失值填充(使用均值填充)
    df_base = df_base.fillna(df_base.mean())
    df_base.to_csv(f'{processed_dir}/base_after_fill.csv', index=False)
    
    # 对类别数据进行one-hot编码
    df_base = pd.get_dummies(df_base, columns=setting['category_names'])
        
    df_base.to_csv(f'{processed_dir}/dataset_im.csv', index=False)