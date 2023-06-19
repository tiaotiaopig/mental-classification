import os
import pandas as pd
import multiprocessing as mp

def txt_to_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep='\t', low_memory=False, encoding='utf-8')

def merge_outer(df_left: pd.DataFrame, df_right: pd.DataFrame) -> pd.DataFrame:
    on = ['subjectkey', 'eventname']
    df_merge = pd.merge(df_left, df_right, how='outer', left_on=on, right_on=on)
    return df_merge

def merge_df_list(df_list: list) -> list:
    list_len = len(df_list)
    if list_len > 1:
        merge_index = [(df_list[index], df_list[index + 1]) for index in range(0, list_len - 1, 2)]
        with mp.Pool(mp.cpu_count() // 2) as pool:
            async_merge = pool.starmap_async(merge_outer, merge_index)
            res = async_merge.get()
        merge_list = [df for df in res]
        if list_len % 2 == 1:
            merge_list.append(df_list[-1])
        df_list = merge_list
    return df_list
        
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(base_dir, 'data/raw')
    file_path_list = [os.path.join(raw_data_dir, file_name) for file_name in filter(lambda x: x.startswith('abcd_'), os.listdir(raw_data_dir))]
    # 多进程读取csv
    with mp.Pool(mp.cpu_count() // 2) as pool:
        async_csv = pool.map_async(txt_to_csv, file_path_list[0:50])
        async_csv.get()
    
    # 多进程合并csv
    df_list = async_csv.get()
    while len(df_list) > 1:
        list_len = len(df_list)
        merge_index = [(df_list[index], df_list[index + 1]) for index in range(0, list_len - 1, 2)]
        with mp.Pool(mp.cpu_count() // 2) as pool:
            async_merge = pool.starmap_async(merge_outer, merge_index)
            res = async_merge.get()
        merge_list = [df for df in res]
        if list_len % 2 == 1:
            merge_list.append(df_list[-1])
        df_list = merge_list
        
    df_list[0].to_csv('data/merge.csv', index=False)
    # df_base = pd.DataFrame({'subjectkey':[], 'eventname':[]})
    # for file_path in file_path_list:
    #     print(df_base.shape)
    #     df_other = pd.read_csv(file_path, sep='\t', low_memory=False, encoding='utf8')
    #     df_base = merge_outer(df_base, df_other, on=['subjectkey', 'eventname'])