# -*- coding:utf-8 -*-  
""" 
@file: feature_process.py 
@time: 2019/06/05
"""

import pandas as pd
import numpy as np
from pprint import pprint

from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import coo_matrix, hstack, csr_matrix

meta_info = pd.DataFrame([{"col": "id", "unique_count": 40428967},
                          {"col": "click", "unique_count": 2},
                          {"col": "hour", "unique_count": 240},
                          {"col": "C1", "unique_count": 7},
                          {"col": "banner_pos", "unique_count": 7},
                          {"col": "site_id", "unique_count": 4737},
                          {"col": "site_domain", "unique_count": 7745},
                          {"col": "site_category", "unique_count": 26},
                          {"col": "app_id", "unique_count": 8552},
                          {"col": "app_domain", "unique_count": 559},
                          {"col": "app_category", "unique_count": 36},
                          {"col": "device_id", "unique_count": 2686408},
                          {"col": "device_ip", "unique_count": 6729486},
                          {"col": "device_model", "unique_count": 8251},
                          {"col": "device_type", "unique_count": 5},
                          {"col": "device_conn_type", "unique_count": 4},
                          {"col": "C14", "unique_count": 2626},
                          {"col": "C15", "unique_count": 8},
                          {"col": "C16", "unique_count": 9},
                          {"col": "C17", "unique_count": 435},
                          {"col": "C18", "unique_count": 4},
                          {"col": "C19", "unique_count": 68},
                          {"col": "C20", "unique_count": 172},
                          {"col": "C21", "unique_count": 60}])

type_mapper = {'id': 'uint64',
               'click': 'int64',
               'C1': 'object',
               'banner_pos': 'object',
               'site_id': 'object',
               'site_domain': 'object',
               'site_category': 'object',
               'app_id': 'object',
               'app_domain': 'object',
               'app_category': 'object',
               'device_id': 'object',
               'device_ip': 'object',
               'device_model': 'object',
               'device_type': 'object',
               'device_conn_type': 'object',
               'C14': 'object',
               'C15': 'object',
               'C16': 'object',
               'C17': 'object',
               'C18': 'object',
               'C19': 'object',
               'C20': 'object',
               'C21': 'object',
               'year': 'object',
               'month': 'object',
               'day': 'object',
               'hour': 'object'}

SAMPLE_SIZE = 5000000
HASH_THRESHOLD = 100
MAPPER = {}


def read_df(col, sample_size=-1):
    if sample_size == -1:
        df = pd.read_csv("../data/train.csv", usecols=[col], nrows=100, dtype=type_mapper)
    else:
        df = pd.read_csv("../data/train.csv", usecols=[col], nrows=sample_size, dtype=type_mapper)
    return df


def extract_time(df):
    df = df.rename(index=str, columns={"hour": "time_info"})
    df = df.astype(dtype={"time_info": np.str})

    year_unique_count = df.time_info.str[:2].unique().shape[0]
    if year_unique_count > 1:
        df.loc[:, "year"] = df.time_info.str[:2]
        meta_info.append(pd.Series({"col": "year", "unique_count": year_unique_count}), ignore_index=True)

    month_unique_count = df.time_info.str[2:4].unique().shape[0]
    if month_unique_count > 1:
        df.loc[:, "month"] = df.time_info.str[2:4]
        meta_info.append(pd.Series({"col": "month", "unique_count": month_unique_count}), ignore_index=True)

    day_unique_count = df.time_info.str[4:6].unique().shape[0]
    if day_unique_count > 1:
        df.loc[:, "day"] = df.time_info.str[4:6]
        meta_info.append(pd.Series({"col": "day", "unique_count": day_unique_count}), ignore_index=True)

    hour_unique_count = df.time_info.str[6:8].unique().shape[0]
    if hour_unique_count > 1:
        df.loc[:, "hour"] = df.time_info.str[6:8]
        meta_info.append(pd.Series({"col": "hour", "unique_count": hour_unique_count}), ignore_index=True)

    df = df.drop(columns=["time_info"])
    if not df.empty:
        return df
    else:
        return pd.DataFrame()


def column_hash(df, col):
    # TODO：待优化，转成超参数。
    col_unique_count = meta_info.loc[meta_info["col"] == col, "unique_count"].tolist()[0]

    if int((col_unique_count / 2)) > 100:
        n_features = 100
    else:
        n_features = int(col_unique_count / 2)
    f = FeatureHasher(input_type="string", n_features=n_features)
    return f


def columns_binary(df, col):
    l = LabelBinarizer()
    l.fit(df[col])
    return l


def feature_extraction(df):
    if df.empty: return
    cols = df.columns
    for col in cols:
        if meta_info.loc[meta_info["col"] == col, "unique_count"].tolist()[0] > HASH_THRESHOLD:
            print("col:{} will be processed by feature hash".format(col))
            transfer = column_hash(df, col)
            MAPPER.update({col: transfer})

        else:
            print("col:{} will be processed by feature binary".format(col))
            transfer = columns_binary(df, col)
            MAPPER.update({col: transfer})


def fit_mapper():
    # 抽取hour数据，类型转换

    all_columns = ['C1',
                   'banner_pos',
                   'site_id',
                   'site_domain',
                   'site_category',
                   'app_id',
                   'app_domain',
                   'app_category',
                   'device_id',
                   'device_ip',
                   'device_model',
                   'device_type',
                   'device_conn_type',
                   'C14',
                   'C15',
                   'C16',
                   'C17',
                   'C18',
                   'C19',
                   'C20',
                   'C21',
                   'hour']
    for col in all_columns:
        df = read_df(col, SAMPLE_SIZE)
        if col == 'hour':
            df = extract_time(df)
        feature_extraction(df)
    return MAPPER


def feature_transform(df, mapper):
    blocks_data = []
    for col in df.columns:
        print("col is:{}".format(col))
        the_mapper = mapper.get(col)
        if the_mapper:
            clean_col_data = the_mapper.transform(df[col])
            blocks_data.append(clean_col_data)
        else:
            print('error!!!')
    clean_data = hstack(blocks_data)
    return clean_data
