# -*- coding:utf-8 -*-  
""" 
@file: feature_process.py 
@time: 2019/06/05
"""

from csv import DictReader
import pandas as pd
import numpy as np
from sklearn.feature_extraction.hashing import FeatureHasher

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, accuracy_score
from csv import DictReader
from sklearn.feature_extraction import DictVectorizer, FeatureHasher, dict_vectorizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn_pandas import DataFrameMapper, cross_val_score

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

HASH_THRESHOLD = 100


def read_df(col):
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
    df = pd.read_csv("../data/train.csv", usecols=[col], nrows=100, dtype=type_mapper)  # FIXME：正式运行时跑全量的数据
    return df


def extract_time(df):
    df = df.rename(index=str, columns={"hour": "time_info"})
    df = df.astype(dtype={"time_info": str})

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

    df.drop(columns=["time_info"])
    if not df.empty:
        return df
    else:
        return None


def column_hash(df):
    pass


def columns_binary(df):
    pass


def feature_extraction(df):
    cols = df.columns
    for col in cols:
        if meta_info[meta_info[col] <= HASH_THRESHOLD].at[0, "unique_count"]:
            print("col:{} will be processed by feature hash")
        else:
            print("col:{} will be processed by feature binary")


def main():
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
        df = read_df(col)
        if col == 'hour':
            df = extract_time(df)
        feature_extraction(df)


if __name__ == '__main__':
    main()
