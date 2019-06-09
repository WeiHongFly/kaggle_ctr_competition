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
    df.loc[:, "year"] = df.time_info.str[:2]
    df.loc[:, "month"] = df.time_info.str[2:4]
    df.loc[:, "day"] = df.time_info.str[4:6]
    df.loc[:, "hour"] = df.time_info.str[6:8]
    df.drop(columns=["time_info"])
    return df


def column_hash(df):
    pass


def columns_binary(df):
    pass


def main():
    # 抽取hour数据，类型转换
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
                   'hour': 'object'}

    all_columns = ['id',
                   'click',
                   'C1',
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


if __name__ == '__main__':
    main()
