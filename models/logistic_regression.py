# -*- coding:utf-8 -*-
""" 
@file: logistic_regression.py 
@time: 2019/05/30
"""

import pandas as pd
import numpy as np
from csv import DictReader
from utils.feature_process import fit_mapper, type_mapper
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import SGDClassifier

columns = ['id',
           'click',
           'hour',
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
           'C21']

batch_size = 200
epoch = 10


def extract_time(df):
    df = df.rename(index=str, columns={"hour": "time_info"})
    df = df.astype(dtype={"time_info": np.str})

    df.loc[:, "year"] = df.time_info.str[:2]
    df.loc[:, "month"] = df.time_info.str[2:4]
    df.loc[:, "day"] = df.time_info.str[4:6]
    df.loc[:, "hour"] = df.time_info.str[6:8]

    df = df.drop(columns=["time_info"])
    return df


class TrainModel(object):
    def __init__(self):
        self.mapper = fit_mapper()

        assert isinstance(self.mapper, dict)
        self.mapper_estimator = DataFrameMapper(list(zip(self.mapper.keys(), self.mapper.values())))
        self.df_obj = pd.read_csv("../data/train.csv", iterator=True, dtype=type_mapper)
        self.model = SGDClassifier(n_jobs=-1, verbose=3, loss="log")

    def read_batch_data(self):
        for i in range(epoch):
            print("epoch:{}".format(i + 1))
            batch_df = self.df_obj.get_chunk(batch_size)
            label = batch_df["click"]
            id = batch_df["id"]
            batch_df = batch_df.drop(columns=["id", "click"])
            batch_df = extract_time(batch_df)
            print(batch_df.shape)
            print(batch_df.columns)
            clean_df = self.mapper_estimator.transform(batch_df)
            self.model.fit(clean_df, label)

    def train(self):
        self.read_batch_data()


if __name__ == '__main__':
    m = TrainModel()
    m.train()
