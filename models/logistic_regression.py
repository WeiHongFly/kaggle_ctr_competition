# -*- coding:utf-8 -*-
""" 
@file: logistic_regression.py 
@time: 2019/05/30
"""
import os
import pandas as pd
import numpy as np
from ..utils.feature_process import train_mapper, type_mapper, feature_transform
from sklearn.linear_model import SGDClassifier
import joblib
from datetime import datetime
import time
import math
from sklearn.metrics import log_loss, confusion_matrix, auc

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

batch_size = 100
epoch = math.ceil(40428967 / 100)  # 需要epoch次可将全部数据取完


# epoch = 3


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
        self.df_obj = pd.read_csv("../data/train.csv", iterator=True, dtype=type_mapper)
        self.model = SGDClassifier(n_jobs=-1, verbose=0, loss="log")

        path_persistence = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_persistence")
        os.mkdir(path_persistence) if not os.path.exists(path_persistence) else None
        self.mapper_path = os.path.join(path_persistence, "mapper.pkl")
        self.model_path = os.path.join(path_persistence, "model.pkl")

    def train_mapper(self):
        if not os.path.exists(self.mapper_path):
            self.mapper = train_mapper()
            joblib.dump(self.mapper, self.mapper_path, compress=7)
        else:
            print("mapper has been pickled")
            self.mapper = joblib.load(self.mapper)

    def train_model(self):
        for i in range(epoch):
            s = time.time()
            print("epoch:=============>{}".format(i + 1))
            batch_df = self.df_obj.get_chunk(batch_size)
            label = batch_df["click"]
            id = batch_df["id"]
            batch_df = batch_df.drop(columns=["id", "click"])
            batch_df = extract_time(batch_df)
            clean_df = feature_transform(batch_df, self.mapper)
            classes = [0, 1] if i == 0 else None
            self.model.partial_fit(clean_df, label, classes=classes)
            y_pre = self.model.predict_proba(clean_df)
            loss = log_loss(label, y_pre)
            print("THE STEP LOG_LOSS:{}".format(loss))
            print("USE TIME {}".format(time.time() - s))

    def main_model(self):
        print("START:{}".format(datetime.now()))
        self.train_model()
        joblib.dump(self.model, self.model_path, compress=7)
        print("END:{}".format(datetime.now()))

    def main_mapper(self):
        print("START:{}".format(datetime.now()))
        self.train_mapper()
        print("END:{}".format(datetime.now()))

    def validation_regression(self):
        if os.path.exists(self.model_path) and os.path.exists(self.mapper_path):
            model = joblib.load(self.model_path)
            mapper = joblib.load(self.mapper_path)
            loss_list = []
            for i in range(epoch):
                batch_df = self.df_obj.get_chunk(batch_size)
                label = batch_df["click"]
                id = batch_df["id"]
                batch_df = batch_df.drop(columns=["id", "click"])
                batch_df = extract_time(batch_df)
                clean_df = feature_transform(batch_df, mapper)
                y_pre_proba = model.predict_proba(clean_df)
                y_pre_label = model.predict(clean_df)
                loss = log_loss(label, y_pre_proba)
                loss_list.append((label.shape[0], loss))
            df = pd.DataFrame(loss_list, columns=["size", "loss"])
            df.loc[:, "batch_loss_sum"] = df["size"] * df["loss"]
            loss = (df["batch_loss_sum"].sum()) / (df["size"].sum())
            print("LOSS:{}".format(loss))



        else:
            print("model and mapper not exists")  # 0.4373294089990498


if __name__ == '__main__':
    m = TrainModel()
    m.main_mapper()
    m.main_model()
    m.validation_regression()
