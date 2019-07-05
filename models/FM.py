import subprocess
import pandas as pd
import numpy as np
import os
import sys
import joblib
from ..utils.feature_process import feature_extraction, feature_transform, type_mapper
from .logistic_regression import extract_time

fm_train_data = "../data/fm_train.txt"


def preprocess(epoch, batch_size):
    # 通过df的方式将流式数据不断的传入该文件，然后处理得到libFM需要的格式
    df_obj = pd.read_csv("../data/train.csv", iterator=True, dtype=type_mapper)
    for i in range(epoch):
        batch_df = df_obj.get_chunk(siz=batch_size)
        label = batch_df["click"]
        id = batch_df["id"]
        batch_df = batch_df.drop(columns=["id", "click"])
        batch_df = extract_time(batch_df)
        mapper = joblib.load("mapper_path")
        clean_df = feature_transform(batch_df, mapper)

        for row in clean_df.iterrows():
            pass


def main():
    # 使用subprocess调用libfm进行训练，本脚本用来处理数据
    pass


if __name__ == '__main__':
    main()
