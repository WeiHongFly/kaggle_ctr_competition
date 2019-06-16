"""
加入数据的采样，通过采样，训练多个模型，进而进行多模型融合
分层采样
随机采样
bagging
"""

import pandas as pd


def sample_df(df, random_state=None, frac=0.1):
    "随机采样"
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Sampling object must be pandas.DataFrame")
    test_df = df.sample(frac=frac, random_state=random_state)
    train_df_index = df.index.difference(test_df.index)
    train_df = df.loc[train_df_index]
    test_df.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    return train_df, test_df
