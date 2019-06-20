# -*- coding:utf-8 -*-
"""https://quinonero.net/Publications/predicting-clicks-facebook.pdf"""

import xgboost as xgb
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder


class TrainModel(object):
    def __init__(self):
        pass

    def preprocess_data(self, df, validation_frac=0.2):
        FEATURE_COLUMNS = ['C1',
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
        CATEGORICAL_COLUMNS = FEATURE_COLUMNS
        X = df[FEATURE_COLUMNS]
        y = df['click'].astype(np.int)
        # feature
        X_test_df = X.sample(frac=validation_frac)
        X_train_df_index = X.index.difference(X_test_df.index)
        X_train_df = X.loc[X_train_df_index]
        X_test_df.reset_index(drop=True, inplace=True)
        X_train_df.reset_index(drop=True, inplace=True)
        # label
        y_test_df = y.sample(frac=validation_frac)
        y_train_df_index = y.index.difference(y_test_df.index)
        y_train_df = y.loc[y_train_df_index]
        y_test_df.reset_index(drop=True, inplace=True)
        y_train_df.reset_index(drop=True, inplace=True)

        # label encoder
        oe = OrdinalEncoder()
        X_train_df = oe.fit_transform(X_train_df)
        X_test_df = oe.fit_transform(X_test_df)

        return X_train_df, X_test_df, y_train_df, y_test_df, CATEGORICAL_COLUMNS

    def gbdt_part(self, df_train):
        X_train_df, X_test_df, y_train_df, y_test_df, categorical_columns = self.preprocess_data(df_train)

        lgb_train = lgb.Dataset(X_train_df, y_train_df, categorical_feature=list(range(21)))
        lgb_eval = lgb.Dataset(X_test_df, y_test_df, reference=lgb_train, categorical_feature=list(range(21)))

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss'},
            'num_leaves': 90,
            'num_trees': 400,
            'learning_rate': 0.03,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=200,
                        valid_sets=lgb_train)

    def lr_part(self):
        pass

    def _train(self):
        pass


def main():
    train_df = pd.read_csv('../data/train.csv', nrows=10000000, dtype=np.str)
    m = TrainModel()
    m.gbdt_part(train_df)


if __name__ == '__main__':
    main()
# 0.458023
# 0.441167
# 0.441372