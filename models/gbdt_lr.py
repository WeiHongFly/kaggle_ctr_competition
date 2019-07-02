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
        self.num_leaf = 30
        self.num_tree = 200
        train_df = pd.read_csv('../data/train.csv', nrows=100000, dtype=np.str)
        self.X_train_df, self.X_test_df, self.y_train_df, self.y_test_df, self.categorical_columns = self.preprocess_data(train_df)

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

    def gbdt_part(self):
        lgb_train = lgb.Dataset(self.X_train_df, self.y_train_df, categorical_feature=list(range(21)))
        lgb_eval = lgb.Dataset(self.X_test_df, self.y_test_df, reference=lgb_train, categorical_feature=list(range(21)))

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss'},
            'num_leaves': self.num_leaf,
            'num_trees': self.num_tree,
            'learning_rate': 0.03,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 3
        }

        gbm = lgb.train(params, lgb_train, num_boost_round=self.num_tree, valid_sets=lgb_train)
        gbm.save_model("../model_persistence/gbdt.txt")
        self.gbm = gbm

    def lr_part(self):
        y_pred_train = self.gbm.predict(self.X_train_df, pred_leaf=True)
        transformed_train_mat = np.zeros(shape=(y_pred_train.shape[0], self.num_tree * self.num_leaf))
        for i in range(len(y_pred_train)):
            col_index = np.arange(self.num_tree) * self.num_leaf + y_pred_train[i]
            transformed_train_mat[i][col_index] = 1

        y_pred_test = self.gbm.predict(self.X_test_df, pred_leaf=True)
        transformed_test_mat = np.zeros(shape=(y_pred_test.shape[0], self.num_tree * self.num_leaf))
        for i in range(len(y_pred_test)):
            col_index = np.arange(self.num_tree) * self.num_leaf + y_pred_test[i]
            transformed_test_mat[i][col_index] = 1

        # train logistic regression
        lr = LogisticRegression(n_jobs=-1, verbose=3, C=0.05)
        lr.fit(transformed_train_mat, self.y_train_df)

        # predict by logistic regression
        y_pred_test = lr.predict_proba(transformed_test_mat)

        # Normalized Entropy
        NE = (-1) / len(y_pred_test) * sum(((1 + self.y_test_df) / 2 * np.log(y_pred_test[:, 1]) + (1 - self.y_test_df) / 2 * np.log(1 - y_pred_test[:, 1])))
        print(NE)


def main():
    m = TrainModel()
    m.gbdt_part()
    m.lr_part()


if __name__ == '__main__':
    main()
