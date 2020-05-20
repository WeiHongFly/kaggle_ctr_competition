# -*- coding:utf-8 -*-  
""" 
@file: criteo_ctr.py 
"""

import tensorflow as tf
from tensorflow import feature_column
import os
import pickle
from tensorflow.keras import layers
import math
import gc
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

train_columns = ["label"] + ["i_%d" % i for i in range(1, 14)] + ["c_%d" % i for i in range(1, 27)]
test_columns = ["i_%d" % i for i in range(1, 14)] + ["c_%d" % i for i in range(1, 27)]
categories_path = r"..\data\pickle_dir\categories.pkl"
dic_value_path = r"..\data\pickle_dir\dic_value.pkl"
path = r"E:\download\dac.tar\dac"
train_data_rows = 45840617


def split_train_data(validation_rate=0.2):
    # split all labeled data to two parts:20% for evaluation 80% for train.
    validation_rows = int(train_data_rows * validation_rate)
    train_rows = train_data_rows - validation_rows
    skip_rows = 0
    for index, nrows in enumerate([10000000, 10000000, 10000000, 6672494]):
        print("%d batch is starting" % (index + 1))
        df_train = pd.read_csv(os.path.join(path, "train.txt"), names=train_columns, sep="\t", nrows=nrows, engine="c", skiprows=skip_rows)
        df_train.to_csv(os.path.join(path, "train_data.csv"), index=False, mode="a", header=True if index == 0 else False)
        del df_train
        gc.collect()
        skip_rows += nrows
        print("batch %d write finished" % (index + 1))

    df_validation = pd.read_csv(os.path.join(path, "train.txt"), names=train_columns, sep="\t", skiprows=train_rows, engine="c")
    df_validation.to_csv(os.path.join(path, "validation_data.csv"), index=False, mode="w")
    del df_validation
    gc.collect()
    print("validation data write finished")


def plot_train(history):
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # training_accuracy = history.history['accuracy']
    # test_accuracy = history.history['val_accuracy']
    #
    # training_auc = history.history['AUC']
    # test_auc = history.history['AUC']

    epoch_count = range(1, len(training_loss) + 1)

    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


@tf.function
def log_normalize(col):
    col_1 = tf.math.log1p(col)
    res = tf.where(tf.math.is_nan(col_1), tf.ones_like(col_1), col_1)
    res = tf.clip_by_value(res, clip_value_min=-1e5, clip_value_max=1e5)
    return res


def load_model(model_path='../models/wide_deep_model'):
    print(os.path.abspath(model_path))
    model = tf.keras.models.load_model(model_path)
    model.summary()


class CriteoCtrPrediction(object):
    """train data rows=45840617"""

    def __init__(self):
        self.test_columns = ["i_%d" % i for i in range(1, 14)] + ["c_%d" % i for i in range(1, 27)]
        self.train_columns = ["label"] + self.test_columns
        self.data_dir = path
        self.categories, self.default_value_ordered = self.load_pickle
        self.data_train_batch = self.read_data(os.path.join(path, "train_data.csv"), label="label")
        self.data_validation_batch = self.read_data(os.path.join(path, "validation_data.csv"), label="label")
        self.data_test_batch = self.read_data(os.path.join(path, "test.txt"), label=None, field_delim="\t", header=False, column_names=test_columns)

    @property
    def load_pickle(self):
        with open(categories_path, "rb") as f:
            categories = pickle.load(f)
            for col in categories:
                for index, v in enumerate(categories[col]):
                    if isinstance(v, float):
                        if math.isnan(v):
                            del categories[col][index]
        with open(dic_value_path, "rb") as f:
            dic_value = pickle.load(f)
        default_value_ordered = [0] + [dic_value.get(c) for c in test_columns]
        return categories, default_value_ordered

    def read_data(self, data_path, label, field_delim=",", header=True, column_names=None):
        data_batch = tf.data.experimental.make_csv_dataset(file_pattern=data_path,
                                                           batch_size=4096,
                                                           num_epochs=1,
                                                           header=header,
                                                           column_names=column_names,
                                                           label_name=label,
                                                           column_defaults=self.default_value_ordered,
                                                           field_delim=field_delim)

        data_batch = data_batch
        return data_batch

    def wide_deep_model(self):
        wide_feature_columns = []
        deep_feature_columns = []
        feature_layer_inputs = {}

        for col in test_columns:
            if col.startswith("c"):
                if len(self.categories[col]) <= 1000:
                    c = feature_column.categorical_column_with_vocabulary_list(key=col, vocabulary_list=self.categories[col])
                else:
                    c = feature_column.categorical_column_with_hash_bucket(key=col, hash_bucket_size=5000)
                c_1_embedding = feature_column.embedding_column(categorical_column=c, dimension=32)
                deep_feature_columns.append(c_1_embedding)
                feature_layer_inputs[col] = tf.keras.Input(shape=(1,), name=col, dtype=tf.string)

            elif col.startswith("i"):
                i_1 = feature_column.numeric_column(col, normalizer_fn=log_normalize)
                wide_feature_columns.append(i_1)
                feature_layer_inputs[col] = tf.keras.Input(shape=(1,), name=col, dtype=tf.float32)

        # cross feature
        c_9_cate = feature_column.categorical_column_with_vocabulary_list(key="c_9", vocabulary_list=self.categories["c_9"])
        c_20_cate = feature_column.categorical_column_with_vocabulary_list(key="c_20", vocabulary_list=self.categories["c_20"])
        c_9_c_20_crossed = feature_column.crossed_column(keys=[c_9_cate, c_20_cate], hash_bucket_size=2000)
        crossed_feature = feature_column.indicator_column(c_9_c_20_crossed)
        wide_feature_columns.append(crossed_feature)

        wide_feature_layer = layers.DenseFeatures(wide_feature_columns)
        wide_part = wide_feature_layer(feature_layer_inputs)

        deep_feature_layer = layers.DenseFeatures(deep_feature_columns)
        deep_part = deep_feature_layer(feature_layer_inputs)

        layer = layers.Dense(1024, activation="relu")(deep_part)
        layer = layers.Dense(512, activation="relu")(layer)
        layer = layers.Dense(256, activation="relu")(layer)

        layer = layers.concatenate([layer, wide_part])

        output = layers.Dense(1, activation="sigmoid")(layer)

        model = tf.keras.Model(inputs=[v for v in feature_layer_inputs.values()], outputs=output,
                               name="criteo_wide_deep_model")

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy', "AUC"])
        return model

    def main(self):
        model = self.wide_deep_model()
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3, min_delta=-1e-3)
        tensor_board = tf.keras.callbacks.TensorBoard(log_dir='../logs', profile_batch=0)
        history = model.fit(self.data_train_batch, validation_data=self.data_validation_batch, epochs=1, verbose=2, workers=4, callbacks=[early_stop, tensor_board])
        model.save('../models/wide_deep_model')
        model = tf.keras.models.load_model('../models/wide_deep_model')
        pprint(history.history)
        plot_train(history)
        print(model.summary())


if __name__ == '__main__':
    ctr = CriteoCtrPrediction()
    ctr.main()
