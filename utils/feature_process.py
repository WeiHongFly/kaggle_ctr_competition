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


def extract_hour_col(row):
    # 从hour中提取时间信息
    pass    


def col_hashing():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
