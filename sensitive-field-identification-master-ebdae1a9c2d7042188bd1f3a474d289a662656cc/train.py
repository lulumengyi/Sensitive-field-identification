#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Train

Usage:
    train.py [-d | --demo_dataset] [-t | --sensitive_type]

Options:
    -h --help                         显示帮助
    -v --version                      显示版本
    -d --demo_dataset                 是否使用示例数据
    -t --sensitive_type               预测标签是否为敏感类型

"""

import pickle
import tensorflow as tf

from docopt import docopt

from sensitive_field_identification.utils.constant import *
from sensitive_field_identification.utils.data_utils import build_char_dict
from sensitive_field_identification.models.wide_deep_char_cnn import WideDeepCharCNN

if __name__ == '__main__':
    args = docopt(__doc__, version='Train 1.0.0')

    if args['--demo_dataset']:
        demo_dataset = '_demo'
    else:
        demo_dataset = ''

    dict_pickle_file_path = 'data/char_dict.pickle'
    text_dict = build_char_dict(char_dict_pickle_file_path=dict_pickle_file_path)

    wide_deep_char_cnn = WideDeepCharCNN(
        wide_features_len=TRANSFORMED_WIDE_FEATURES_LEN,
        deep_features_len=TRANSFORMED_DEEP_FEATURES_LEN,
        vocabulary_size=len(text_dict),
        sensitive_type_label=args['--sensitive_type'],
        num_class=COLUMN_SENSITIVE_TYPE_VALUES_LEN)

    LOGGER.info('Loading train dataset ...')
    wide_features_train_pickle_file_path = \
        'data/wide_features{demo_dataset}_train.pickle'.format(demo_dataset=demo_dataset)
    deep_features_train_pickle_file_path = \
        'data/deep_features{demo_dataset}_train.pickle'.format(demo_dataset=demo_dataset)
    is_sensitive_labels_train_pickle_file_path = \
        'data/is_sensitive_labels{demo_dataset}_train.pickle'.format(demo_dataset=demo_dataset)
    sensitive_type_labels_train_pickle_file_path = \
        'data/sensitive_type_labels{demo_dataset}_train.pickle'.format(demo_dataset=demo_dataset)

    with open(wide_features_train_pickle_file_path, 'rb') as f:
        train_wide_features = pickle.load(f)
    with open(deep_features_train_pickle_file_path, 'rb') as f:
        train_deep_features = pickle.load(f)
    with open(is_sensitive_labels_train_pickle_file_path, 'rb') as f:
        train_is_sensitive_labels = pickle.load(f)
    with open(sensitive_type_labels_train_pickle_file_path, 'rb') as f:
        train_sensitive_type_labels = pickle.load(f)

    LOGGER.info('Loading valid dataset ...')
    wide_features_valid_pickle_file_path = \
        'data/wide_features{demo_dataset}_valid.pickle'.format(demo_dataset=demo_dataset)
    deep_features_valid_pickle_file_path = \
        'data/deep_features{demo_dataset}_valid.pickle'.format(demo_dataset=demo_dataset)
    is_sensitive_labels_valid_pickle_file_path = \
        'data/is_sensitive_labels{demo_dataset}_valid.pickle'.format(demo_dataset=demo_dataset)
    sensitive_type_labels_valid_pickle_file_path = \
        'data/sensitive_type_labels{demo_dataset}_valid.pickle'.format(demo_dataset=demo_dataset)

    with open(wide_features_valid_pickle_file_path, 'rb') as f:
        valid_wide_features = pickle.load(f)
    with open(deep_features_valid_pickle_file_path, 'rb') as f:
        valid_deep_features = pickle.load(f)
    with open(is_sensitive_labels_valid_pickle_file_path, 'rb') as f:
        valid_is_sensitive_labels = pickle.load(f)
    with open(sensitive_type_labels_valid_pickle_file_path, 'rb') as f:
        valid_sensitive_type_labels = pickle.load(f)

    checkpoint_dir = 'data/wide_deep_char_cnn{demo_dataset}_checkpoint'.format(demo_dataset=demo_dataset)
    if not tf.gfile.Exists(checkpoint_dir):
        tf.gfile.MakeDirs(checkpoint_dir)

    summaries_dir = 'data/wide_deep_char_cnn{demo_dataset}_summaries'.format(demo_dataset=demo_dataset)

    wide_deep_char_cnn.train(
        train_wide_features, train_deep_features,
        train_is_sensitive_labels, train_sensitive_type_labels,
        valid_wide_features=valid_wide_features,
        valid_deep_features=valid_deep_features,
        valid_is_sensitive_labels=valid_is_sensitive_labels,
        valid_sensitive_type_labels=valid_sensitive_type_labels,
        checkpoint_dir=checkpoint_dir,
        summaries_dir=summaries_dir)
