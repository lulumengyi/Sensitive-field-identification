#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Test

Usage:
    test.py [-d | --demo_dataset] [-t | --sensitive_type]

Options:
    -h --help                         显示帮助
    -v --version                      显示版本
    -d --demo_dataset                 是否使用示例数据
    -t --sensitive_type               预测标签是否为敏感类型


"""

import pickle

from docopt import docopt

from sensitive_field_identification.utils.constant import *
from sensitive_field_identification.utils.data_utils import build_char_dict
from sensitive_field_identification.models.wide_deep_char_cnn import WideDeepCharCNN

if __name__ == '__main__':
    args = docopt(__doc__, version='Test 1.0.0')

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

    LOGGER.info('Loading test dataset ...')
    wide_features_test_pickle_file_path = \
        'data/wide_features{demo_dataset}_test.pickle'.format(demo_dataset=demo_dataset)
    deep_features_test_pickle_file_path = \
        'data/deep_features{demo_dataset}_test.pickle'.format(demo_dataset=demo_dataset)
    is_sensitive_labels_test_pickle_file_path = \
        'data/is_sensitive_labels{demo_dataset}_test.pickle'.format(demo_dataset=demo_dataset)
    sensitive_type_labels_test_pickle_file_path = \
        'data/sensitive_type_labels{demo_dataset}_test.pickle'.format(demo_dataset=demo_dataset)

    with open(wide_features_test_pickle_file_path, 'rb') as f:
        test_wide_features = pickle.load(f)
    with open(deep_features_test_pickle_file_path, 'rb') as f:
        test_deep_features = pickle.load(f)
    with open(is_sensitive_labels_test_pickle_file_path, 'rb') as f:
        test_is_sensitive_labels = pickle.load(f)
    with open(sensitive_type_labels_test_pickle_file_path, 'rb') as f:
        test_sensitive_type_labels = pickle.load(f)

    checkpoint_dir = 'data/wide_deep_char_cnn{demo_dataset}_checkpoint'.format(demo_dataset=demo_dataset)

    wide_deep_char_cnn.test(
        test_wide_features, test_deep_features,
        test_is_sensitive_labels, test_sensitive_type_labels,
        checkpoint_dir=checkpoint_dir)
