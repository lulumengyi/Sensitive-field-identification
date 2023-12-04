#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Dataset

Usage:
    dataset.py [-s | --split_dataset] [-o | --over_sampling]
               [-c | --cache_dataset] [-l | --include_valid_dataset] [-d | --demo_dataset]
               [-n=<N> | --n_jobs=<N>] [-k=<K> | --chunksize=<K>]
    dataset.py (-h | --help)
    dataset.py --version

Options:
    -h --help                         显示帮助
    -v --version                      显示版本
    -s --split_dataset                分割数据集
    -o --over_sampling                是否过采样
    -c --cache_dataset                缓存数据集
    -l --include_valid_dataset        包含验证集
    -d --demo_dataset                 是否使用示例数据
    -n=<N> --n_jobs=<N>               并行数
    -k=<K> --chunksize=<K>            进程处理的块大小

"""

import pandas as pd

from docopt import docopt
from sklearn.model_selection import train_test_split

from sensitive_field_identification.utils.constant import *
from sensitive_field_identification.utils.data_utils import build_char_dict, build_dataset


if __name__ == '__main__':
    args = docopt(__doc__, version='Dataset 1.0.0')

    file_dir = os.path.dirname(__file__)

    if args['--split_dataset']:
        # Split Train and Test Dataset
        LOGGER.info('Spliting train and test dataset ...')
        column_meta_file_path = 'data/column_meta.tsv'
        column_meta_train_file_path = 'data/column_meta_train.tsv'
        column_meta_test_file_path = 'data/column_meta_test.tsv'

        with open(column_meta_file_path, 'r') as f:
            column_meta = pd.read_csv(column_meta_file_path, sep='\t', quotechar='"', dtype=str, error_bad_lines=False)

        column_meta = column_meta.sample(frac=1.0)
        column_meta_train, column_meta_test = train_test_split(column_meta, test_size=0.3)

        LOGGER.info('Saving train dataset to [{column_meta_train_file_path}] ...'
                    .format(column_meta_train_file_path=column_meta_train_file_path))
        column_meta_train.to_csv(column_meta_train_file_path, sep='\t', index=False)
        LOGGER.info('Saving test dataset to [{column_meta_test_file_path}] ...'
                    .format(column_meta_test_file_path=column_meta_test_file_path))
        column_meta_test.to_csv(column_meta_test_file_path, sep='\t', index=False)

    if args['--cache_dataset']:
        n_jobs = int(args['--n_jobs']) if args['--n_jobs'] else os.cpu_count()
        chunksize = int(args['--chunksize']) if args['--chunksize'] else DEFAULT_CHUNK_SIZE
        over_sampling = True if args['--over_sampling'] else False

        if args['--demo_dataset']:
            demo_dataset = '_demo'
        else:
            demo_dataset = ''

        # Building CHAR dictionary
        LOGGER.info('Building CHAR dictionary ...')
        dict_pickle_file_path = 'data/char_dict.pickle'
        text_dict = build_char_dict(char_dict_pickle_file_path=dict_pickle_file_path)

        # Cache Train Dataset
        LOGGER.info('Caching train dataset ...')
        column_meta_file_path = 'data/column_meta{demo_dataset}_train.tsv'.format(demo_dataset=demo_dataset)
        column_value_file_path = 'data/column_value.tsv'
        wide_features_pickle_file_path = \
            'data/wide_features{demo_dataset}_train.pickle'.format(demo_dataset=demo_dataset)
        deep_features_pickle_file_path = \
            'data/deep_features{demo_dataset}_train.pickle'.format(demo_dataset=demo_dataset)
        is_sensitive_labels_pickle_file_path = \
            'data/is_sensitive_labels{demo_dataset}_train.pickle'.format(demo_dataset=demo_dataset)
        sensitive_type_labels_pickle_file_path = \
            'data/sensitive_type_labels{demo_dataset}_train.pickle'.format(demo_dataset=demo_dataset)
        _, _, _, _ = build_dataset(
            column_meta_file_path, column_value_file_path, text_dict,
            n_jobs=n_jobs, chunksize=chunksize, over_sampling=over_sampling,
            wide_features_pickle_file_path=wide_features_pickle_file_path,
            deep_features_pickle_file_path=deep_features_pickle_file_path,
            is_sensitive_labels_pickle_file_path=is_sensitive_labels_pickle_file_path,
            sensitive_type_labels_pickle_file_path=sensitive_type_labels_pickle_file_path)

        if args['--include_valid_dataset']:
            # Cache Valid Dataset
            LOGGER.info('Caching valid dataset ...')
            column_meta_file_path = 'data/column_meta{demo_dataset}_valid.tsv'.format(demo_dataset=demo_dataset)
            column_value_file_path = 'data/column_value.tsv'
            wide_features_pickle_file_path = \
                'data/wide_features{demo_dataset}_valid.pickle'.format(demo_dataset=demo_dataset)
            deep_features_pickle_file_path = \
                'data/deep_features{demo_dataset}_valid.pickle'.format(demo_dataset=demo_dataset)
            is_sensitive_labels_pickle_file_path = \
                'data/is_sensitive_labels{demo_dataset}_valid.pickle'.format(demo_dataset=demo_dataset)
            sensitive_type_labels_pickle_file_path = \
                'data/sensitive_type_labels{demo_dataset}_valid.pickle'.format(demo_dataset=demo_dataset)
            _, _, _, _ = build_dataset(
                column_meta_file_path, column_value_file_path, text_dict,
                n_jobs=n_jobs, chunksize=chunksize, over_sampling=over_sampling,
                wide_features_pickle_file_path=wide_features_pickle_file_path,
                deep_features_pickle_file_path=deep_features_pickle_file_path,
                is_sensitive_labels_pickle_file_path=is_sensitive_labels_pickle_file_path,
                sensitive_type_labels_pickle_file_path=sensitive_type_labels_pickle_file_path)

        # Cache Test Dataset
        LOGGER.info('Caching test dataset ...')
        column_meta_file_path = 'data/column_meta{demo_dataset}_test.tsv'.format(demo_dataset=demo_dataset)
        column_value_file_path = 'data/column_value.tsv'
        wide_features_pickle_file_path = \
            'data/wide_features{demo_dataset}_test.pickle'.format(demo_dataset=demo_dataset)
        deep_features_pickle_file_path = \
            'data/deep_features{demo_dataset}_test.pickle'.format(demo_dataset=demo_dataset)
        is_sensitive_labels_pickle_file_path = \
            'data/is_sensitive_labels{demo_dataset}_test.pickle'.format(demo_dataset=demo_dataset)
        sensitive_type_labels_pickle_file_path = \
            'data/sensitive_type_labels{demo_dataset}_test.pickle'.format(demo_dataset=demo_dataset)
        _, _, _, _ = build_dataset(
            column_meta_file_path, column_value_file_path, text_dict,
            n_jobs=n_jobs, chunksize=chunksize, over_sampling=over_sampling,
            wide_features_pickle_file_path=wide_features_pickle_file_path,
            deep_features_pickle_file_path=deep_features_pickle_file_path,
            is_sensitive_labels_pickle_file_path=is_sensitive_labels_pickle_file_path,
            sensitive_type_labels_pickle_file_path=sensitive_type_labels_pickle_file_path)
