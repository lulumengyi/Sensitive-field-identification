#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import csv
import unicodedata
import gc
import pandas as pd
import numpy as np
import logging
import functools
import math

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor

from .constant import *
from ..pre_processing.nl_handler import NLHandler


def zh_punc_to_en_punc(text: str) -> str:
    """ 中文标点转英文标点

    Args:
        text: 带有中文标点的文本

    Returns:
        转换后的文本

    """
    text_ = text

    for zh_p, en_p in zip(punctuation_zh, punctuation_zh_en):
        text_ = text_.replace(zh_p, en_p)

    return text_


def clean_text(text: str) -> str:
    """ 清洗文本

    Args:
        text: 待清洗文本

    Returns:
        清洗后的文本

    """

    text_ = unicodedata.normalize('NFKC', text)
    text_ = keep_pattern.sub(' ', text_)
    text_ = multi_space_pattern.sub(' ', text_)
    text_ = text_.strip().upper()

    return text_


def build_char_dict(
        char_dict_pickle_file_path: str=None,
        char_dict_text_file_path: str=None,
        column_meta_file_path: str=None,
        column_meta_dict_columns: list=None,
        column_value_file_path: str=None,
        column_value_dict_columns: list=None) -> dict:
    """ 构建字符级别词典

    Args:
        char_dict_pickle_file_path: 词典文件位置
        column_meta_file_path: 列 Meta 信息文件位置
        column_value_file_path:  列 Value 信息文件位置

    Returns:

    """

    if not os.path.exists(char_dict_pickle_file_path):
        LOGGER.info('Building CHAR dictionary ...')

        char_counter = {}

        LOGGER.info('Dealing with column meta file [{column_meta_file_path}] ...'
                    .format(column_meta_file_path=column_meta_file_path))
        with open(column_meta_file_path, 'r') as f:
            f_ = csv.DictReader(f, delimiter='\t')

            for row in tqdm(f_, desc='CHAR_META_FILE', unit=' ROWS'):
                for column_meta_dict_column in column_meta_dict_columns:
                    cleaned_column = clean_text(row[column_meta_dict_column])

                    for char in cleaned_column:
                        if char in char_counter:
                            char_counter[char] = char_counter[char] + 1
                        else:
                            char_counter[char] = 1

        LOGGER.info('Dealing with column value file [{column_value_file_path}] ...'
                    .format(column_value_file_path=column_value_file_path))
        with open(column_value_file_path, 'r') as f:
            f_ = csv.DictReader(f, delimiter='\t')

            for row in tqdm(f_, desc='CHAR_VALUE_FILE', unit=' ROWS'):
                for column_value_dict_column in column_value_dict_columns:
                    cleaned_column = clean_text(row[column_value_dict_column])

                    for char in cleaned_column:
                        if char in char_counter:
                            char_counter[char] = char_counter[char] + 1
                        else:
                            char_counter[char] = 1

        char_dict = {}
        char_dict[PAD] = 0
        char_dict[UNK] = 1
        char_dict[EOS] = 2
        char_dict_init_len = len(char_dict)

        sorted_char_counter = sorted(char_counter.items(), key=lambda item:item[1], reverse=True)
        for index, item in enumerate(sorted_char_counter):
            char = item[0]
            char_dict[char] = index + char_dict_init_len

        LOGGER.info('Dumping CHAR dictionary pickle file to [{char_dict_pickle_file_path}] ...'
                    .format(char_dict_pickle_file_path=char_dict_pickle_file_path))
        with open(char_dict_pickle_file_path, 'wb') as f:
            pickle.dump(char_dict, f)

        if char_dict_text_file_path:
            LOGGER.info('Saving CHAR dictionary text file to [{char_dict_text_file_path}] ...'
                        .format(char_dict_text_file_path=char_dict_text_file_path))
            with open(char_dict_text_file_path, 'w') as f:
                f_ = csv.DictWriter(f, fieldnames=['char', 'id'], delimiter='\t')
                f_.writeheader()

                for key, value in char_dict.items():
                    f_.writerow({'char': key, 'id': value})
    else:
        LOGGER.info('Loading char dictionary from pickle file from [{char_dict_pickle_file_path}] ...'
                    .format(char_dict_pickle_file_path=char_dict_pickle_file_path))
        with open(char_dict_pickle_file_path, 'rb') as f:
            char_dict = pickle.load(f)

    return char_dict


def build_word_dict(
        word_dict_pickle_file_path: str=None,
        word_dict_text_file_path: str=None,
        column_meta_file_path: str=None,
        column_meta_dict_columns: list=None,
        column_value_file_path: str=None,
        column_value_dict_columns: list=None) -> dict:
    """ 构建词级别词典

    Args:
        word_dict_pickle_file_path: 词典文件位置
        column_meta_file_path: 列 Meta 信息文件位置
        column_value_file_path: 列 Value 信息文件位置

    Returns:

    """

    if not os.path.exists(word_dict_pickle_file_path):
        LOGGER.info('Building WORD dictionary ...')

        nlp_handler = NLHandler()
        word_counter = {}

        LOGGER.info('Dealing with column meta file [{column_meta_file_path}] ...'
                    .format(column_meta_file_path=column_meta_file_path))
        with open(column_meta_file_path, 'r') as f:
            f_ = csv.DictReader(f, delimiter='\t')

            for row in tqdm(f_, desc='WORD_META_FILE', unit=' ROWS'):
                for column_meta_dict_column in column_meta_dict_columns:
                    cleaned_column = clean_text(row[column_meta_dict_column])

                    for token in nlp_handler.tokenize(cleaned_column):
                        word = token['word']
                        if word in word_counter:
                            word_counter[word] = word_counter[word] + 1
                        else:
                            word_counter[word] = 1

        LOGGER.info('Dealing with column value file [{column_value_file_path}] ...'
                    .format(column_value_file_path=column_value_file_path))
        with open(column_value_file_path, 'r') as f:
            f_ = csv.DictReader(f, delimiter='\t')

            for row in tqdm(f_, desc='WORD_VALUE_FILE', unit=' ROWS'):
                for column_value_dict_column in column_value_dict_columns:
                    cleaned_column = clean_text(row[column_value_dict_column])

                    for token in nlp_handler.tokenize(cleaned_column):
                        word = token['word']
                        if word in word_counter:
                            word_counter[word] = word_counter[word] + 1
                        else:
                            word_counter[word] = 1

        word_dict = {}
        word_dict[PAD] = 0
        word_dict[UNK] = 1
        word_dict[EOS] = 2
        word_dict_init_len = len(word_dict)

        sorted_word_counter = sorted(word_counter.items(), key=lambda item:item[1], reverse=True)
        for index, item in enumerate(sorted_word_counter):
            word = item[0]
            word_dict[word] = index + word_dict_init_len

        LOGGER.info('Dumping WORD dictionary pickle file to [{word_dict_pickle_file_path}] ...'
                    .format(word_dict_pickle_file_path=word_dict_pickle_file_path))
        with open(word_dict_pickle_file_path, 'wb') as f:
            pickle.dump(word_dict, f)

        if word_dict_text_file_path:
            LOGGER.info('Saving CHAR dictionary text file to [{word_dict_text_file_path}] ...'
                        .format(word_dict_text_file_path=word_dict_text_file_path))
            with open(word_dict_text_file_path, 'w') as f:
                f_ = csv.DictWriter(f, fieldnames=['word', 'id'], delimiter='\t')
                f_.writeheader()

                for key, value in word_dict.items():
                    f_.writerow({'word': key, 'id': value})
    else:
        LOGGER.info('Loading char dictionary from pickle file from [{word_dict_pickle_file_path}] ...'
                    .format(word_dict_pickle_file_path=word_dict_pickle_file_path))
        with open(word_dict_pickle_file_path, 'rb') as f:
            word_dict = pickle.load(f)

    return word_dict


def gen_text_seq(text: str, max_length: int, text_dict: dict) -> list:
    """ 构建文本序列

    Args:
        text: 原始文本
        max_length: 序列最大长度
        text_dict: 文本词典

    Returns:
        文本序列

    """

    cleaned_text = clean_text(text)

    text_seq = list(map(lambda c: text_dict.get(c, text_dict[UNK]), cleaned_text))
    text_seq.append(text_dict[EOS])
    text_seq = text_seq[:max_length]
    text_seq.extend([text_dict[PAD]] * (max_length - len(text_seq)))

    return text_seq


def transform_wide_features(features_dict: dict) -> dict:
    """ Wide 特征变换

    Args:
        features_dict: 原始特征词典

    Returns:
        变换后特征词典

    """

    features_transformed_dict = {}

    ## Wide Features
    wide_features = []

    # Wide Feature: Database Port
    features_transformed_dict['database_port'] = DATABASE_PORT_VALUE_DEFAULT
    if 'database_port' in features_dict:
        database_port = features_dict['database_port'].upper().strip()
        if database_port in DATABASE_PORT_VALID_VALUES:
            features_transformed_dict['database_port'] = database_port

    features_transformed_dict['database_port_dummy'] = DATABASE_PORT_OHE.transform(DATABASE_PORT_LE.transform(
        [features_transformed_dict['database_port']]).reshape(1, -1)).toarray()[0]

    wide_features.extend(features_transformed_dict['database_port_dummy'])

    # Wide Feature: Database Name Length
    features_transformed_dict['database_name_length'] = 0.0
    if 'database_name' in features_dict:
        database_name_length = len(features_dict['database_name'].strip()) / DATABASE_NAME_MAX_LEN
        features_transformed_dict['database_name_length'] = \
            database_name_length if database_name_length < 1.0 else 1.0

    wide_features.append(features_transformed_dict['database_name_length'])

    # Wide Feature: Database Comment Length
    features_transformed_dict['database_comment_length'] = 0.0
    if 'database_comment' in features_dict:
        database_comment_length = len(features_dict['database_comment'].strip()) / DATABASE_COMMENT_MAX_LEN
        features_transformed_dict['database_comment_length'] = \
            database_comment_length if database_comment_length < 1.0 else 1.0

    wide_features.append(features_transformed_dict['database_comment_length'])

    # Wide Feature: Database Type
    features_transformed_dict['database_type'] = DATABASE_TYPE_VALUE_DEFAULT
    if 'database_type' in features_dict:
        database_type = features_dict['database_type'].upper().strip()
        if database_type in DATABASE_TYPE_VALID_VALUES:
            features_transformed_dict['database_type'] = database_type

    features_transformed_dict['database_type_dummy'] = DATABASE_TYPE_OHE.transform(DATABASE_TYPE_LE.transform(
        [features_transformed_dict['database_type']]).reshape(1, -1)).toarray()[0]

    wide_features.extend(features_transformed_dict['database_type_dummy'])

    # Wide Feature: Database Is Shard
    features_transformed_dict['database_is_shard'] = DATABASE_IS_SHARD_VALUE_DEFAULT
    if 'database_is_shard' in features_dict:
        database_is_shard = features_dict['database_is_shard'].upper().strip()
        if database_is_shard in DATABASE_IS_SHARD_VALID_VALUES:
            features_transformed_dict['database_is_shard'] = database_is_shard

    features_transformed_dict['database_is_shard_dummy'] = DATABASE_IS_SHARD_OHE.transform(
        DATABASE_IS_SHARD_LE.transform([features_transformed_dict['database_is_shard']]).reshape(1, -1)).toarray()[0]

    wide_features.extend(features_transformed_dict['database_is_shard_dummy'])

    # Wide Feature: Table Name Length
    features_transformed_dict['table_name_length'] = 0.0
    if 'table_name' in features_dict:
        table_name_length = len(features_dict['table_name'].strip()) / TABLE_NAME_MAX_LEN
        features_transformed_dict['table_name_length'] = \
            table_name_length if table_name_length < 1.0 else 1.0

    wide_features.append(features_transformed_dict['table_name_length'])

    # Wide Feature: Table Comment Length
    features_transformed_dict['table_comment_length'] = 0.0
    if 'table_comment' in features_dict:
        table_comment_length = len(features_dict['table_comment'].strip()) / TABLE_COMMENT_MAX_LEN
        features_transformed_dict['table_comment_length'] = \
            table_comment_length if table_comment_length < 1.0 else 1.0

    wide_features.append(features_transformed_dict['table_comment_length'])

    # Wide Feature: Table Theme
    features_transformed_dict['table_theme'] = TABLE_THEME_VALUE_DEFAULT
    if 'table_theme' in features_dict:
        table_theme = features_dict['table_theme'].upper().strip()
        if table_theme in TABLE_THEME_VALID_VALUES:
            features_transformed_dict['table_theme'] = table_theme

    features_transformed_dict['table_theme_dummy'] = TABLE_THEME_OHE.transform(TABLE_THEME_LE.transform(
        [features_transformed_dict['table_theme']]).reshape(1, -1)).toarray()[0]

    wide_features.extend(features_transformed_dict['table_theme_dummy'])

    # Wide Feature: Column Name Length
    features_transformed_dict['column_name_length'] = 0.0
    if 'column_name' in features_dict:
        column_name_length = len(features_dict['column_name'].strip()) / COLUMN_NAME_MAX_LEN
        features_transformed_dict['column_name_length'] = \
            column_name_length if column_name_length < 1.0 else 1.0

    wide_features.append(features_transformed_dict['column_name_length'])

    # Wide Feature: Column Comment Length
    features_transformed_dict['column_comment_length'] = 0.0
    if 'column_comment' in features_dict:
        column_comment_length = len(features_dict['column_comment'].strip()) / COLUMN_COMMENT_MAX_LEN
        features_transformed_dict['column_comment_length'] = \
            column_comment_length if column_comment_length < 1.0 else 1.0

    wide_features.append(features_transformed_dict['column_comment_length'])

    # Wide Feature: Column Type
    features_transformed_dict['column_type'] = COLUMN_TYPE_VALUE_DEFAULT
    if 'column_type' in features_dict:
        column_type = features_dict['database_type'].split('(')[0].upper().strip()
        if column_type in COLUMN_TYPE_VALID_VALUES:
            features_transformed_dict['column_type'] = column_type

    features_transformed_dict['column_type_dummy'] = COLUMN_TYPE_OHE.transform(COLUMN_TYPE_LE.transform(
        [features_transformed_dict['column_type']]).reshape(1, -1)).toarray()[0]

    wide_features.extend(features_transformed_dict['column_type_dummy'])

    # Wide Feature: Column Is Primary Key
    features_transformed_dict['column_is_primary_key'] = COLUMN_IS_PRIMARY_KEY_VALUE_DEFAULT
    if 'column_is_primary_key' in features_dict:
        column_is_primary_key = features_dict['column_is_primary_key'].upper().strip()
        if column_is_primary_key in COLUMN_IS_PRIMARY_KEY_VALID_VALUES:
            features_transformed_dict['column_is_primary_key'] = column_is_primary_key

    features_transformed_dict['column_is_primary_key_dummy'] = \
        COLUMN_IS_PRIMARY_KEY_OHE.transform(COLUMN_IS_PRIMARY_KEY_LE.transform(
            [features_transformed_dict['column_is_primary_key']]).reshape(1, -1)).toarray()[0]

    wide_features.extend(features_transformed_dict['column_is_primary_key_dummy'])

    # Wide Feature: Column Allow Null
    features_transformed_dict['column_allow_null'] = COLUMN_ALLOW_NULL_VALUE_DEFAULT
    if 'column_allow_null' in features_dict:
        column_allow_null = features_dict['column_allow_null'].upper().strip()
        if column_allow_null in COLUMN_ALLOW_NULL_VALID_VALUES:
            features_transformed_dict['column_allow_null'] = column_allow_null

    features_transformed_dict['column_allow_null_dummy'] = COLUMN_ALLOW_NULL_OHE.transform(
        COLUMN_ALLOW_NULL_LE.transform([features_transformed_dict['column_allow_null']]).reshape(1, -1)).toarray()[0]

    wide_features.extend(features_transformed_dict['column_allow_null_dummy'])

    # Wide Feature: Column Value Length
    features_transformed_dict['column_value_length'] = 0.0
    if 'column_value' in features_dict:
        column_value_length = len(features_dict['column_value'].strip()) / COLUMN_VALUE_MAX_LEN
        features_transformed_dict['column_value_length'] = \
            column_value_length if column_value_length < 1.0 else 1.0

    wide_features.append(features_transformed_dict['column_value_length'])

    # Wide Features
    features_transformed_dict['wide_features'] = wide_features

    assert len(wide_features) == TRANSFORMED_WIDE_FEATURES_LEN

    return features_transformed_dict


def transform_wide_features_index_row(index_row) -> dict:
    return transform_wide_features(index_row[1].to_dict())['wide_features']


def transform_wide_features_parallel(
        column_value_with_meta: pd.DataFrame,
        n_jobs=DEFAULT_PARALLEL_NUM_JOBS,
        chunksize=DEFAULT_CHUNK_SIZE) -> list:
    """ 并行 Wide 特征变换

    Args:
        column_value_with_meta: 待转换特征 DataFrame
        n_jobs: 并行数量
        chunksize: 进程处理块大小

    Returns:

    """

    wide_features = []

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for result in tqdm(pool.map(transform_wide_features_index_row,
                                    column_value_with_meta.iterrows(),
                                    chunksize=chunksize),
                           desc='Transforming Wide Features: ',
                           total=len(column_value_with_meta.index), unit=' ROW'):
            wide_features.append(result)

    return wide_features


def transform_deep_features(features_dict: dict, text_dict: dict) -> dict:
    """ Deep 特征变换

    Args:
        features_dict: 原始特征词典
        text_dict: 文本词典

    Returns:
        变换后特征词典

    """

    features_transformed_dict = {}

    ## Deep Features
    deep_features = []

    # Deep Feature: Database Name
    database_name = features_dict.get('database_name', '')
    features_transformed_dict['database_name'] = gen_text_seq(database_name, DATABASE_NAME_MAX_LEN, text_dict)
    deep_features.extend(features_transformed_dict['database_name'])

    # Deep Feature: Database Comment
    database_comment = features_dict.get('database_comment', '')
    features_transformed_dict['database_comment'] = gen_text_seq(database_comment, DATABASE_COMMENT_MAX_LEN, text_dict)
    deep_features.extend(features_transformed_dict['database_comment'])

    # Deep Feature: Table Name
    table_name = features_dict.get('table_name', '')
    features_transformed_dict['table_name'] = gen_text_seq(table_name, TABLE_NAME_MAX_LEN, text_dict)
    deep_features.extend(features_transformed_dict['table_name'])

    # Deep Feature: Table Comment
    table_comment = features_dict.get('table_comment', '')
    features_transformed_dict['table_comment'] = gen_text_seq(table_comment, TABLE_COMMENT_MAX_LEN, text_dict)
    deep_features.extend(features_transformed_dict['table_comment'])

    # Deep Feature: Column Name
    column_name = features_dict.get('column_name', '')
    features_transformed_dict['column_name'] = gen_text_seq(column_name, COLUMN_NAME_MAX_LEN, text_dict)
    deep_features.extend(features_transformed_dict['column_name'])

    # Deep Feature: Column Comment
    column_comment = features_dict.get('column_comment', '')
    features_transformed_dict['column_comment'] = gen_text_seq(column_comment, COLUMN_COMMENT_MAX_LEN, text_dict)
    deep_features.extend(features_transformed_dict['column_comment'])

    # Deep Feature: Column Value
    column_value = features_dict.get('column_value', '')
    features_transformed_dict['column_value'] = gen_text_seq(column_value, COLUMN_VALUE_MAX_LEN, text_dict)
    deep_features.extend(features_transformed_dict['column_value'])

    ## Depp Features
    features_transformed_dict['deep_features'] = deep_features

    assert len(deep_features) == TRANSFORMED_DEEP_FEATURES_LEN

    return features_transformed_dict


def transform_deep_features_index_row(index_row, text_dict: dict = None) -> dict:
    return transform_deep_features(index_row[1].to_dict(), text_dict)['deep_features']


def transform_deep_features_parallel(
        column_value_with_meta: pd.DataFrame,
        text_dict: dict,
        n_jobs=DEFAULT_PARALLEL_NUM_JOBS,
        chunksize=DEFAULT_CHUNK_SIZE) -> list:
    """ 并行 Deep 特征变换

    Args:
        column_value_with_meta: 待转换特征 DataFrame
        text_dict: 文本词典
        n_jobs: 并行数量
        chunksize: 进程处理块大小

    Returns:

    """

    transform_deep_features_map_fun = functools.partial(transform_deep_features_index_row, text_dict=text_dict)

    deep_features = []

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for result in tqdm(pool.map(transform_deep_features_map_fun,
                                    column_value_with_meta.iterrows(),
                                    chunksize=chunksize),
                           desc='Transforming Deep Features: ',
                           total=len(column_value_with_meta.index), unit=' ROW'):
            deep_features.append(result)

    return deep_features


def transform_is_sensitive_labels(label: str):
    """ 是否敏感标签转换

    Args:
        label: 标签

    Returns:

    """

    label_ = label.upper().strip()

    if label_ == '1':
        return 1
    else:
        return 0


def transform_is_sensitive_labels_parallel(
        column_value_with_meta: pd.DataFrame,
        n_jobs=DEFAULT_PARALLEL_NUM_JOBS,
        chunksize=DEFAULT_CHUNK_SIZE) -> list:
    """ 并行是否敏感标签转换

    Args:
        column_value_with_meta: 待转换特征 DataFrame
        n_jobs: 并行数量
        chunksize: 进程处理块大小

    Returns:

    """

    labels = []

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for result in tqdm(pool.map(transform_is_sensitive_labels,
                                    column_value_with_meta['column_is_sensitive'],
                                    chunksize=chunksize),
                           desc='Transforming Is Sensitive Labels: ',
                           total=len(column_value_with_meta.index), unit=' ROW'):
            labels.append(result)

    return labels


def transform_sensitive_type_labels(label: str):
    """ 敏感类型标签转换

    Args:
        label: 标签

    Returns:

    """

    label_ = label.upper().strip()

    if label_ == '':
        label_ = COLUMN_SENSITIVE_TYPE_VALUE_NOT
    elif not label_ in COLUMN_SENSITIVE_TYPE_VALID_VALUES:
        label_ = COLUMN_SENSITIVE_TYPE_VALUE_OTHER

    return list(COLUMN_SENSITIVE_TYPE_OHE.transform(
        COLUMN_SENSITIVE_TYPE_LE.transform([label_]).reshape(1, -1)).toarray()[0])


def transform_sensitive_type_labels_parallel(
        column_value_with_meta: pd.DataFrame,
        n_jobs=DEFAULT_PARALLEL_NUM_JOBS,
        chunksize=DEFAULT_CHUNK_SIZE) -> list:
    """ 并行敏感类型标签转换

    Args:
        column_value_with_meta: 待转换特征 DataFrame
        n_jobs: 并行数量
        chunksize: 进程处理块大小

    Returns:

    """

    labels = []

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for result in tqdm(pool.map(transform_sensitive_type_labels,
                                    column_value_with_meta['column_sensitive_type'],
                                    chunksize=chunksize),
                           desc='Transforming Sensitive Type Labels: ',
                           total=len(column_value_with_meta.index), unit=' ROW'):
            labels.append(result)

    return labels


def split_dataset(
        column_meta_file_path: str, test_frac: float,
        column_meta_train_file_path: str, column_meta_test_file_path: str):
    column_meta = pd.read_csv(column_meta_file_path, sep='\t', quotechar='"', dtype=str, error_bad_lines=False)
    column_meta = column_meta.dropna(subset=['column_is_sensitive'])
    column_meta = column_meta.replace(np.nan, '')

    column_meta_train, column_meta_test = train_test_split(column_meta, test_size=test_frac)

    column_meta_train.to_csv(column_meta_train_file_path, sep='\t', index=False)
    column_meta_test.to_csv(column_meta_test_file_path, sep='\t', index=False)


def build_dataset(
        column_meta_file_path: str=None, column_value_file_path: str=None, text_dict: dict=None,
        n_jobs=DEFAULT_PARALLEL_NUM_JOBS, chunksize=DEFAULT_CHUNK_SIZE,
        load_directly=False, cache_dataset=True, over_sampling=False,
        wide_features_pickle_file_path: str=None,
        deep_features_pickle_file_path: str=None,
        is_sensitive_labels_pickle_file_path: str=None,
        sensitive_type_labels_pickle_file_path: str=None):
    """ 构建数据集 (训练，验证和测试)

    Args:
        column_meta_file_path: 字段元信息文件路径
        column_value_file_path: 字段值信息文件路径
        text_dict: 文本词典
        n_jobs: 并行数
        chunksize: 进程处理块大小
        load_directly: 直接载入
        cache_dataset: 缓存数据
        over_sampling: 是否针对不均衡样本进行过采样
        wide_features_pickle_file_path: Wide 特征 PICKLE 文件路径
        deep_features_pickle_file_path: Deep 特征 PICKLE 文件路径
        is_sensitive_labels_pickle_file_path: 是否敏感标签 PICKLE 文件路径
        sensitive_type_labels_pickle_file_path: 敏感类型标签 PICKLE 文件路径

    Returns:
        数据集

    """

    if load_directly:
        # Load directly
        LOGGER.info('Loading wide features from [{wide_features_pickle_file_path}] ...'
                    .format(wide_features_pickle_file_path=wide_features_pickle_file_path))
        with open(wide_features_pickle_file_path, 'rb') as f:
            wide_features = pickle.load(f)

        LOGGER.info('Loading deep features from [{deep_features_pickle_file_path}] ...'
                    .format(deep_features_pickle_file_path=deep_features_pickle_file_path))
        with open(deep_features_pickle_file_path, 'rb') as f:
            deep_features = pickle.load(f)

        LOGGER.info('Loading is sensitive labels from [{is_sensitive_labels_pickle_file_path}] ...'
                    .format(is_sensitive_labels_pickle_file_path=is_sensitive_labels_pickle_file_path))
        with open(is_sensitive_labels_pickle_file_path, 'rb') as f:
            is_sensitive_labels = pickle.load(f)

        LOGGER.info('Loading sensitive type labels from [{sensitive_type_labels_pickle_file_path}] ...'
                    .format(sensitive_type_labels_pickle_file_path=sensitive_type_labels_pickle_file_path))
        with open(sensitive_type_labels_pickle_file_path, 'rb') as f:
            sensitive_type_labels = pickle.load(f)
    else:
        # Load data
        LOGGER.info('Loading column meta file from [{column_meta_file_path}] ...'
                    .format(column_meta_file_path=column_meta_file_path))
        column_meta = pd.read_csv(column_meta_file_path, sep='\t', quotechar='"', dtype=str, error_bad_lines=False)
        column_meta['stage_table_name'] = column_meta['stage_table_name'].str.upper().str.strip()
        column_meta['stage_column_name'] = column_meta['stage_column_name'].str.upper().str.strip()

        LOGGER.info('Loading column value file from [{column_value_file_path}] ...'
                    .format(column_value_file_path=column_value_file_path))
        column_value = pd.read_csv(column_value_file_path, sep='\t', quotechar='"', dtype=str, error_bad_lines=False)
        column_value = column_value[['stage_table_name', 'stage_column_name', 'column_value']]
        column_value['stage_table_name'] = column_value['stage_table_name'].str.upper().str.strip()
        column_value['stage_column_name'] = column_value['stage_column_name'].str.upper().str.strip()
        # column_value = column_value.dropna(subset=['column_value'])

        # Merge data
        LOGGER.info('Merging column meta and value data ...')
        column_value_with_meta = pd.merge(column_value, column_meta, how='inner',
                                          on=['stage_table_name', 'stage_column_name'])
        column_value_with_meta = column_value_with_meta.replace(np.nan, '')

        if over_sampling:
            LOGGER.info('Over sampling ...')

            column_value_with_meta_minority_class = column_value_with_meta[
                column_value_with_meta['column_is_sensitive'] == '1']
            column_value_with_meta_other_class = column_value_with_meta[
                column_value_with_meta['column_is_sensitive'] != '1']
            frac = len(column_value_with_meta_other_class.index) / len(column_value_with_meta_minority_class.index)
            repeated_times = math.ceil(frac)
            LOGGER.info('Repeated times for minority class: {repeated_times}'.format(repeated_times=repeated_times))

            column_value_with_meta_concat_list = [column_value_with_meta_minority_class] * repeated_times
            column_value_with_meta_concat_list.append(column_value_with_meta_other_class)

            column_value_with_meta = pd.concat(column_value_with_meta_concat_list)


        LOGGER.info('Shuffling column meta and value data ...')
        column_value_with_meta = column_value_with_meta.sample(frac=1.0)
        del column_meta
        del column_value
        gc.collect()

        # Transform data
        LOGGER.info('Transforming wide features ...')
        wide_features = transform_wide_features_parallel(
            column_value_with_meta, n_jobs=n_jobs, chunksize=chunksize)

        LOGGER.info('Transforming deep features ...')
        deep_features = transform_deep_features_parallel(
            column_value_with_meta, text_dict, n_jobs=n_jobs, chunksize=chunksize)

        LOGGER.info('Transforming is sensitive labels ...')
        is_sensitive_labels = transform_is_sensitive_labels_parallel(
            column_value_with_meta, n_jobs=n_jobs, chunksize=chunksize)

        LOGGER.info('Transforming sensitive type labels ...')
        sensitive_type_labels = transform_sensitive_type_labels_parallel(
            column_value_with_meta, n_jobs=n_jobs, chunksize=chunksize)

        # Cache
        if cache_dataset:
            LOGGER.info('Dumping wide features to [{wide_features_pickle_file_path}] ...'
                        .format(wide_features_pickle_file_path=wide_features_pickle_file_path))
            with open(wide_features_pickle_file_path, 'wb') as f:
                pickle.dump(wide_features, f)

            LOGGER.info('Dumping deep features to [{deep_features_pickle_file_path}] ...'
                        .format(deep_features_pickle_file_path=deep_features_pickle_file_path))
            with open(deep_features_pickle_file_path, 'wb') as f:
                pickle.dump(deep_features, f)

            LOGGER.info('Dumping is sensitive labels to [{is_sensitive_labels_pickle_file_path}] ...'
                        .format(is_sensitive_labels_pickle_file_path=is_sensitive_labels_pickle_file_path))
            with open(is_sensitive_labels_pickle_file_path, 'wb') as f:
                pickle.dump(is_sensitive_labels, f)

            LOGGER.info('Dumping sensitive type labels to [{sensitive_type_labels_pickle_file_path}] ...'
                        .format(sensitive_type_labels_pickle_file_path=sensitive_type_labels_pickle_file_path))
            with open(sensitive_type_labels_pickle_file_path, 'wb') as f:
                pickle.dump(sensitive_type_labels, f)

    return wide_features, deep_features, is_sensitive_labels, sensitive_type_labels


def batch_iter(
        wide_features, deep_features, is_sensitive_labels, sensitive_type_labels,
        batch_size: int, num_epochs: int):
    """

    Args:
        wide_features: Wide 特征
        deep_features: Deep 特征
        is_sensitive_labels: 是否敏感标签
        sensitive_type_labels: 敏感类型标签
        batch_size: 批大小
        num_epochs: 轮数

    Returns:

    """

    wide_features_array = np.array(wide_features)
    deep_features_array = np.array(deep_features)
    is_sensitive_labels_array = np.array(is_sensitive_labels).reshape(-1, 1)
    sensitive_type_labels_array = np.array(sensitive_type_labels)

    num_batches_per_epoch = (len(is_sensitive_labels) - 1) // batch_size + 1

    for _ in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            s = batch_num * batch_size
            e = min((batch_num + 1) * batch_size, len(is_sensitive_labels))
            yield wide_features_array[s:e], deep_features_array[s:e], \
                  is_sensitive_labels_array[s:e], sensitive_type_labels_array[s:e]

