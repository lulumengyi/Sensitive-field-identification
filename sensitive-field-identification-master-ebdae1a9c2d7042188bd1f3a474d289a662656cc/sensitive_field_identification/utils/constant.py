#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np
import zhon.hanzi as hz
import os
import logging
import logging.config

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 特殊标记
PAD = '<PAD>'  # 补全的空白字符
UNK = '<UNK>'  # 未知字符
EOS = '<EOS>'  # 结束字符

# 中英文非句尾标点
non_stops_zh = hz.non_stops
non_stops_zh_en = (
    # Fullwidth ASCII variants
    '"#$%&\'()*+,-'
    '/:;<=>@[\\]^_'
    '`{|}~()'

    # Halfwidth CJK punctuation
    '(),'

    # CJK symbols and punctuation
    ' , '

    # CJK angle and corner brackets
    '()()()()()'

    # CJK brackets and symbols/punctuation
    '()()()()~"""'

    # Other CJK symbols
    ' '

    # Special CJK indicators
    '  '

    # Dashes
    '--'

    # Quotation marks and apostrophe
    '\'\'\'""""'

    # General punctuation
    '  '

    # Overscores and underscores
    '_'

    # Small form variants
    ',;'

    # Latin punctuation
    ' '
)

# 中英文句尾标点
stops_zh = hz.stops
stops_zh_en = (
    '!'  # Fullwidth exclamation mark
    '?'  # Fullwidth question mark
    '.'  # Halfwidth ideographic full stop
    '.'  # Ideographic full stop
)

# 中英文标点
punctuation_zh = hz.punctuation
punctuation_zh_en = non_stops_zh_en + stops_zh_en

assert len(punctuation_zh) == len(punctuation_zh_en)

# 保留字符的正则
keep_pattern = re.compile(r'[^A-Za-z0-9(),!?\'\"{chinese}]'.format(chinese=hz.characters))

# 清理多个连续空白字符的正则
multi_space_pattern = re.compile(r'\s{2,}')

# 字段元信息文件列名
COLUMN_META_COLUMN_NAMES = [
    'stage_table_name', 'stage_column_name', 'database_uri', 'database_port', 'database_name', 'database_comment',
    'database_type', 'database_is_shard_table', 'table_name', 'table_comment', 'table_theme', 'column_name',
    'column_comment', 'column_type', 'column_is_primary_key', 'column_allow_null',
    'column_is_sensitive', 'column_sensitive_type']
# 字段值数据文件列名
COLUMN_VALUE_COLUMN_NAMES = [
    'stage_table_name', 'stage_column_name', 'column_value', 'column_is_sensitive', 'column_sensitive_type']

# 数据库端口
DATABASE_PORT_VALID_VALUES = [
    '1433', '1521', '2181', '3306', '3307', '3308', '3309', '3310', '3358',
    '5088', '8020', '27017', '27018', '27019', '49638', '51899']
DATABASE_PORT_VALUE_DEFAULT = 'PORT_DEFAULT'
DATABASE_PORT_VALUES = [DATABASE_PORT_VALUE_DEFAULT]
DATABASE_PORT_VALUES.extend(DATABASE_PORT_VALID_VALUES)
DATABASE_PORT_VALUES_LEN = len(DATABASE_PORT_VALUES)
DATABASE_PORT_LE = LabelEncoder().fit(DATABASE_PORT_VALUES)
DATABASE_PORT_OHE = OneHotEncoder().fit(np.array(range(DATABASE_PORT_VALUES_LEN)).reshape(-1, 1))

# 数据库名称
DATABASE_NAME_MAX_LEN = 25

# 数据库注释
DATABASE_COMMENT_MAX_LEN = 10

# 数据库类型
DATABASE_TYPE_VALID_VALUES = [
    'DEL-MYSQL', 'FILE', 'HBASE', 'HIVE', 'JDMETA', 'JRDW', 'JREP', 'JRTC',
    'MONGODB', 'MYSQL', 'ORACLE', 'SQLSERVER', 'TEXT']
DATABASE_TYPE_VALUE_DEFAULT = 'DATABASE_TYPE_DEFAULT'
DATABASE_TYPE_VALUES = [DATABASE_TYPE_VALUE_DEFAULT]
DATABASE_TYPE_VALUES.extend(DATABASE_TYPE_VALID_VALUES)
DATABASE_TYPE_VALUES_LEN = len(DATABASE_TYPE_VALUES)
DATABASE_TYPE_LE = LabelEncoder().fit(DATABASE_TYPE_VALUES)
DATABASE_TYPE_OHE = OneHotEncoder().fit(np.array(range(DATABASE_TYPE_VALUES_LEN)).reshape(-1, 1))

# 数据库是否分库分表
DATABASE_IS_SHARD_VALID_VALUES = ['1', '0']
DATABASE_IS_SHARD_VALUE_DEFAULT = 'DATABASE_IS_SHARD_DEFAULT'
DATABASE_IS_SHARD_VALUES = [DATABASE_IS_SHARD_VALUE_DEFAULT]
DATABASE_IS_SHARD_VALUES.extend(DATABASE_IS_SHARD_VALID_VALUES)
DATABASE_IS_SHARD_VALUES_LEN = len(DATABASE_IS_SHARD_VALUES)
DATABASE_IS_SHARD_LE = LabelEncoder().fit(DATABASE_IS_SHARD_VALUES)
DATABASE_IS_SHARD_OHE = OneHotEncoder().fit(np.array(range(DATABASE_IS_SHARD_VALUES_LEN)).reshape(-1, 1))

# 数据表名称
TABLE_NAME_MAX_LEN = 60

# 数据表注释
TABLE_COMMENT_MAX_LEN = 25

# 数据表主题
TABLE_THEME_VALID_VALUES = [
    'APP', '京东商城', '众筹', '供应链金融', '保障险', '其他',
    '农村金融', '外部数据', '客服', '市场', '支付', '流量',
    '消费金融', '理财', '用户生态', '财务', '账户融合', '风控']
TABLE_THEME_VALUE_DEFAULT = 'TABLE_THEME_DEFAULT'
TABLE_THEME_VALUES = [TABLE_THEME_VALUE_DEFAULT]
TABLE_THEME_VALUES.extend(TABLE_THEME_VALID_VALUES)
TABLE_THEME_VALUES_LEN = len(TABLE_THEME_VALUES)
TABLE_THEME_LE = LabelEncoder().fit(TABLE_THEME_VALUES)
TABLE_THEME_OHE = OneHotEncoder().fit(np.array(range(TABLE_THEME_VALUES_LEN)).reshape(-1, 1))

# 字段名称
COLUMN_NAME_MAX_LEN = 30

# 字段注释
COLUMN_COMMENT_MAX_LEN = 30

# 字段类型
COLUMN_TYPE_VALID_VALUES = [
    'BIGINT', 'CHAR', 'DATETIME', 'DECIMAL', 'DOUBLE', 'INT', 'NUMBER',
    'STRING', 'TIMESTAMP', 'TINYINT', 'VARCHAR', 'VARCHAR2']
COLUMN_TYPE_VALUE_DEFAULT = 'COLUMN_TYPE_DEFAULT'
COLUMN_TYPE_VALUES = [COLUMN_TYPE_VALUE_DEFAULT]
COLUMN_TYPE_VALUES.extend(COLUMN_TYPE_VALID_VALUES)
COLUMN_TYPE_VALUES_LEN = len(COLUMN_TYPE_VALUES)
COLUMN_TYPE_LE = LabelEncoder().fit(COLUMN_TYPE_VALUES)
COLUMN_TYPE_OHE = OneHotEncoder().fit(np.array(range(COLUMN_TYPE_VALUES_LEN)).reshape(-1, 1))

# 字段是否为主键
COLUMN_IS_PRIMARY_KEY_VALID_VALUES = ['1', '0']
COLUMN_IS_PRIMARY_KEY_VALUE_DEFAULT = 'COLUMN_IS_PRIMARY_KEY_DEFAULT'
COLUMN_IS_PRIMARY_KEY_VALUES = [COLUMN_IS_PRIMARY_KEY_VALUE_DEFAULT]
COLUMN_IS_PRIMARY_KEY_VALUES.extend(COLUMN_IS_PRIMARY_KEY_VALID_VALUES)
COLUMN_IS_PRIMARY_KEY_VALUES_LEN = len(COLUMN_IS_PRIMARY_KEY_VALUES)
COLUMN_IS_PRIMARY_KEY_LE = LabelEncoder().fit(COLUMN_IS_PRIMARY_KEY_VALUES)
COLUMN_IS_PRIMARY_KEY_OHE = OneHotEncoder().fit(np.array(range(COLUMN_IS_PRIMARY_KEY_VALUES_LEN)).reshape(-1, 1))

# 字段可否为空
COLUMN_ALLOW_NULL_VALID_VALUES = ['1', '0']
COLUMN_ALLOW_NULL_VALUE_DEFAULT = 'COLUMN_ALLOW_NULL_DEFAULT'
COLUMN_ALLOW_NULL_VALUES = [COLUMN_ALLOW_NULL_VALUE_DEFAULT]
COLUMN_ALLOW_NULL_VALUES.extend(COLUMN_ALLOW_NULL_VALID_VALUES)
COLUMN_ALLOW_NULL_VALUES_LEN = len(COLUMN_ALLOW_NULL_VALUES)
COLUMN_ALLOW_NULL_LE = LabelEncoder().fit(COLUMN_ALLOW_NULL_VALUES)
COLUMN_ALLOW_NULL_OHE = OneHotEncoder().fit(np.array(range(COLUMN_ALLOW_NULL_VALUES_LEN)).reshape(-1, 1))

# 字段值
COLUMN_VALUE_MAX_LEN = 50

# 字段元信息转换后特征长度
TRANSFORMED_WIDE_FEATURES_LEN = (
    DATABASE_PORT_VALUES_LEN # 数据库端口 database_port_dummy
    + 1 # 数据库名称长度 database_name_length
    + 1 # 数据库注释长度 database_comment_length
    + DATABASE_TYPE_VALUES_LEN # 数据库类型 database_type_dummy
    + DATABASE_IS_SHARD_VALUES_LEN # 数据库是否为分库分表 database_is_shard_dummy
    + 1 # 表名称长度 table_name_length
    + 1 # 表注释长度 table_comment_length
    + TABLE_THEME_VALUES_LEN # 表主题 table_theme_dummy
    + 1 # 字段名称长度 column_name_length
    + 1 # 字段注释长度 column_comment_length
    + COLUMN_TYPE_VALUES_LEN # 字段类型 column_type_dummy
    + COLUMN_IS_PRIMARY_KEY_VALUES_LEN # 字段是否是主键 column_is_primary_key_dummy
    + COLUMN_ALLOW_NULL_VALUES_LEN # 字段是否可以为空 column_allow_null
    + 1 # 字段值长度 column_value_length
)

# 字段值信息转换后特征长度
TRANSFORMED_DEEP_FEATURES_LEN = (
    DATABASE_NAME_MAX_LEN # 数据库名称 database_name
    + DATABASE_COMMENT_MAX_LEN # 数据库注释 database_comment
    + TABLE_NAME_MAX_LEN # 表名称 table_name
    + TABLE_COMMENT_MAX_LEN # 表注释 table_comment
    + COLUMN_NAME_MAX_LEN # 字段名称 column_name
    + COLUMN_COMMENT_MAX_LEN # 字段注释 column_comment
    + COLUMN_VALUE_MAX_LEN # 字段值 column_value
)

# 字段是否敏感
COLUMN_IS_SENSITIVE_VALUES = ['1', '0']
COLUMN_IS_SENSITIVE_VALUE_NOT = '0'
COLUMN_IS_SENSITIVE_VALUES_LEN = len(COLUMN_IS_SENSITIVE_VALUES)
COLUMN_IS_SENSITIVE_LE = LabelEncoder().fit(COLUMN_IS_SENSITIVE_VALUES)
COLUMN_IS_SENSITIVE_OHE = OneHotEncoder().fit(np.array(range(COLUMN_IS_SENSITIVE_VALUES_LEN)).reshape(-1, 1))

# 字段敏感类型
COLUMN_SENSITIVE_TYPE_VALID_VALUES = [
    '身份证', '姓名', '地址', '手机', '卡号', '其他', '固定电话', '邮箱', '京东PIN']
COLUMN_SENSITIVE_TYPE_VALUE_OTHER = 'UNKNOWN'
COLUMN_SENSITIVE_TYPE_VALUE_NOT = '非敏感'
COLUMN_SENSITIVE_TYPE_VALUES = [COLUMN_SENSITIVE_TYPE_VALUE_NOT, COLUMN_SENSITIVE_TYPE_VALUE_OTHER]
COLUMN_SENSITIVE_TYPE_VALUES.extend(COLUMN_SENSITIVE_TYPE_VALID_VALUES)
COLUMN_SENSITIVE_TYPE_VALUES_LEN = len(COLUMN_SENSITIVE_TYPE_VALUES)
COLUMN_SENSITIVE_TYPE_LE = LabelEncoder().fit(COLUMN_SENSITIVE_TYPE_VALUES)
COLUMN_SENSITIVE_TYPE_OHE = OneHotEncoder().fit(np.array(range(COLUMN_SENSITIVE_TYPE_VALUES_LEN)).reshape(-1, 1))

# 字段敏感类型
COLUMN_SENSITIVE_TYPE_VALID_VALUES = [
    '卡号', '固定电话', '地址', '姓名', '手机', '身份证', '邮箱']
COLUMN_SENSITIVE_TYPE_VALUE_NOT = '非敏感'
COLUMN_SENSITIVE_TYPE_VALUES = [COLUMN_SENSITIVE_TYPE_VALUE_NOT]
COLUMN_SENSITIVE_TYPE_VALUES.extend(COLUMN_SENSITIVE_TYPE_VALID_VALUES)
COLUMN_SENSITIVE_TYPE_VALUES_LEN = len(COLUMN_SENSITIVE_TYPE_VALUES)
COLUMN_SENSITIVE_TYPE_LE = LabelEncoder().fit(COLUMN_SENSITIVE_TYPE_VALUES)
COLUMN_SENSITIVE_TYPE_OHE = OneHotEncoder().fit(np.array(range(COLUMN_SENSITIVE_TYPE_VALUES_LEN)).reshape(-1, 1))

# 并行相关
DEFAULT_PARALLEL_NUM_JOBS = 4
DEFAULT_CHUNK_SIZE = 10000

# 日志相关
logging.config.fileConfig(os.path.join(os.path.dirname(__file__), 'logging.conf'))
LOGGER = logging.getLogger('default')

# WEB 相关
STATUS_OK = 200
STATUS_PARAMETERS_ERROR = 301
