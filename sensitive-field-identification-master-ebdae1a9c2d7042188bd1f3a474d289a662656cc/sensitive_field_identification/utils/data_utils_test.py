#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy.testing as npt

from .data_utils import *

class DataUtilsTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DataUtilsTestCase, self).__init__(*args, **kwargs)

        file_dir = os.path.dirname(__file__)

        self._char_dict_text_file_path = os.path.abspath(os.path.join(file_dir, '../../data/char_dict.tsv'))
        self._word_dict_text_file_path = os.path.abspath(os.path.join(file_dir, '../../data/word_dict.tsv'))
        self._char_dict_pickle_file_path = os.path.abspath(os.path.join(file_dir, '../../data/char_dict.pickle'))
        self._word_dict_pickle_file_path = os.path.abspath(os.path.join(file_dir, '../../data/word_dict.pickle'))

        if os.path.exists(self._char_dict_pickle_file_path):
            self._char_dict = build_char_dict(char_dict_pickle_file_path=self._char_dict_pickle_file_path)

        if os.path.exists(self._word_dict_pickle_file_path):
            self._word_dict = build_word_dict(word_dict_pickle_file_path=self._word_dict_pickle_file_path)

        self._column_meta_file_path = os.path.abspath(os.path.join(file_dir, '../../data/column_meta.tsv'))
        self._column_value_file_path = os.path.abspath(os.path.join(file_dir, '../../data/column_value.tsv'))
        column_meta = pd.read_csv(self._column_meta_file_path, sep='\t', quotechar='"', dtype=str, error_bad_lines=False)
        column_meta['stage_table_name'] = column_meta['stage_table_name'].str.upper().str.strip()
        column_meta['stage_column_name'] = column_meta['stage_column_name'].str.upper().str.strip()
        column_value = pd.read_csv(self._column_value_file_path, sep='\t', quotechar='"', dtype=str, error_bad_lines=False)
        column_value = column_value[['stage_table_name', 'stage_column_name', 'column_value']]
        column_value['stage_table_name'] = column_value['stage_table_name'].str.upper().str.strip()
        column_value['stage_column_name'] = column_value['stage_column_name'].str.upper().str.strip()

        column_value_with_meta = pd.merge(column_value, column_meta, how='inner',
                                          on=['stage_table_name', 'stage_column_name'])
        column_value_with_meta = column_value_with_meta.replace(np.nan, '')
        self._column_value_with_meta = column_value_with_meta.sample(frac=1.0)

        del column_meta
        del column_value
        gc.collect()

        wide_features_test_pickle_file_path = os.path.abspath(
            os.path.join(file_dir, '../../data/wide_features_test.pickle'))
        deep_features_test_pickle_file_path = os.path.abspath(
            os.path.join(file_dir, '../../data/deep_features_test.pickle'))
        is_sensitive_labels_test_pickle_file_path = os.path.abspath(
            os.path.join(file_dir, '../../data/is_sensitive_labels_test.pickle'))
        sensitive_type_labels_test_pickle_file_path = os.path.abspath(
            os.path.join(file_dir, '../../data/sensitive_type_labels_test.pickle'))

        with open(wide_features_test_pickle_file_path, 'rb') as f:
            self._wide_features_test = pickle.load(f)
        with open(deep_features_test_pickle_file_path, 'rb') as f:
            self._deep_features_test = pickle.load(f)
        with open(is_sensitive_labels_test_pickle_file_path, 'rb') as f:
            self._is_sensitive_labels_test = pickle.load(f)
        with open(sensitive_type_labels_test_pickle_file_path, 'rb') as f:
            self._sensitive_type_labels_test = pickle.load(f)

    def test_zh_punc_to_en_punc_01(self):
        text = '中文。english.'
        text_ = zh_punc_to_en_punc(text)
        expected_text = '中文.english.'

        self.assertEqual(expected_text, text_)

    def test_clean_text_01(self):
        pass

    def test_build_char_dict(self):
        column_meta_dict_columns = ['database_name', 'database_comment', 'table_name', 'table_comment',
                                    'table_theme', 'column_name', 'column_comment']
        column_value_dict_columns = ['column_value']

        build_char_dict(
            self._char_dict_pickle_file_path, self._char_dict_text_file_path,
            self._column_meta_file_path, column_meta_dict_columns,
            self._column_value_file_path, column_value_dict_columns)

    def test_build_word_dict(self):
        column_meta_dict_columns = ['database_name', 'database_comment', 'table_name', 'table_comment',
                                    'table_theme', 'column_name', 'column_comment']
        column_value_dict_columns = ['column_value']

        build_word_dict(
            self._word_dict_pickle_file_path, self._word_dict_text_file_path,
            self._column_meta_file_path, column_meta_dict_columns,
            self._column_value_file_path, column_value_dict_columns)

    def test_transform_wide_features(self):
        features_dict = {
            'stage_table_name': 's_m85_acctdetail_0030_x',
            'stage_column_name': 'recpay_date',
            'column_value': '',
            'column_is_sensitive_x': '0',
            'column_sensitive_type_x': '0',
            'database_uri': '172.24.138.185',
            'database_port': '3306',
            'database_name': 'recpayable_',
            'database_comment': '',
            'database_type': 'mysql',
            'database_is_shard': '1',
            'table_name': 'acctdetail_0030_',
            'table_comment': 'pop商家险新台账数据表',
            'table_theme': '其他',
            'column_name': 'recpay_date',
            'column_comment': '应收应付时间',
            'column_type': 'datetime',
            'column_is_primary_key': '0',
            'column_allow_null': '1',
            'column_is_sensitive_y': '0',
            'column_sensitive_type_y': ''}
        features_transformed_dict = transform_wide_features(features_dict)

        wide_features = features_transformed_dict['wide_features']
        expected_wide_features = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.44, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.26666666666666666, 0.48, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.36666666666666664, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0]
        self.assertAlmostEqual(expected_wide_features, wide_features, delta=1e-6)

    def test_transform_deep_features(self):
        features_dict = {
            'stage_table_name': 's_m50_btlmk_acctdetail',
            'stage_column_name': 'update_date',
            'column_value': '2018-07-01 10:20:02',
            'database_uri': 'db-btrecpjrcw-02.pekdc1.jdfin.local',
            'database_port': '3306',
            'database_name': 'recpayable',
            'database_comment': '',
            'database_type': 'mysql',
            'database_is_shard': '0',
            'table_name': 'btlmk_acctdetail',
            'table_comment': '白条联名卡',
            'table_theme': '消费金融',
            'column_name': 'update_date',
            'column_comment': '更新时间',
            'column_type': 'timestamp',
            'column_is_primary_key': '0',
            'column_allow_null': '0',
            'column_is_sensitive': '0',
            'column_sensitive_type': ''}
        features_transformed_dict = transform_deep_features(features_dict, self._char_dict)

        deep_features = features_transformed_dict['deep_features']
        expected_deep_features = [
            25, 8, 22, 31, 13, 32, 13, 28, 7, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 28, 19, 7, 27, 36, 4, 13, 22, 22, 19, 21, 8, 19, 13, 24, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95, 90, 186, 104,
            130, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 31, 21, 13, 19, 8, 4, 21, 13, 19, 8,
            2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 245, 101, 81, 92, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 5, 9, 4, 3, 12, 4, 3, 5, 4, 5, 3, 4, 6, 3, 4, 3, 6, 2,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.assertEqual(expected_deep_features, deep_features)

    def test_transform_is_sensitive_label(self):
        label = '1'
        transformed_label = transform_is_sensitive_labels(label)
        expected_transformed_label = 1
        self.assertEqual(expected_transformed_label, transformed_label)

        label = '0'
        transformed_label = transform_is_sensitive_labels(label)
        expected_transformed_label = 0
        self.assertEqual(expected_transformed_label, transformed_label)

        label = 'others'
        transformed_label = transform_is_sensitive_labels(label)
        expected_transformed_label = 0
        self.assertEqual(expected_transformed_label, transformed_label)

        label = ''
        transformed_label = transform_is_sensitive_labels(label)
        expected_transformed_label = 0
        self.assertEqual(expected_transformed_label, transformed_label)

    def test_transform_sensitive_type_label(self):
        # Sorted COLUMN_SENSITIVE_TYPE_VALUES is list below:
        # ['UNKNOWN', '京东PIN', '其他', '卡号', '固定电话', '地址', '姓名', '手机', '身份证', '邮箱', '非敏感']

        label = '身份证'
        transformed_label = transform_sensitive_type_labels(label)
        expected_transformed_label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        npt.assert_array_almost_equal(expected_transformed_label, transformed_label)

        label = 'balabalabala'
        transformed_label = transform_sensitive_type_labels(label)
        expected_transformed_label = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        npt.assert_array_almost_equal(expected_transformed_label, transformed_label)

        label = ''
        transformed_label = transform_sensitive_type_labels(label)
        expected_transformed_label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        npt.assert_array_almost_equal(expected_transformed_label, transformed_label)

    def test_transform_wide_features_parallel(self):
        column_value_with_meta_samples = self._column_value_with_meta.head(10)
        wide_features = transform_wide_features_parallel(column_value_with_meta_samples)
        wide_features.sort()

        expected_wide_features = list(map(
            lambda index_row: transform_wide_features(index_row[1].to_dict())['wide_features'],
            tqdm(column_value_with_meta_samples.iterrows())))
        expected_wide_features.sort()

        npt.assert_array_almost_equal(expected_wide_features, wide_features)

    def test_transform_deep_features_parallel(self):
        column_value_with_meta_samples = self._column_value_with_meta.head(10)
        deep_features = transform_deep_features_parallel(column_value_with_meta_samples, self._char_dict)
        deep_features.sort()

        expected_deep_features = list(map(
            lambda index_row: transform_deep_features(index_row[1].to_dict(), self._char_dict)['deep_features'],
            tqdm(column_value_with_meta_samples.iterrows())))
        expected_deep_features.sort()

        npt.assert_array_almost_equal(expected_deep_features, deep_features)

    def test_transform_is_sensitive_label_parallel(self):
        column_value_with_meta_samples = self._column_value_with_meta.head(10)
        labels = transform_is_sensitive_labels_parallel(column_value_with_meta_samples)
        labels.sort()

        expected_labels = list(map(
            lambda label: transform_is_sensitive_labels(label),
            tqdm(column_value_with_meta_samples['column_is_sensitive'])))
        expected_labels.sort()

        self.assertEqual(expected_labels, labels)

    def test_transform_sensitive_type_label_parallel(self):
        column_value_with_meta_samples = self._column_value_with_meta.head(10)
        labels = transform_sensitive_type_labels_parallel(column_value_with_meta_samples)
        labels.sort()

        expected_labels = list(map(
            lambda label: transform_sensitive_type_labels(label),
            tqdm(column_value_with_meta_samples['column_sensitive_type'])))
        expected_labels.sort()

        npt.assert_array_almost_equal(expected_labels, labels)

    def test_batch_iter(self):
        sample_cnt = 100
        wide_features_test = self._wide_features_test[0:sample_cnt]
        deep_features_test = self._deep_features_test[0:sample_cnt]
        is_sensitive_labels_test = self._is_sensitive_labels_test[0:sample_cnt]
        sensitive_type_labels_test = self._sensitive_type_labels_test[0:sample_cnt]

        batch_size = 10
        num_epochs = 2

        batches = batch_iter(
            wide_features_test, deep_features_test,
            is_sensitive_labels_test, sensitive_type_labels_test,
            batch_size, num_epochs)

        batches_cnt = 0

        for wide_features_batch, deep_features_batch, \
            is_sensitive_labels_batch, sensitive_type_labels_batch in batches:
            start_index = (batches_cnt * batch_size) % sample_cnt
            end_index = start_index + batch_size

            npt.assert_array_almost_equal(np.array(wide_features_test[start_index:end_index]), wide_features_batch)
            npt.assert_array_almost_equal(np.array(deep_features_test[start_index:end_index]), deep_features_batch)
            npt.assert_array_almost_equal(
                np.array(is_sensitive_labels_test[start_index:end_index]), is_sensitive_labels_batch)
            npt.assert_array_almost_equal(
                np.array(sensitive_type_labels_test[start_index:end_index]), sensitive_type_labels_batch)

            batches_cnt += 1

        self.assertEqual(sample_cnt // batch_size * 2, batches_cnt)


if __name__ == '__main__':
    unittest.main()
