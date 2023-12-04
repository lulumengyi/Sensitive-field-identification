#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pickle
import os
import tensorflow as tf

from ..utils.constant import *
from ..utils.data_utils import build_char_dict
from ..models.wide_deep_char_cnn import WideDeepCharCNN

class WideDeepCharCNNTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(WideDeepCharCNNTestCase, self).__init__(*args, **kwargs)

        self.file_dir = os.path.dirname(__file__)

        dict_pickle_file_path = os.path.abspath(os.path.join(self.file_dir, '../../data/char_dict.pickle'))
        self._text_dict = build_char_dict(char_dict_pickle_file_path=dict_pickle_file_path)

        LOGGER.info('Loading test dataset ...')
        wide_features_test_pickle_file_path = os.path.abspath(
            os.path.join(self.file_dir,'../../data/wide_features_demo_test.pickle'))
        deep_features_test_pickle_file_path = os.path.abspath(
            os.path.join(self.file_dir, '../../data/deep_features_demo_test.pickle'))
        is_sensitive_labels_test_pickle_file_path = os.path.abspath(
            os.path.join(self.file_dir, '../../data/is_sensitive_labels_demo_test.pickle'))
        sensitive_type_labels_test_pickle_file_path = os.path.abspath(
            os.path.join(self.file_dir, '../../data/sensitive_type_labels_demo_test.pickle'))

        with open(wide_features_test_pickle_file_path, 'rb') as f:
            self._wide_features_test = pickle.load(f)
        with open(deep_features_test_pickle_file_path, 'rb') as f:
            self._deep_features_test = pickle.load(f)
        with open(is_sensitive_labels_test_pickle_file_path, 'rb') as f:
            self._is_sensitive_labels_test = pickle.load(f)
        with open(sensitive_type_labels_test_pickle_file_path, 'rb') as f:
            self._sensitive_type_labels_test = pickle.load(f)


    def test_train_01(self):
        wide_deep_char_cnn = WideDeepCharCNN(
            wide_features_len=TRANSFORMED_WIDE_FEATURES_LEN,
            deep_features_len=TRANSFORMED_DEEP_FEATURES_LEN,
            vocabulary_size=len(self._text_dict),
            sensitive_type_label=True,
            num_class=COLUMN_SENSITIVE_TYPE_VALUES_LEN)

        checkpoint_dir = os.path.abspath(os.path.join(self.file_dir,'../../data/wide_deep_char_cnn_checkpoint'))
        if not tf.gfile.Exists(checkpoint_dir):
            tf.gfile.MakeDirs(checkpoint_dir)

        summaries_dir = os.path.abspath(os.path.join(self.file_dir,'../../data/wide_deep_char_cnn_summaries'))
        if not tf.gfile.Exists(summaries_dir):
            tf.gfile.MakeDirs(summaries_dir)

        wide_deep_char_cnn.train(
            self._wide_features_test, self._deep_features_test,
            self._is_sensitive_labels_test, self._sensitive_type_labels_test,
            checkpoint_dir=checkpoint_dir, summaries_dir=summaries_dir)


if __name__ == '__main__':
    unittest.main()
