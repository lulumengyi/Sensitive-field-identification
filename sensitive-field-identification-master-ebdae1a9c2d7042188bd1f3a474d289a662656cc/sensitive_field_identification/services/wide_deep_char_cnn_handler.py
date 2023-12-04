#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy

from collections import Counter
from flask_restful import Resource, request
from jsonschema import Draft4Validator

from ..utils.constant import *
from ..utils.data_utils import build_char_dict, transform_wide_features, transform_deep_features
from ..models.wide_deep_char_cnn import WideDeepCharCNN
from ..utils.ump import monitor

dict_pickle_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/char_dict.pickle'))
checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/wide_deep_char_cnn_checkpoint'))

text_dict = build_char_dict(char_dict_pickle_file_path=dict_pickle_file_path)
model = WideDeepCharCNN(
    wide_features_len=TRANSFORMED_WIDE_FEATURES_LEN,
    deep_features_len=TRANSFORMED_DEEP_FEATURES_LEN,
    vocabulary_size=len(text_dict),
    num_class=COLUMN_SENSITIVE_TYPE_VALUES_LEN)
model.restore(checkpoint_dir)

post_json_schema = {
    'type': 'object',
    'properties': {
        'database_port': {'type': 'string'},
        'database_name': {'type': 'string'},
        'database_comment': {'type': 'string'},
        'database_type': {'type': 'string'},
        'database_is_shard': {'type': 'string'},
        'table_name': {'type': 'string'},
        'table_comment': {'type': 'string'},
        'table_theme': {'type': 'string'},
        'column_name': {'type': 'string'},
        'column_comment': {'type': 'string'},
        'column_type': {'type': 'string'},
        'column_is_primary_key': {'type': 'string'},
        'column_allow_null': {'type': 'string'},
        'column_value': {'type': 'array', 'items': {'type': 'string'}}
    },
    'required': [
        'database_port', 'database_name','database_comment', 'database_type', 'database_is_shard', 'table_name',
        'table_comment', 'table_theme', 'column_name', 'column_comment',
        'column_type', 'column_is_primary_key', 'column_allow_null', 'column_value'
    ]
}

post_json_validator = Draft4Validator(post_json_schema)


class WideDeepCharCNNHandler(Resource):
    def predict(self, raw_features_without_column_value_dict: dict, column_value_list: list) -> dict:
        """ 预测

        Args:
            raw_features_without_column_value_dict: 原始特征字典
            column_value_list: COLUMN VALUE 列表

        Returns:
            预测结果字典

        """

        predictions = []

        if len(column_value_list) == 0:
            column_value_list = ['']

        for column_value in column_value_list:
            raw_features_dict = copy.deepcopy(raw_features_without_column_value_dict)
            raw_features_dict['column_value'] = column_value

            wide_features = transform_wide_features(raw_features_dict)['wide_features']
            deep_features = transform_deep_features(raw_features_dict, text_dict)['deep_features']

            prediction = model.predict(wide_features, deep_features)
            predictions.append(prediction)

        label, counts = Counter(predictions).most_common(1)[0]
        probability = 0 if len(predictions) == 0 else counts / len(predictions)

        result = {
            'label': label,
            'raw_label': COLUMN_SENSITIVE_TYPE_LE.inverse_transform(label),
            'probability': probability,
            'predictions': predictions
        }

        return result

    @monitor('sensitive-field-identification', 'sensitive-field-identification-WideDeepCharCNNHandler-post')
    def post(self):
        args = request.get_json()

        result = {'status': STATUS_OK, 'error_msg': ''}

        try:
            post_json_validator.validate(args, post_json_schema)
        except Exception as e:
            result['status'] = STATUS_PARAMETERS_ERROR
            result['error_msg'] = getattr(e, 'message', repr(e))

        if result['status'] != STATUS_OK:
            return result

        raw_features_without_column_value_dict = args
        column_value_list = args['column_value']

        prediction = self.predict(raw_features_without_column_value_dict, column_value_list)
        result = {**result, **prediction}

        return result
