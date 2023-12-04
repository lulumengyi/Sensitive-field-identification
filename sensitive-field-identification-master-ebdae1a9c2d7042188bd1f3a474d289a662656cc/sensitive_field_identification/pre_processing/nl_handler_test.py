#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from .nl_handler import NLHandler

class NLHandlerCase(unittest.TestCase):
    def setUp(self):
        self._nlh = NLHandler()

    def test_tokenize_01(self):
        text = '我是中文thisisenglishtext我还是中文this is also english-text'
        tokens = self._nlh.tokenize(text)
        token_words = [token['word'] for token in tokens]
        expected_token_words = ['我', '是', '中文', 'this', 'is', 'english', 'text',
                                '我', '还是', '中文', 'this', 'is', 'also', 'english', 'text']

        self.assertEqual(expected_token_words, token_words)

    def test_tokenize_02(self):
        text = '我是中文thisis123englishtext'
        tokens = self._nlh.tokenize(text)
        token_words = [token['word'] for token in tokens]
        expected_token_words = ['我', '是', '中文', 'this', 'is', '1', '2', '3', 'english', 'text']

        self.assertEqual(expected_token_words, token_words)

    def test_split_num_01(self):
        text = 'english123中文456balabala'
        tokens = self._nlh.split_num(text)
        expected_tokens = ['english', '1', '2', '3', '中文', '4', '5', '6', 'balabala']

        self.assertEqual(expected_tokens, tokens)

    def test_split_num_02(self):
        text = 'english123中文456balabala'
        tokens = self._nlh.split_num(text, num_to_digits=False)
        expected_tokens = ['english', '123', '中文', '456', 'balabala']

        self.assertEqual(expected_tokens, tokens)


if __name__ == '__main__':
    unittest.main()
