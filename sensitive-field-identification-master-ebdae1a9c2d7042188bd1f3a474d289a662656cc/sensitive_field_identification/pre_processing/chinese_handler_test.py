#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from .chinese_handler import ChineseHandler

class ChineseHandlerTestCase(unittest.TestCase):
    def setUp(self):
        self._chinese_handler = ChineseHandler()

    def tearDown(self):
        pass

    def test_tokenize_chinese_01(self):
        text = '这是斯坦福中文分词器测试'
        tokens = self._chinese_handler.tokenize_chinese(text)
        token_words = [token['word'] for token in tokens]
        expected_token_words = ['这', '是', '斯坦福', '中文', '分词', '器', '测试']

        self.assertEqual(expected_token_words, token_words)

    def test_tokenize_chinese_02(self):
        text = 'this is another test'
        tokens = self._chinese_handler.tokenize_chinese(text)
        token_words = [token['word'] for token in tokens]
        expected_token_words = ['this', 'is', 'another', 'test']

        self.assertEqual(expected_token_words, token_words)

    def test_tokenize_chinese_03(self):
        text = ' '
        tokens = self._chinese_handler.tokenize_chinese(text)
        expected_tokens = []

        self.assertEqual(expected_tokens, tokens)


if __name__ == '__main__':
    unittest.main()
