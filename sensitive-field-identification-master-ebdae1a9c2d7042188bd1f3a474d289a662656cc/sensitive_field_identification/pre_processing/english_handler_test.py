#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from .english_handler import EnglishHandler

class EnglishHandlerTestCase(unittest.TestCase):
    def setUp(self):
        self._eh = EnglishHandler()

    def test_tokenize_english_01(self):
        text = 'thisisatest.'
        tokens = self._eh.tokenize_english(text)
        expected_tokens = ['this', 'is', 'a', 'test']

        self.assertEqual(expected_tokens, tokens)

    def test_tokenize_english_02(self):
        text = 'this-is-anothertest-withslash'
        tokens = self._eh.tokenize_english(text)
        expected_tokens = ['this', 'is', 'another', 'test', 'with', 'slash']

        self.assertEqual(expected_tokens, tokens)

    def test_tokenize_english_03(self):
        text = ''
        tokens = self._eh.tokenize_english(text)
        expected_tokens = []

        self.assertEqual(expected_tokens, tokens)


if __name__ == '__main__':
    unittest.main()
