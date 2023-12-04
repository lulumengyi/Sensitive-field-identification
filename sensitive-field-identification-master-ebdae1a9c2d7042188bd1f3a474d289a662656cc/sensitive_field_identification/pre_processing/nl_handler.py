#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .english_handler import EnglishHandler
from .chinese_handler import ChineseHandler

class NLHandler(object):
    def __init__(self):
        self._english_handler = EnglishHandler()
        self._chinese_handler = ChineseHandler()


    def tokenize(self, text: str, num_to_digits: bool = True) -> list:
        """ 分词

        Args:
            text: 待处理文本
            num_to_digits: 是否将数字拆分成单个数字

        Returns:

        """

        tokens = self._chinese_handler.tokenize_chinese(text)

        token_words = [token['word'] for token in tokens]
        token_words_ = list()

        for token_word in token_words:
            num_split_tokens = self.split_num(token_word, num_to_digits=num_to_digits)

            for num_split_token in num_split_tokens:
                english_tokens = self._english_handler.tokenize_english(num_split_token)

                if english_tokens is None or len(english_tokens) == 0:
                    token_words_.append(num_split_token)
                else:
                    token_words_.extend(english_tokens)

        return self._chinese_handler.tokenize_chinese(' '.join(token_words_))


    def split_num(self, text: str, num_to_digits: bool = True) -> list:
        """ 分割数字

        Args:
            text: 待处理文本
            num_to_digits: 是否将数字拆分成单个数字

        Returns:

        """

        tokens = list()

        text_buffer = ''
        last_char_type = ''

        for char in text:
            if char in '0123456789':
                current_char_type = 'DIGIT'
            else:
                current_char_type = 'OTHERS'

            if current_char_type != last_char_type or \
                    num_to_digits and current_char_type == 'DIGIT':
                if text_buffer != '':
                    tokens.append(text_buffer)
                text_buffer = char
            else:
                text_buffer = text_buffer + char

            last_char_type = current_char_type

        if text_buffer != '':
            tokens.append(text_buffer)

        return tokens
