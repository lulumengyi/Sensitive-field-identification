#!/usr/bin/env python
# -*- coding: utf-8 -*-

from corenlp import CoreNLPClient

class ChineseHandler(object):
    def __init__(self):
        self._core_nlp_client = CoreNLPClient(start_server=False, endpoint='http://127.0.0.1:10000')

    def tokenize_chinese(self, text: str):
        if text.strip() == '':
            return []

        ann = self._core_nlp_client.annotate(text, annotators='tokenize pos'.split())
        tokens = []

        for sentence in ann.sentence:
            for token in sentence.token:
                token_ = {
                    'word': token.word,
                    'pos': token.pos
                }

                tokens.append(token_)

        return tokens
