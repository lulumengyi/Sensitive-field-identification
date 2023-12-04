#!/usr/bin/env python
# -*- coding: utf-8 -*-

import wordsegment as ws

class EnglishHandler(object):
    def __init__(self):
        ws.load()

    def tokenize_english(self, text: str):
        tokens = ws.segment(text)
        return tokens

