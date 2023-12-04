#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Server

Usage:
    server.py [-a=<A> | --host=<A>] [-p=<P> | --port=<P>] [-d | --debug]
    server.py (-h | --help)
    server.py --version

Options:
    -h --help                         显示帮助
    -v --version                      显示版本
    -a=<A> --host=<A>                 Host
    -p=<P> --port=<P>                 Port
    -d --debug                        是否开启 Debug

"""

from docopt import docopt
from flask import Flask
from flask_restful import Api

from sensitive_field_identification.services.wide_deep_char_cnn_handler import WideDeepCharCNNHandler
from sensitive_field_identification.utils.constant import LOGGER

app = Flask(__name__)
api = Api(app)

api.add_resource(WideDeepCharCNNHandler, '/WideDeepCharCNN')

if __name__ == '__main__':
    args = docopt(__doc__, version='Server 1.0.0')

    if args['--host']:
        host = args['--host']
    else:
        host = '0.0.0.0'

    if args['--port']:
        port = int(args['--port'])
    else:
        port = 9999

    if args['--debug']:
        debug = True
    else:
        debug = False

    LOGGER.info('Starting server ...')

    app.run(host=host, port=port, debug=debug)
