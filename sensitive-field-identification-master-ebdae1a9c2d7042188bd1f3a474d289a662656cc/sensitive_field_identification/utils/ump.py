#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import json
import logzero
import socket
import os
import random
import functools

from logzero import logger

from .constant import STATUS_OK

# 定义日志输出格式
log_format = '%(message)s'
formatter = logzero.LogFormatter(fmt=log_format)
logzero.setup_default_logger(formatter=formatter)
# 获取进程ID
pid = os.getpid()
log_file_date = datetime.datetime.now().strftime('%y%m%d%H%M%S%f')[:-3]
rand_num = random.randint(10000, 99999)

log_file = "%s_%s_%s_tp.log" % (log_file_date, pid, rand_num)
# 定义日志输出路径及切分规则
logzero.logfile("/export/home/tomcat/UMP-Monitor/logs/%s" % log_file, maxBytes=1000 * 1000 * 50, backupCount=3, formatter=formatter)


def monitor(appName, key):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            profile = Profiler(appName, key, True)
            profile.register_info()
            result = func(*args, **kw)
            if result['status'] == STATUS_OK:
                profile.register_info_end()
            else:
                profile.function_error()
            return result
        return wrapper
    return decorator

class Profiler(object):

    def __init__(self, app_name, key, enable_tp, heart=False):
        """
        :param app_name:    J-ONE 应用名
        :param key          监控点名称
        :param enable_tp:   是否开启TP监控
        :param heart:       是否开启心跳监控
        """
        self.__app_name = app_name
        self.__key = key
        self.__enable_tp = enable_tp
        self.__heart = heart

    def _start(self):
        """
        :return: 创建profiler实例时间
        """
        start_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        self.__start = int(start_str)

    def register_info(self):
        self._start()

    def register_info_end(self):
        json = self._register_info(0)
        logger.info("%s", json)

    def function_error(self):
        json = self._register_info(1)
        logger.info("%s", json)

    def _register_info(self, processState):
        """
        :param processState: 按UMP平台规定——程序结束状态 0 正常 ；1 异常
        :return: 输出到日志的内容
        """
        # time.sleep(0.1)  # todo remove
        end_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        end = int(end_str)
        cost_time = end - self.__start

        ump_json = self._create_ump_json(self.__app_name, self.__key, processState, cost_time)
        return ump_json

    def _create_ump_json(self, appName, key, processState, elapsedTime):
        """
        根据UMP输出日志格式，生成规定的日志内容
        :param appName:
        :param key:
        :param processState:
        :param elapsedTime:
        :return:
        """
        return Formatter(self.__start, appName, key, socket.gethostname(), processState, elapsedTime)


class Formatter(object):
    def __init__(self, time, appName, key, hostname, processState, elapsedTime):
        """
        :param time: 该条日志产生时间 格式：yyyyMMddHHmmssSSS
        :param appName: 应用名
        :param key:为注册的监控点 key_name 值，须与 ump 平台配置的对应监控点 key 保持一致
        :param hostname:机器主机名或 IP，务必保证只使用一种
        :param processState:方法的执行状态，0 为正常，1 为异常。用于统计可用率
        :param elapsedTime:为方法执行消耗的时间，默认单位（毫秒 ms）
        """
        self.__time = time
        self.__appName = appName
        self.__key = key
        self.__hostname = hostname
        self._processState = processState
        self.__elapsedTime = elapsedTime

    def __str__(self):
        json_dict = {
            "time": self.__time,
            "appName": self.__appName,
            "key": self.__key,
            "hostname": self.__hostname,
            "processState": self._processState,
            "elapsedTime": self.__elapsedTime,
        }
        return json.dumps(json_dict)
