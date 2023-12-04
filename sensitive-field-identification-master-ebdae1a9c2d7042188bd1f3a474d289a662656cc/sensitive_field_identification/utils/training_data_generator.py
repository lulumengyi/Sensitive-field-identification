#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
from tqdm import tqdm

from .constant import LOGGER

PASSWORD = 'key:x17Rp28n36'
DT = '2018-07-01'

STAGE_TABLES_FILE_PATH = '../../data/all_stage_tables.tsv'
STAGE_TABLES_WITH_DT_FILE_PATH = '../../data/all_stage_tables_with_dt.tsv'
STAGE_TABLES_COLUMNS_FILE_PATH = '../../data/all_stage_tables_columns.tsv'


class TrainingDataGenerator(object):
    def __init__(
            self,
            tables_file_path: str,
            tables_with_dt_file_path: str,
            tables_columns_file_path: str):
        """

        Args:
            tables_file_path: 表名文件路径
            tables_with_dt_file_path: 表名文件路径 (包含 DT 字段的表)
            tables_columns_file_path:  表字段信息文件路径
        """

        self._tables_file_path = tables_file_path
        self._tables_with_dt_file_path = tables_with_dt_file_path
        self._tables_columns_file_path = tables_columns_file_path

        self._tables = {}

        self.load_tables()
        self.load_tables_dt()
        self.load_tables_column()

    def load_tables(self):
        """ 载入表信息
        """

        with open(self._tables_file_path, 'r') as f:
            f_ = csv.DictReader(f)
            for row in f_:
                table_name = row['table_name'].upper()
                self._tables[table_name] = {}
                self._tables[table_name]['dt'] = False
                self._tables[table_name]['insensitive_columns'] = []
                self._tables[table_name]['sensitive_columns'] = []
                self._tables[table_name]['sensitive_types'] = {}

    def load_tables_dt(self):
        """ 载入表信息 (包含 DT 字段的表)
        """

        with open(self._tables_with_dt_file_path, 'r') as f:
            f_ = csv.DictReader(f)
            for row in f_:
                table_name = row['table_name'].upper()

                if table_name in self._tables:
                    self._tables[table_name]['dt'] = True

    def load_tables_column(self):
        """ 载入表字段信息
        """

        with open(self._tables_columns_file_path) as f:
            f_ = csv.DictReader(f, delimiter='\t')
            for row in f_:
                table_name = row['table_name'].upper()
                column_name = row['column_name'].upper()
                column_is_sensitive = row['column_is_sensitive'].upper()
                column_sensitive_type = row['column_sensitive_type'].upper()

                if table_name in self._tables:
                    if column_is_sensitive == '1':
                        self._tables[table_name]['sensitive_columns'].append(column_name)
                        self._tables[table_name]['sensitive_types'][column_name] = column_sensitive_type
                    elif column_is_sensitive == '0':
                        self._tables[table_name]['insensitive_columns'].append(column_name)

    def column_need_quote(self, column_name: str) -> bool:
        """ 判断字段是否需要用引号 (`) 括起来

        Args:
            column_name: 字段名称

        Returns:

        """

        if column_name.startswith('_'):
            return True
        elif ':' in column_name:
            return True
        elif column_name.upper() in ['MORE', 'END', 'DIV', 'EXCHANGE', 'FROM', 'CURRENT', 'FUNCTION']:
            return True
        else:
            return False

    def gen_sensitive_column(self, column_name: str, password: str) -> str:
        """ 生成敏感字段查询字段

        Args:
            column_name: 字段名称
            password: 解密密码

        Returns:

        """

        if self.column_need_quote(column_name):
            column_name_ = '`{column_name}`'.format(column_name=column_name)
        else:
            column_name_ = column_name

        return "UNPASSWORD({column_name}, '{password}') AS {column_name}".format(
            column_name=column_name_, password=password)

    def gen_insensitive_column(self, column_name) -> str:
        """ 生成非敏感字段查询字段

        Args:
            column_name: 字段名称

        Returns:

        """

        if self.column_need_quote(column_name):
            return '`{column_name}`'.format(column_name=column_name)
        else:
            return column_name

    def gen_where_dt_condition(self, dt: str) -> str:
        """ 生成 DT 条件

        Args:
            dt: DT

        Returns:

        """

        return "dt = '{dt}'".format(dt=dt).strip()

    def gen_where_sensitive_not_null_condition(self, sensitive_columns: iter, password: str) -> str:
        """ 生成非空条件信息

        Args:
            sensitive_columns: 敏感字段
            password: 密码

        Returns:

        """

        columns = map(lambda column: "{column} IS NOT NULL AND "
                                     "UPPER({column}) != 'NULL' AND "
                                     "UNPASSWORD({column}, '{password}') IS NOT NULL AND "
                                     "UPPER(UNPASSWORD({column}, '{password}')) != 'NULL'"
                      .format(column=column, password=password), sensitive_columns)
        return ' AND '.join(columns).strip()

    def gen_insensitive_columns(self, insensitive_columns: iter) -> str:
        """ 生成非敏感字段

        Args:
            insensitive_columns: 非敏感字段

        Returns:

        """

        columns = map(lambda column_name: self.gen_insensitive_column(column_name), insensitive_columns)
        return ', '.join(columns).strip()

    def gen_sensitive_columns(self, sensitive_columns: iter, password: str) -> str:
        """ 生成敏感字段

        Args:
            sensitive_columns: 敏感字段
            password: 密码

        Returns:

        """

        columns = map(lambda column_name: self.gen_sensitive_column(column_name, password), sensitive_columns)
        return ', '.join(columns).strip()

    def gen_limit(self, limit: int) -> str:
        """ 生成查询条数限制

        Args:
            limit: 条数

        Returns:

        """

        return "DISTRIBUTE BY RAND() SORT BY RAND() LIMIT {limit}".format(limit=limit)

    def gen_header(self) -> str:
        """ 生成导出包含列名设置语句

        Returns:

        """

        return 'SET hive.cli.print.header=true;'

    def gen_columns(self, insensitive_columns: iter, sensitive_columns: iter, password: str) -> str:
        """ 生成所有字段的查询字段

        Args:
            insensitive_columns: 非敏感字段
            sensitive_columns: 敏感字段
            password: 密码

        Returns:

        """

        if len(insensitive_columns) == 0:
            return self.gen_sensitive_columns(sensitive_columns, password)
        elif len(sensitive_columns) == 0:
            return self.gen_insensitive_columns(insensitive_columns)
        else:
            return '{insensitive_columns}, {sensitive_columns}'.format(
                insensitive_columns = self.gen_insensitive_columns(insensitive_columns),
                sensitive_columns = self.gen_sensitive_columns(sensitive_columns, password))

    def gen_sqls(
            self,
            limit: int,
            password: str,
            dt: str,
            with_header: bool,
            sql_files_directory: str):
        """ 生成表的查询 SQL

        Args:
            limit: 条数
            password: 密码
            dt: DT
            with_header: 是否包含列名
            sql_files_directory: SQL 文件存储路径

        Returns:

        """

        for table_name, table_info in tqdm(self._tables.items(), unit=' files', ascii=True):
            # 如果不包含列，则不生成 SQL
            columns_count = len(table_info['insensitive_columns']) + len(table_info['sensitive_columns'])
            if columns_count == 0:
                continue

            columns = self.gen_columns(table_info['insensitive_columns'], table_info['sensitive_columns'], password)

            where_dt_condition = self.gen_where_dt_condition(dt)
            where_sensitive_not_null_condition = self.gen_where_sensitive_not_null_condition(
                table_info['sensitive_columns'], password)

            where_conditions = ['1 = 1']
            if where_sensitive_not_null_condition != '':
                where_conditions.append(where_sensitive_not_null_condition)
            if where_dt_condition != '':
                where_conditions.append(where_dt_condition)

            limit = self.gen_limit(limit)

            sql = 'SELECT {columns} FROM STAGE.{table_name} WHERE {where_conditions} {limit};'.format(
                columns=columns, table_name=table_name,
                where_conditions=' AND '.join(where_conditions), limit=limit)

            if with_header:
                sql = self.gen_header() + sql

            sql_file_path = '{directory}/{filename}.sql'.format(directory=sql_files_directory, filename=table_name)

            with open(sql_file_path, 'w') as f:
                f.write(sql)

    def gen_training_data(self, sql_result_files_directory: str, training_data_file_path: str):
        """ 根据 SQL 的运行结果导出的文件生成原始的训练数据

        Args:
            sql_result_files_directory: SQL 结果文件目录
            training_data_file_path: 训练数据文件路径

        Returns:

        """

        files = os.listdir(sql_result_files_directory)
        tsv_files = filter(lambda file: os.path.splitext(file)[1] == '.tsv', files)

        with open(training_data_file_path, 'w') as f:
            f_writer = csv.DictWriter(f, fieldnames=['stage_table_name', 'stage_column_name', 'column_value',
                                                     'column_is_sensitive', 'column_sensitive_type'],
                                      delimiter='\t')
            f_writer.writeheader()

            for tsv_file in tqdm(tsv_files, unit=' files', ascii=True):
                tsv_file_path = os.path.join(sql_result_files_directory, tsv_file)

                with open(tsv_file_path, 'r') as f_:
                    f_reader = csv.DictReader(f_, delimiter='\t')
                    table_name = os.path.splitext(tsv_file)[0].upper()

                    if table_name in self._tables:
                        table = self._tables[table_name]

                        try:
                            for row in f_reader:
                                # insensitive columns
                                for column_name in table['insensitive_columns']:
                                    f_writer.writerow({
                                        'stage_table_name': table_name.lower(), 'stage_column_name': column_name.lower(),
                                        'column_value': row[column_name.lower()],
                                        'column_is_sensitive': 0, 'column_sensitive_type': 0})

                                # sensitive columns
                                for column_name in table['sensitive_columns']:
                                    f_writer.writerow({
                                        'stage_table_name': table_name.lower(), 'stage_column_name': column_name.lower(),
                                        'column_value': row[column_name.lower()],
                                        'column_is_sensitive': 1,
                                        'column_sensitive_type': table['sensitive_types'][column_name]})
                        except Exception as err:
                            LOGGER.warning('Failed with table [{table_name}]'.format(table_name=table_name))


if __name__ == '__main__':
    generator = TrainingDataGenerator(
        STAGE_TABLES_FILE_PATH,
        STAGE_TABLES_WITH_DT_FILE_PATH,
        STAGE_TABLES_COLUMNS_FILE_PATH)

    # Generate Hive SQLs
    # generator.gen_sqls(100, PASSWORD, DT, True, '../../data/sqls')

    # Run Hive SQLs
    # Please use shell commands to do these.

    # Generate Training Data
    generator.gen_training_data('../../data/tsvs', '../../data/column_value.tsv')
