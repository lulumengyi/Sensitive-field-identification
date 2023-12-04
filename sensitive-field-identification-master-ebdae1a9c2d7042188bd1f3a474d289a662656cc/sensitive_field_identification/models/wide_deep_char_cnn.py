#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import pandas as pd
import tableprint as tp
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from ..utils.constant import LOGGER, COLUMN_SENSITIVE_TYPE_LE
from ..utils.data_utils import batch_iter


class WideDeepCharCNN(object):
    def __init__(
            self, wide_features_len: int, deep_features_len: int,
            vocabulary_size: int, sensitive_type_label=True, num_class: int=None,
            wide_layers_units: list=None, wide_layers_dropout=0.5,
            deep_embedding_size=128, deep_num_filters: list=None,
            deep_kernel_sizes: list=None, deep_pooling_sizes: list=None,
            concat_layers_units: list=None, concat_layers_dropout=0.5,
            weighted_loss=False, class_weights: list=None,
            learning_rate=1e-3, optimizer=tf.train.AdamOptimizer):
        """

        Args:
            wide_features_len: Wide 特征长度
            deep_features_len: Deep 特征长度
            vocabulary_size: 词典大小
            sensitive_type_label: 标签为敏感类型
            num_class: 类个数
            wide_layers_units: Wide 部分隐含层节点个数
            wide_layers_dropout: Wide 部分 DROPOUT 比例
            deep_embedding_size: Deep 部分 EMBEDDING 维度
            deep_num_filters: Deep 部分 CNN Filter 个数
            deep_kernel_sizes: Deep 部分 CNN 卷积核大小
            deep_pooling_sizes: Deep 部分 POOLING 大小
            concat_layers_units: Concat 部分隐含层节点个数
            concat_layers_dropout: Concat 部分 DROPOUT 比例
            weighted_loss: 是否使用待权重的损失
            class_weights: 类型权重 (多分类为权重列表，二分类为正样本权重)
            learning_rate: 学习率
            optimizer: 优化器
        """

        LOGGER.info('Building wide & deep char CNN network ...')

        self.kernel_initializer = tf.truncated_normal_initializer(stddev=0.05)
        self.is_training = tf.placeholder(tf.bool, [], name='is_training')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.wide_layers_units = [32, 64, 32] if wide_layers_units is None else wide_layers_units
        self.wide_keep_prob = tf.where(self.is_training, wide_layers_dropout, 1.0, name='wide_keep_prob')

        self.deep_embedding_size = deep_embedding_size
        self.deep_num_filters = [64, 64, 64] if deep_num_filters is None else deep_num_filters
        self.deep_kernel_sizes = [3, 3, 3] if deep_kernel_sizes is None else deep_kernel_sizes
        self.deep_pooling_sizes = [2, 2, 2] if deep_pooling_sizes is None else deep_pooling_sizes

        self.concat_layers_units = [512, 256, 128] if concat_layers_units is None else concat_layers_units
        self.concat_keep_prob = tf.where(self.is_training, concat_layers_dropout, 1.0, name='concat_keep_prob')

        self.learning_rate = learning_rate

        self.wide_features = tf.placeholder(tf.float32, [None, wide_features_len], name='wide_features')
        self.deep_features = tf.placeholder(tf.int32, [None, deep_features_len], name='deep_features')

        self.sensitive_type_label = sensitive_type_label

        self.weighted_loss = weighted_loss
        if sensitive_type_label:
            if class_weights:
                self.class_weights = tf.constant([class_weights], name='class_weights')
            else:
                ratio = 40
                weight_base = 1 / (1 + ratio * (num_class - 1))
                class_weights_ = [ratio * weight_base] * (num_class - 1)
                class_weights_.append(weight_base)
                self.class_weights = tf.constant([class_weights_], name='class_weights')
        else:
            self.class_weights = class_weights

        if sensitive_type_label:
            self.labels = tf.placeholder(tf.int32, [None, num_class], name='labels')
        else:
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')

        # Wide Layers
        with tf.name_scope('wide_layers'):
            wide_last_layer = self.wide_features
            wide_fc_layers = []

            for index, num_units in enumerate(self.wide_layers_units):
                with tf.name_scope('dense_with_dropout_{index}'.format(index=index+1)):
                    wide_fc_layer = tf.layers.dense(
                        wide_last_layer,
                        num_units,
                        activation=tf.nn.relu,
                        kernel_initializer=self.kernel_initializer,
                        name='wide_dense_{index}'.format(index=index+1))
                    wide_fc_layers.append(wide_fc_layer)

                    wide_dropout_layer = tf.nn.dropout(wide_fc_layer, self.wide_keep_prob,
                                                       name='wide_dropout_{index}'.format(index=index+1))

                    wide_last_layer = wide_dropout_layer

        # Deep Layers
        with tf.name_scope('deep_layers'):
            with tf.name_scope('embedding'):
                init_embeddings = tf.random_uniform([vocabulary_size, self.deep_embedding_size])
                self.embeddings = tf.get_variable('embeddings', initializer=init_embeddings)
                self.deep_embeddings = tf.nn.embedding_lookup(self.embeddings, self.deep_features)
                self.deep_embeddings = tf.expand_dims(self.deep_embeddings, -1)

            deep_last_layer = self.deep_embeddings
            deep_layer_len = deep_features_len
            deep_conv_max_pooling_layers = []

            for index, (deep_num_filter, kernel_size_1, pooling_size) in \
                    enumerate(zip(self.deep_num_filters, self.deep_kernel_sizes, self.deep_pooling_sizes)):
                with tf.name_scope('conv_max_pooling_{index}'.format(index=index+1)):
                    kernel_size_2 = self.deep_embedding_size \
                        if index == 0 else deep_num_filter

                    deep_conv_layer = tf.layers.conv2d(
                        deep_last_layer,
                        filters=deep_num_filter,
                        kernel_size=[kernel_size_1, kernel_size_2],
                        strides=(1, 1),
                        padding='VALID',
                        activation=tf.nn.relu,
                        name='deep_conv_2d_{index}'.format(index=index+1))
                    deep_conv_max_pooling_layers.append(deep_conv_layer)
                    deep_layer_len = deep_layer_len - kernel_size_1 + 1

                    deep_max_pooling_layer = tf.layers.max_pooling2d(
                        deep_conv_layer,
                        pool_size=[pooling_size, 1],
                        strides=[pooling_size, 1],
                        padding='VALID',
                        name='deep_max_pooling_2d_{index}'.format(index=index+1))
                    deep_max_pooling_layer = tf.transpose(deep_max_pooling_layer, [0, 1, 3, 2])
                    deep_conv_max_pooling_layers.append(deep_max_pooling_layer)
                    deep_layer_len = deep_layer_len // 2

                    deep_last_layer = deep_max_pooling_layer

            deep_flatten_layer = tf.reshape(deep_last_layer, [-1, deep_layer_len * self.deep_num_filters[-1]])

        # Concat Layers
        with tf.name_scope('concat'):
            wide_and_deep_concat_layer = tf.concat([wide_last_layer, deep_flatten_layer], 1)

            concat_last_layer = wide_and_deep_concat_layer
            concat_fc_layers = []

            for index, num_units in enumerate(self.concat_layers_units):
                with tf.name_scope('concat_dense_with_dropout_{index}'.format(index=index+1)):
                    concat_fc_layer = tf.layers.dense(
                        concat_last_layer,
                        num_units,
                        activation=tf.nn.relu,
                        kernel_initializer=self.kernel_initializer,
                        name='concat_dense_{index}'.format(index=index+1))
                    concat_fc_layers.append(concat_fc_layer)

                    concat_dropout_layer = tf.nn.dropout(concat_fc_layer, self.concat_keep_prob,
                                                         name='concat_dropout_{index}'.format(index=index+1))

                    concat_last_layer = concat_dropout_layer

            self.logits = tf.layers.dense(
                concat_last_layer,
                num_class if self.sensitive_type_label else 1,
                activation=None,
                kernel_initializer=self.kernel_initializer,
                name='concat_output')

        # Loss
        with tf.name_scope('loss'):
            if self.sensitive_type_label:
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.logits, labels=self.labels, name='softmax_cross_entropy_with_logits')

                if self.weighted_loss:
                    class_weights_batch = tf.transpose(
                        tf.matmul(tf.cast(self.labels, tf.float32), tf.transpose(self.class_weights)),
                        name='class_weights_batch')
                    weighted_cross_entropy = tf.multiply(class_weights_batch, self.cross_entropy,
                                                         name='weighted_cross_entropy')
                    self.loss = tf.reduce_mean(weighted_cross_entropy)
                else:
                    self.loss = tf.reduce_mean(self.cross_entropy)
            else:
                if self.weighted_loss:
                    self.cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
                        logits=self.logits, targets=self.labels, pos_weight=self.class_weights,
                        name='weighted_cross_entropy_with_logits')
                else:
                    self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.logits, labels=self.labels, name='sigmoid_cross_entropy_with_logits')

                self.loss = tf.reduce_mean(self.cross_entropy)

            self.optimizer = optimizer(learning_rate=self.learning_rate).minimize(
                self.loss, global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)

        # Stats
        with tf.name_scope('stats'):
            if self.sensitive_type_label:
                self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32, name='predictions')
                self.stats_labels = tf.argmax(self.labels, -1, output_type=tf.int32, name='labels')
                correct_predictions = tf.equal(self.predictions, self.stats_labels)
            else:
                self.predictions = tf.nn.sigmoid(self.logits, name='predictions')
                self.stats_labels = tf.cast(self.labels, tf.float32, name='labels')
                correct_predictions = tf.equal(tf.round(self.predictions), self.stats_labels)

            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

        tf.summary.scalar('accuracy', self.accuracy)

        self.merged_summaries = tf.summary.merge_all()

        # For predict
        self.graph = None
        self.sess = None

    def train(
            self, train_wide_features, train_deep_features,
            train_is_sensitive_labels, train_sensitive_type_labels,
            split_valid_dataset=False, valid_dataset_ratio=0.2,
            valid_wide_features=None, valid_deep_features=None,
            valid_is_sensitive_labels=None, valid_sensitive_type_labels=None,
            batch_size=128, num_epochs=10,
            log_steps=100, valid_steps=1000,
            checkpoint_dir='../../data/wide_deep_char_cnn',
            checkpoint_filename_prefix='wide_deep_char_cnn',
            summaries_dir: str=None):
        """ 训练模型

        Args:
            train_wide_features: Wide 特征训练集
            train_deep_features: Deep 特征训练集
            train_is_sensitive_labels: 是否敏感标签训练集
            train_sensitive_type_labels: 敏感类型标签训练集
            split_valid_dataset: 是否划分验证集
            valid_dataset_ratio: 验证集比例
            valid_wide_features: Wide 特征验证集
            valid_deep_features: Deep 特征验证集
            valid_is_sensitive_labels: 是否敏感标签验证集
            valid_sensitive_type_labels: 敏感类型标签验证集
            batch_size: 批大小
            num_epochs: 轮数
            log_steps: 日志打印步数
            valid_steps: 验证步数
            checkpoint_dir: 检查点文件目录
            checkpoint_filename_prefix: 检查点文件名前缀
            summaries_dir: Summary 目录

        Returns:

        """

        LOGGER.info('Training model ...')

        LOGGER.info('Cleaning summaries files ...')
        summaries_abs_dir = os.path.join(os.path.dirname(__file__), 'summaries') \
            if summaries_dir is None else summaries_dir

        if tf.gfile.Exists(summaries_abs_dir):
            tf.gfile.DeleteRecursively(summaries_abs_dir)
        tf.gfile.MakeDirs(summaries_abs_dir)

        if split_valid_dataset:
            LOGGER.info('Splitting train and test dataset ...')
            train_wide_features_, valid_wide_features_, \
            train_deep_features_, valid_deep_features_, \
            train_is_sensitive_labels_, valid_is_sensitive_labels_, \
            train_sensitive_type_labels_, valid_sensitive_type_labels_ = \
                train_test_split(train_wide_features, train_deep_features,
                                 train_is_sensitive_labels, train_sensitive_type_labels,
                                 test_size=valid_dataset_ratio)
        else:
            train_wide_features_ = train_wide_features
            valid_wide_features_ = valid_wide_features
            train_deep_features_ = train_deep_features
            valid_deep_features_ = valid_deep_features
            train_is_sensitive_labels_ = train_is_sensitive_labels
            valid_is_sensitive_labels_ = valid_is_sensitive_labels
            train_sensitive_type_labels_ = train_sensitive_type_labels
            valid_sensitive_type_labels_ = valid_sensitive_type_labels

        with tf.Session() as sess:
            summaries_writer = tf.summary.FileWriter(os.path.join(summaries_abs_dir, 'train'), sess.graph)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())

            train_batches = batch_iter(train_wide_features_, train_deep_features_, train_is_sensitive_labels_,
                                       train_sensitive_type_labels_, batch_size, num_epochs)
            max_accuracy = 0.0

            for train_wide_features_batch, train_deep_features_batch, \
                train_is_sensitive_labels_batch, train_sensitive_type_labels_batch in train_batches:
                train_feed_dict = {
                    self.wide_features: train_wide_features_batch,
                    self.deep_features: train_deep_features_batch,
                    self.labels: train_sensitive_type_labels_batch if \
                        self.sensitive_type_label else train_is_sensitive_labels_batch,
                    self.is_training: True
                }

                _, step, loss, accuracy, merged_summaries = \
                    sess.run([self.optimizer, self.global_step, self.loss, self.accuracy, self.merged_summaries],
                             feed_dict=train_feed_dict)
                summaries_writer.add_summary(merged_summaries, step)

                if step % log_steps == 0:
                    LOGGER.info('Step {step}: loss = {loss}, accuracy = {accuracy}'
                                .format(step=step, loss=loss, accuracy=accuracy))

                if step % valid_steps == 0:
                    valid_batches = batch_iter(valid_wide_features_, valid_deep_features_, valid_is_sensitive_labels_,
                                               valid_sensitive_type_labels_, batch_size, 1)
                    valid_predictions = []
                    valid_labels = []
                    sum_accuracy, cnt = 0, 0

                    for valid_wide_features_batch, valid_deep_features_batch, \
                        valid_is_sensitive_labels_batch, valid_sensitive_type_labels_batch in valid_batches:
                        valid_feed_dict = {
                            self.wide_features: valid_wide_features_batch,
                            self.deep_features: valid_deep_features_batch,
                            self.labels: valid_sensitive_type_labels_batch if \
                                self.sensitive_type_label else valid_is_sensitive_labels_batch,
                            self.is_training: False
                        }

                        predictions, stats_labels, accuracy = sess.run(
                            [self.predictions, self.stats_labels, self.accuracy], feed_dict=valid_feed_dict)

                        valid_predictions.extend(predictions)
                        valid_labels.extend(stats_labels)

                        sum_accuracy += accuracy
                        cnt += 1

                    valid_accuracy = sum_accuracy / cnt

                    LOGGER.info('Validation accuracy = {valid_accuracy}'.format(valid_accuracy=valid_accuracy))

                    self.performance(valid_predictions, valid_labels)

                    if valid_accuracy >= max_accuracy:
                        max_accuracy = valid_accuracy
                        model_file_path = '{checkpoint_dir}/{checkpoint_filename_prefix}_{timestamp}.ckpt'.format(
                            checkpoint_dir=checkpoint_dir, checkpoint_filename_prefix=checkpoint_filename_prefix,
                            timestamp=int(time.time()))
                        saver.save(sess, model_file_path, global_step=step)
                        LOGGER.info('Model is saved to {model_file_path}'.format(model_file_path=model_file_path))

                summaries_writer.close()

    def test(
            self, wide_features, deep_features,
            is_sensitive_labels, sensitive_type_labels,
            batch_size=128,
            checkpoint_dir='../../data/wide_deep_char_cnn'):
        """

        Args:
            wide_features: Wide 特征
            deep_features: Deep 特征
            is_sensitive_labels: 是否敏感标签
            sensitive_type_labels: 敏感类型标签
            batch_size: 批大小
            checkpoint_dir: 检查点文件路径

        Returns:

        """

        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        graph = tf.Graph()

        with graph.as_default():
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph('{checkpoint_file}.meta'.format(checkpoint_file=checkpoint_file))
                saver.restore(sess, checkpoint_file)

                test_wide_features = graph.get_operation_by_name('wide_features').outputs[0]
                test_deep_features = graph.get_operation_by_name('deep_features').outputs[0]
                labels = graph.get_operation_by_name('labels').outputs[0]
                is_training = graph.get_operation_by_name('is_training').outputs[0]
                predictions = graph.get_operation_by_name('stats/predictions').outputs[0]
                stats_labels = graph.get_operation_by_name('stats/labels').outputs[0]
                accuracy = graph.get_operation_by_name('stats/accuracy').outputs[0]

                test_batches = batch_iter(wide_features, deep_features,
                                          is_sensitive_labels, sensitive_type_labels, batch_size, 1)

                test_predictions = []
                test_labels = []
                sum_accuracy, cnt = 0, 0

                for test_wide_features_batch, test_deep_features_batch, \
                    test_is_sensitive_type_labels_batch, test_sensitive_type_labels_batch in test_batches:
                    test_feed_dict = {
                        test_wide_features: test_wide_features_batch,
                        test_deep_features: test_deep_features_batch,
                        labels: test_sensitive_type_labels_batch if \
                            self.sensitive_type_label else test_is_sensitive_type_labels_batch,
                        is_training: False
                    }

                    predictions_out, labels_out, accuracy_out = sess.run(
                        [predictions, stats_labels, accuracy], feed_dict=test_feed_dict)

                    test_predictions.extend(predictions_out)
                    test_labels.extend(labels_out)

                    sum_accuracy += accuracy_out
                    cnt += 1

                test_accuracy = sum_accuracy / cnt
                LOGGER.info('Test accuracy: {test_accuracy}'.format(test_accuracy=test_accuracy))

                self.performance(test_predictions, test_labels)

    def restore(self, checkpoint_dir: str):
        """ 恢复模型

        Args:
            checkpoint_dir: 检查点文件路径

        Returns:

        """

        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.sess = tf.Session()
            saver = tf.train.import_meta_graph('{checkpoint_file}.meta'.format(checkpoint_file=checkpoint_file))
            saver.restore(self.sess, checkpoint_file)

    def predict(self, wide_features, deep_features):
        """ 预测

        Args:
            wide_features: Wide 特征
            deep_features: Deep 特征

        Returns:
            预测结果

        """

        predict_wide_features = self.graph.get_operation_by_name('wide_features').outputs[0]
        predict_deep_features = self.graph.get_operation_by_name('deep_features').outputs[0]
        is_training = self.graph.get_operation_by_name('is_training').outputs[0]
        predictions = self.graph.get_operation_by_name('stats/predictions').outputs[0]

        predict_feed_dict = {
            predict_wide_features: np.array([wide_features]),
            predict_deep_features: np.array([deep_features]),
            is_training: False
        }

        predictions_out = self.sess.run([predictions], feed_dict=predict_feed_dict)

        return np.asscalar(predictions_out[0])

    def performance(self, predictions, labels):
        """ 模型性能

        Args:
            predictions: 预测结果
            labels: 真实结果

        Returns:

        """

        if self.sensitive_type_label:
            LOGGER.info('Classification Report: ')
            p, r, f1, s = precision_recall_fscore_support(y_true=labels, y_pred=predictions)
            report = pd.DataFrame({'class': np.append(COLUMN_SENSITIVE_TYPE_LE.classes_, 'AVG / TOTAL'),
                                   'precision': np.append(p, np.average(p)),
                                   'recall': np.append(r, np.average(r)),
                                   'f1-score': np.append(f1, np.average(f1)),
                                   'support': np.append(s, np.sum(s))})
            tp.dataframe(report)

            LOGGER.info('Confusion Matrix: ')
            confusion_matrix_ = confusion_matrix(y_true=labels, y_pred=predictions)
            confusion_matrix_ = confusion_matrix_.astype(str)
            confusion_matrix_ = np.insert(confusion_matrix_, 0, COLUMN_SENSITIVE_TYPE_LE.classes_, 1)
            headers = ['']
            headers.extend(list(COLUMN_SENSITIVE_TYPE_LE.classes_))
            tp.table(confusion_matrix_, headers=headers)
        else:
            pass
