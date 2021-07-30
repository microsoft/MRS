"""
    Code for computing metrics and logging training and validation
    @author: Budhaditya Deb
"""
from collections import OrderedDict
from typing import Any
import time
import numpy as np
import os
import sys
#import tensorboardX
import torch
import logging


class Metric(object):
    def __init__(self, lm_alpha, metric_type='min'):
        self.metric_type = metric_type
        self.num_steps_in_cache = 0.0
        self.running_sum = 0
        self.running_avg = 0
        self.prev_value = 0
        self.current_value = 0
        self.best_value = 0
        if self.metric_type == 'min':
            self.best_value = np.inf
            self.current_value = np.inf
            self.prev_value = np.inf
        if self.metric_type == 'max':
            self.best_value = 0
            self.current_value = 0
            self.prev_value = 0
        # set of hyperparameters to keep track for model selection
        self.best_lm_alpha = lm_alpha
        self.best_epoch = 0
        self.best_minibatch_number = 0

    def _update_best(self, value, epoch, minibatch_number, lm_alpha):
        self.best_value = value
        self.best_minibatch_number = minibatch_number
        self.best_epoch = epoch
        self.best_lm_alpha = lm_alpha

    def update_running_avg(self, value):
        self.running_sum += value
        self.num_steps_in_cache += 1
        self.running_avg = self.running_sum / self.num_steps_in_cache

    def get_avg_value(self):
        return self.running_avg

    def reset(self):
        self.num_steps_in_cache = 0.0
        self.running_sum = 0

    def update_metric(self, value, epoch, minibatch_number, lm_alpha):
        self.prev_value = self.current_value
        self.current_value = value
        if self.metric_type == 'min':
            if value < self.best_value:
                self._update_best(value, epoch, minibatch_number, lm_alpha)

        if self.metric_type == 'max':
            if value > self.best_value:
                self._update_best(value, epoch, minibatch_number, lm_alpha)


class MetricsLogger(object):
    # Simple class to log metrics during training run.
    def __init__(self, params, dataset_len, lm_alpha, valid_log_name=None):
        self.params = params
        self.dataset_len = dataset_len
        self.num_steps_per_epoch = dataset_len / float(params.batch_size)
        self.train_start_time = time.time()
        self.batch_retrieve_factor = 1
        self.elapsed_time = 0
        self.epoch_fraction = 0
        self.epochs = 0
        self.est_epoch_time = 0
        self.minibatch_number = 0
        self.recent_items = 0
        self.seqs_per_batch = 0
        self.subbatch_count = 0
        self.time_per_step = 0
        self.total_items = 0
        self.train_metrics = {}
        self.train_steps = 0
        self.valid_metrics = {}
        self.elapsed_epoch = 0.0
        self.validation_steps = 0
        self.total_valid_items = 0

        # add standard metrics
        self.train_metrics = OrderedDict()
        self.valid_metrics = OrderedDict()
        self.train_metrics['loss'] = Metric(lm_alpha, metric_type='min')
        self.train_metrics['precision'] = Metric(lm_alpha, metric_type='max')

        self.valid_metrics['loss'] = Metric(lm_alpha, metric_type='min')
        self.valid_metrics['macro_mrr'] = Metric(lm_alpha, metric_type='max')
        self.valid_metrics['mrr_f1'] = Metric(lm_alpha, metric_type='max')
        self.valid_metrics['mrr'] = Metric(lm_alpha, metric_type='max')
        self.valid_metrics['p_at_k'] = Metric(lm_alpha, metric_type='max')
        self.valid_metrics['precision'] = Metric(lm_alpha, metric_type='max')
        self.valid_metrics['sbleu'] = Metric(lm_alpha, metric_type='max')

        # add rouge ensemble metrics
        self.valid_metrics['Sentence_ROUGE_uniform_f'] = Metric(lm_alpha, metric_type='max')
        self.valid_metrics['Sentence_ROUGE_weight_f'] = Metric(lm_alpha, metric_type='max')

        self.train_log_file_name = os.path.join(params.model_output_dir, 'train.log.%d.%d.txt' % (params.node_rank, params.local_rank))
        if valid_log_name is None:
            self.valid_log_file_name = os.path.join(params.model_output_dir, 'valid.log.%d.%d.txt' % (params.node_rank, params.local_rank))
        else:
            self.valid_log_file_name = os.path.join(params.model_output_dir, valid_log_name)

    def set_dataset_len(self, dataset_len):
        self.dataset_len = dataset_len
        self.num_steps_per_epoch = self.dataset_len / float(self.params.batch_size)

    def add_metric(self, metric_name, lm_alpha, metric_type, for_train=True, for_validation=True):
        # add named metrics for different models if not covered by the current list
        if for_train is True:
            self.train_metrics[metric_name] = Metric(lm_alpha, metric_type=metric_type)

        if for_validation is True:
            self.valid_metrics[metric_name] = Metric(lm_alpha, metric_type=metric_type)

    def update_validation_metrics(self, epoch, minibatch_number, lm_alpha, metrics_dict):
        # use this for updating validation metrics
        for key, value in metrics_dict.items():
            self.valid_metrics[key].update_metric(value, epoch, minibatch_number, lm_alpha)

    def update_train_run_stats(self, batch_size, loss_dict, epoch, minibatch_number):
        # this keeps a running average of training time metrics
        self.train_steps += 1
        self.total_items += batch_size
        self.epochs = epoch
        self.minibatch_number = minibatch_number
        for key, value in loss_dict.items():
            self.train_metrics[key].update_running_avg(value.item())
        # self.update_timing_variables()
        current_time = time.time()
        self.seqs_per_batch = float(self.total_items) / self.train_steps
        self.time_per_step = (current_time - self.train_start_time) / self.train_steps
        self.est_epoch_time = self.time_per_step * self.num_steps_per_epoch
        self.est_epoch_time = '{}h:{}m'.format(int(self.est_epoch_time/3600), int(self.est_epoch_time/60) % 60)
        self.elapsed_time = int(current_time - self.train_start_time)
        self.elapsed_time = '{}h:{}m:{}s'.format(int(self.elapsed_time/3600), int(self.elapsed_time/60) % 60, (self.elapsed_time % 60))
        # TODO: Add brackets here to make the computation unambiguous
        self.elapsed_epoch = float(self.train_steps) / self.num_steps_per_epoch * 100 - 100 * (self.epochs)

    def update_validation_run_stats(self, batch_size, loss_dict, epoch, minibatch_number):
        # Similar to training run statistics, this is used for going through multiple minibatches of validation data to compute stats
        # this keeps a running average of validation time metrics
        self.validation_steps += 1
        self.total_valid_items += batch_size
        self.epochs = epoch
        self.minibatch_number = minibatch_number
        # TODO: Why do we have this here? .. What's the difference between this and the update_validation_metrics function?
        for key, value in loss_dict.items():
            self.valid_metrics[key].update_running_avg(value.item())

    def set_validation_running_stats(self, params, epoch, minibatch_number):
        # we call this after the training run completes to update the validation stats and keep track of the best validation scores
        updated_valid_metrics = {}
        for key, metric in self.valid_metrics.items():
            updated_valid_metrics[key] = metric.get_avg_value()
        self.update_validation_metrics(epoch, minibatch_number, params.lm_alpha, updated_valid_metrics)

    def print_train_metrics(self, force=False, reset=True):
        # prints training metrics to stdout and saves in the log file
        if not os.path.exists(self.train_log_file_name):
            # Add exists_ok=True for cascaded runs in AML
            os.makedirs(os.path.dirname(self.train_log_file_name), exist_ok=True)
            with open(self.train_log_file_name, 'w', encoding='utf-8') as f_out:
                log_str = "Time Epoch MB_Steps Elapsed_epoch step_time epoch_time Num_Seqs"
                for key in self.train_metrics.keys():
                    log_str += " %s" % key
                f_out.write("%s\n" % log_str)

        if self.train_steps % self.params.steps_per_print == 0 or force is True:
            print_str = "%s Epoch=%d MB=%d steps=%d elapsed_epoch=%f step_time=%f epoch_time=%s seqs=%d" % (
                self.elapsed_time, self.epochs, self.minibatch_number,
                self.train_steps, self.elapsed_epoch, self.time_per_step, self.est_epoch_time,
                self.seqs_per_batch)
            log_str = "%s %d %d %f %f %s %d" % (
                self.elapsed_time, self.epochs,
                self.train_steps, self.elapsed_epoch, self.time_per_step,
                self.est_epoch_time, self.seqs_per_batch)

            with open(self.train_log_file_name, 'a', encoding='utf8') as f_out:
                for key, metric in self.train_metrics.items():
                    val = metric.get_avg_value()
                    print_str = print_str + " %s=%f" % (key, val)
                    log_str += " %f" % val
                    if reset is True:
                        metric.reset()
                f_out.write("%s\n" % log_str)

            print("SystemLog: Device=%s, %s" % (self.params.device, print_str))
        sys.stdout.flush()

    def print_validation_metrics(self):
        # print validation metrics to stdout and saves in the log file
        if not os.path.exists(self.valid_log_file_name):
            with open(self.valid_log_file_name, 'w', encoding='utf-8') as f_out:
                log_str = "Epoch"
                for key in self.valid_metrics.keys():
                    log_str += " %s epoch lm_alpha" % key
                f_out.write("%s\n" % log_str)

        with open(self.valid_log_file_name, 'a', encoding='utf8') as f_out:
            log_str = "%d " % self.epochs
            for key, metric in self.valid_metrics.items():
                print("SystemLog: Epoch=%d, %s: Prev:=%f, Current=%f, Best=%f, Epoch=%f, Minibatch=%f lm_alpha=%lf" % (
                    self.epochs,
                    key, metric.prev_value, metric.current_value, metric.best_value,
                    metric.best_epoch, metric.best_minibatch_number, metric.best_lm_alpha))
                log_str += "%f %d %f " % (metric.best_value, metric.best_epoch, metric.best_lm_alpha)
            f_out.write("%s\n" % log_str)

    def get_latest_valid_metric(self):
        # get validation metric of most recently validation
        metric_dict = {}
        for key, metric in self.valid_metrics.items():
            # TODO: Don't access the private members like this ... write a function instead
            metric_dict[key] = metric.current_value
        return metric_dict

    def get_best_valid_metric(self):
        # get validation metric of most recently validation
        metric_dict = {}
        for key, metric in self.valid_metrics.items():
            # TODO: Don't access the private members like this ... write a function instead
            metric_dict[key] = metric.best_value
        return metric_dict