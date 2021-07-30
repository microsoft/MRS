"""
    SR Model Class with functions for training, validation and export
    Copyright: Microsoft Search, Assistance and Intelligence Team, 2020

    - The main inputs required for the model are:
        - Training and Validation data directory: directories with one or many text files in TSV format
        - Vocab Directory
        - Response set directory
        - self.params.json file which lists all the parameters required in the scrint
    - To feed the model we use the MRDataset class in util.py.
    - Model is saved in checkpoints and model with best validation loss is retained during training

    @author: Budhaditya Deb
"""

import time
import os
import json
import sys
import pdb
import gc
import copy
import shutil
from itertools import islice
from operator import itemgetter

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
import importlib

from retrieval_rs.models.common_model_lib.model_factory import get_module_class_from_factory
from retrieval_rs.models.matching.data_processing import create_dataloaders, CollateMRSequence, get_dataset
from retrieval_rs.models.matching.utils import validation_loop
from retrieval_rs.models.matching.utils import set_devices_ids_for_DDP

from retrieval_rs.constants.file_names import RESPONSE_MAPPING_FILE_NAME, RESPONSE_SET_FILE_NAME, OUTPUT_RESPONSES_FILE_NAME
from retrieval_rs.models.common.tokenization import Vocab, Tokenizer
from retrieval_rs.models.common.utils import save_model, load_checkpoint, get_file_from_dir
from retrieval_rs.models.common.utils import get_tokenizer, get_optimizer
from retrieval_rs.models.common.metrics import MetricsLogger
from retrieval_rs.models.common.model_params import  build_argparser, print_parameters, save_parameters
from retrieval_rs.models.pytorch_transformers.optimization import warmup_linear_decay_exp


IS_ONNX_PRESENT = False
try:
    import onnxruntime as rt
    IS_ONNX_PRESENT = True
# TODO: Catch the exact exception
except:
    print("SystemLog: Onnxruntime not found")

IS_NLTK_PRESENT = False
try:
    # TODO: There are better ways to set flags like these without actually importing the module (use importlib)
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
    IS_NLTK_PRESENT = True
# TODO: Catch the exact exception
except Exception as ex:
    print("SystemLog: nltk not found %s" % ex)

IS_APEX_PRESENT = False
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as apex_DDP
    IS_APEX_PRESENT = True
except ImportError as e:
    print("SystemLog: Apex not found")

class SRModel():
    def __init__(self, params):
        self.params = params
        self.create_dataloaders = create_dataloaders
        self.get_dataset = get_dataset
        self.validation_loop = validation_loop

    def init_dataloaders(self):
        if self.params.run_mode == "train":
            self.mr_dataset_train, self.mr_dataloader_train = self.create_dataloaders(self.params, run_mode='train')
        self.mr_dataset_valid, self.mr_dataloader_valid = self.create_dataloaders(self.params, run_mode='valid')
        self.mr_dataset_gmr, self.mr_dataloader_gmr = self.create_dataloaders(self.params, run_mode='gmr')

    def init_train(self):
        if self.params.distributed_data_parallel and self.params.use_cuda:
            set_devices_ids_for_DDP(self.params)
        else:
            self.params.local_rank = 0
            self.params.node_rank = 0
        # ------------ Create Model --------------
        self.train_model = self.initialize_model()
        print("SystemLog: Setting train model to device", self.params.device)
        self.train_model = self.train_model.to(self.params.device)
        if self.params.local_rank == 0 and self.params.node_rank == 0:
            self.train_model.print_model_dict()
        # -------- Create Optimizer and Scheduler ----------
        self.train_model, self.optimizer = get_optimizer(self.train_model, self.params)
        if self.params.fp16:
            if self.params.amp_opt_lvl == 'O2':
                self.train_model.half()
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.8, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        # -------- Create Datasets and Metrics ----------
        self.tokenizer = get_tokenizer(self.params)
        self.init_dataloaders()
        valid_log_name = "valid.log.%d.%d.txt"%(self.params.node_rank, self.params.local_rank)
        self.metrics_logger = self.create_metrics_logger(len(self.mr_dataset_train), valid_log_name)
        # ---------- load checkpoint -----------
        if self.params.model_input_dir:
            old_params, self.train_model, _, _, _ = load_checkpoint(self.params, self.train_model, optimizer=self.optimizer, metrics_logger=self.metrics_logger)
        else:
            print("SystemLog: model_input_dir is not provided -- no model checkpoint will be loaded")
        # ---------------- Create distributed data readers --------------------
        if self.params.distributed_data_parallel and self.params.use_cuda:
            self.setup_distributed_data_parallel()
            # -------------- unwrap train mode if ddp for other uses --------------
            self.train_model_inst = self.train_model.module
        else:
            self.train_model_inst = self.train_model
        # ---------- save self.params at the beginning -----------------
        if self.params.local_rank == 0 and self.params.node_rank == 0:
            print_parameters(self.params)
            save_parameters(self.params, mode='train')
        #----------------set bookkeeping variables----------------
        self.summary_writer = None

    def train_step(self, i_batch, sample_batched, epoch):
        # --------take train step and update model parameters--------
        output = self.train_model(sample_batched)
        loss_dict = self.train_model_inst.loss(output)
        # --------bookkeeping and updating stats--------
        if self.params.distributed_data_parallel and dist.get_rank() == 0 and self.summary_writer is not None:
            for k, v in loss_dict:
                self.summary_writer.add_scalar('Train/%s', k, v.item(), self.train_steps)
        self.metrics_logger.update_train_run_stats(output[0].shape[0], loss_dict, epoch, i_batch+1)
        if self.params.local_rank == 0 and self.params.node_rank == 0:
            self.metrics_logger.print_train_metrics()
        # ------------ Gadient accumulation step ------------
        if self.params.gradient_accumulation_steps > 1:
            loss_dict['loss'] /= self.params.gradient_accumulation_steps
        # ------------ Back-Propagation step ------------
        if self.params.fp16:
            with amp.scale_loss(loss_dict['loss'], self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_dict['loss'].backward()
        # ------------ Model self.params update and Scheduler Step ------------
        if i_batch % self.params.gradient_accumulation_steps == 0:
            if self.params.fp16:
                lr_this_step = self.params.learning_rate * warmup_linear_decay_exp(self.global_step, self.params.decay_rate, self.params.decay_step, self.params.total_training_steps, self.params.warmup_proportion)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            # ------------ Model params update ------------
            if self.params.clip_grad_norm > 0:
                if self.params.fp16:
                    clip_grad_norm_(amp.master_params(self.optimizer), self.params.clip_grad_norm)
            self.optimizer.step()
            self.train_model.zero_grad()
            self.optimizer.zero_grad()
            self.global_step += 1
        return output, loss_dict

    def update_metrics_and_save_step(self, epoch, i_batch, metrics_logger):
        # --------- Update the tracked metric ---------
        best_metrics_dict = metrics_logger.get_best_valid_metric()
        best_tracked_metric_new = best_metrics_dict[self.tracked_metric]
        if self.params.distributed_data_parallel and dist.get_rank() == 0 and self.summary_writer is not None:
            for k, v in best_metrics_dict:
                self.summary_writer.add_scalar('Evaluation/%s', k, v, self.train_steps)

        if self.params.local_rank == 0 and self.params.node_rank == 0:
            print("SystemLog: Node:%d, %s: Prev best = %f, current best = %f"%(self.params.local_rank, self.tracked_metric, self.best_tracked_metric, best_tracked_metric_new))
        # --------- Update model.best.pt if metrics improve ---------
        if best_tracked_metric_new > self.best_tracked_metric:
            self.best_tracked_metric = best_tracked_metric_new
            if self.params.local_rank == 0 and self.params.node_rank == 0:
                print('SystemLog: Node=%d, Updating best model %d'%(self.params.local_rank, epoch))
                save_model(self.params, self.train_model_inst, self.optimizer, metrics_logger, epoch, i_batch+1, update_best_model=True)
            self.num_validations_since_last_metric_improvement = 0
        else:
            if self.params.architecture == "mcvae_model":
                self.num_validations_since_last_metric_improvement += 1
                self.train_model_inst.cvae_loss.decay_update()
        # -------- Save a checkpoint --------
        if self.params.local_rank == 0 and self.params.node_rank == 0:
            if  self.params.save_freq != -1 and self.train_steps % self.params.save_freq == 0:
                save_model(self.params, self.train_model_inst, self.optimizer, metrics_logger, epoch, i_batch+1, update_best_model=False)

    def train(self):
        self.init_train()
        # ---------- print some stats at the beginning -----------------
        # _, best_mrr_new, best_loss_new, best_precision_new = self.validation_loop(self.params, self.train_model_inst, self.mr_dataloader_valid, self.mr_dataset_valid, self.mr_dataloader_gmr, self.tokenizer, self.metrics_logger, 0, 0)
        if self.params.local_rank == 0 and self.params.node_rank == 0:
            print('SystemLog: ---------------------------------------------------------')
            print('SystemLog: Starting training loop with %d items per epoch' %len(self.mr_dataset_train))
        #----------------reset set model variables for training ----------------
        self.start_epoch = self.metrics_logger.epochs
        self.train_model_inst.set_run_mode(run_mode="train")
        self.train_model.train()
        self.train_steps = 0
        self.global_step = 1
        self.best_tracked_metric = 0
        self.tracked_metric = "Sentence_ROUGE_weight_f"
        self.train_model.zero_grad()
        self.optimizer.zero_grad()
        self.num_validations_since_last_metric_improvement = 0
        #---------------- main training loop with validation ----------------
        for epoch in range(self.start_epoch, self.params.max_epochs):
            for i_batch, sample_batched in enumerate(self.mr_dataloader_train):
                # sample_batched has the following members in the list: message_tuple, rsponse_tuple, message, reply
                if  len(sample_batched['rsp_batch_tuple'][0]) > 1:
                    # --------take train step and update model parameters--------
                    output, loss_dict = self.train_step(i_batch, sample_batched, epoch)
                    self.train_steps += 1
                    # --------Run validation save checkpoints --------
                    if self.params.validation_freq != -1 and self.train_steps % self.params.validation_freq == 0:
                        _, best_mrr_new, best_loss_new, best_precision_new = self.validation_loop(self.params, self.train_model_inst, self.mr_dataloader_valid, self.mr_dataset_valid, self.mr_dataloader_gmr, self.tokenizer, self.metrics_logger, epoch, i_batch)
                        # --------- Update the tracked metric ---------
                        self.update_metrics_and_save_step(epoch, i_batch, self.metrics_logger)
                # ---------- Early Stopping Criteria -----------
                if self.num_validations_since_last_metric_improvement == self.params.max_metrics_plateau_for_early_stopping:
                    print("SystemLog: Early stopping due to no metrics improvement in last %s validation runs"%self.num_validations_since_last_metric_improvement)
                    return

    def init_validation(self):
        self.params.local_rank = 0
        self.params.node_rank = 0
        print_parameters(self.params)
        self.tokenizer = get_tokenizer(self.params)
        self.params.vocab_size = self.tokenizer.get_vocab_size()
        if self.params.local_rank == 0 and self.params.node_rank == 0:
            print("SystemLog: Vocab size used in model %d" % self.params.vocab_size)
        save_parameters(self.params, mode='compute_metrics')
        # ----------------Create Dataset objects and Dataloaders----------------
        self.init_dataloaders()

        self.metrics_logger = self.create_metrics_logger(len(self.mr_dataset_valid))
        # --------------- Initialize model ---------------
        self.train_model = self.initialize_model()
        self.train_model_inst = self.train_model

    def compute_validation_metrics(self):
        """
            Function to evaluate metrics on GMR and validation set
        """
        with torch.no_grad():
            self.init_validation()
            if self.params.model_input_dir:
                """
                    model_input_directory is not required when loading from tnlr (Bert/TNLR matching model). Make sure the model_input_directory is required depending on the model type being evaluated.
                """
                old_params, self.train_model, _, _, _ = load_checkpoint(self.params, self.train_model)
            else:
                print("SystemLog: !!! ============================================================ !!!")
                print("SystemLog: !!! WARNING: model_input_dir is not provided for Compute Metrics !!!")
                print("SystemLog: model_input_directory is not usually required when loading from tnlr")
                print("SystemLog: !!! ============================================================ !!!")
            self.train_model = self.train_model.to(self.params.device)
            self.metrics_logger, _, _, _ = self.validation_loop(self.params, self.train_model, self.mr_dataloader_valid, self.mr_dataset_valid, self.mr_dataloader_gmr, self.tokenizer, self.metrics_logger, 0, 0)
            return self.metrics_logger
 
    def eval(self):
        print('SystemLog: ---------------------------------------------------------')
        print("SystemLog: Eval not implemented yet")
        print('SystemLog: ---------------------------------------------------------')

    def initialize_model(self):
        """
            Gets the architecture instance from model factory.
            Some model specific initialization if needed (currently just sets the cuda, but in future can hold other initialization code)
        """
        vocab = Vocab(None, self.params.generic_vocab_input_dir, None, min_count=-1, initial_vocab=None, _PAD="<pad>", _UNK="<unk>", _EOS="<eos>")
        vocab.init_vocab(-1)
        self.params.vocab_size = vocab.get_vocab_size()
        print("SystemLog: Vocab size used in model %d" % self.params.vocab_size)

        model_class = get_module_class_from_factory(self.params.architecture)
        model_inst = model_class(self.params)
        if self.params.use_cuda:
            model_inst = model_inst.cuda()
        print("SystemLog: Setting train model to device", self.params.device)
        return model_inst

    def create_metrics_logger(self, dataset_len, valid_log_name=None):
        # TODO: possible replace this with some factory method
        metrics_logger = MetricsLogger(self.params, dataset_len, self.params.lm_alpha, valid_log_name=valid_log_name)
        metrics_logger.add_metric("Sentence_ROUGE_uniform_f", self.params.lm_alpha, for_train=False, metric_type='max')
        metrics_logger.add_metric("Sentence_ROUGE_weight_f", self.params.lm_alpha, for_train=False, metric_type='max')
        metrics_logger.add_metric("Diversity_ROUGE_uniform_f", self.params.lm_alpha, for_train=False, metric_type='min')
        metrics_logger.add_metric("Diversity_ROUGE_weight_f", self.params.lm_alpha, for_train=False, metric_type='min')
        if 'mcvae' in self.params.architecture:
            metrics_logger.add_metric("recon_loss", self.params.lm_alpha, metric_type='min')
            metrics_logger.add_metric("kl_d", self.params.lm_alpha, metric_type='min')
            metrics_logger.add_metric("matching_loss", self.params.lm_alpha, metric_type='min')
            metrics_logger.add_metric("matching_precision", self.params.lm_alpha, metric_type='max')
        return metrics_logger

    def setup_distributed_data_parallel(self):

        """ Setup DDP wrapper along with dataloaders
        """
        print("SystemLog: Initiating distributed mode with local_rank=%d" % (self.params.local_rank))
        # ----------- Create DDP wrapper for the models -----------
        if self.params.fp16:
            if not IS_APEX_PRESENT:
                raise ValueError("SystemLog: fp16 is set to True, however no Apex installation was found .. exiting ...")
            self.train_model = apex_DDP(self.train_model)
        else:
            self.train_model = torch.nn.parallel.DistributedDataParallel(self.train_model, device_ids=[self.params.local_rank], output_device=self.params.local_rank, find_unused_parameters=True)
        # ----------- Create DDP wrapper for the dataset -----------
        print("SystemLog: Create DDP wrapper with mode with local_rank=%d" % (self.params.local_rank))
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.mr_dataset_train)
        self.mr_dataloader_train = DataLoader(self.mr_dataset_train, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=CollateMRSequence(self.params), pin_memory=False, sampler=train_sampler)
        print("SystemLog: Number of items in the train sampler in GPU %d = %d" % (self.params.local_rank, len(train_sampler)))
        self.metrics_logger.set_dataset_len(len(train_sampler))
