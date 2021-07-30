"""
    SR Model Class for Mlultilingual with functions for training, validation and export
    Copyright: Microsoft Search, Assistance and Intelligence Team, 2020

    - The main inputs required for the model are:
        - Training and Validation data directory: directories with one or many text files in TSV format as follows
            - train/language/files
            - valid/language/files
            - gmr/language/files
            - rsp_set/language/files
        - Vocab Directory
    - To feed the model we use the MRDataset class in util.py.
    - Model is saved in checkpoints and model with best validation loss is retained during training

    @author: Budhaditya Deb
"""

import time
import collections
import random
import pdb
import os
import copy
import shutil

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from retrieval_rs.constants.file_names import OUTPUT_RESPONSES_FILE_NAME
from retrieval_rs.models.common_model_lib.sr_model import SRModel
from retrieval_rs.models.matching.data_processing import get_dataset, create_dataloaders
from retrieval_rs.models.matching.utils import validation_loop
from retrieval_rs.models.common.utils import load_checkpoint, get_file_from_dir
from retrieval_rs.models.matching.data_processing import get_dataset, CollateMRSequence
from retrieval_rs.models.common.response_set import ResponseSet
from retrieval_rs.models.common_model_lib.sr_module import SRModule
from retrieval_rs.models.common.model_params import  print_parameters, save_parameters

IS_APEX_PRESENT = False
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as apex_DDP
    IS_APEX_PRESENT = True
except ImportError as e:
    print("SystemLog: Apex not found")

class SRMLTLModule(SRModule):
    """
        This is used as an super class similar to models/common_modl_lib/sr_module with some additional functions required for multi-lingual models.
        Multi-Lingual models should derive from this class if there are common elements across models.
    """
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        #self.infer_language = 'en' # default
        # MZ: change default to first test language
        self.infer_language = params.test_langs[0]
        self.train_language = 'en' # default

    def init_response_set(self):
        self.lm_scores_multi = {}
        self.response_set_multi = {}
        self.rsp_dataloaders_multi = {}
        # create temp copy for different response sets
        for lg in self.params.test_langs:
            print("SystemLog: Creating Dataloader for [%s] Language"%lg)
            rsp_dataset = get_dataset(self.params, run_mode="rsp_set", language=lg)
            self.rsp_dataloaders_multi[lg] = DataLoader(rsp_dataset, batch_size=self.params.batch_size,
                                    shuffle=False, num_workers=0, collate_fn=CollateMRSequence(self.params))
            #  For the response set class, we need to change the pointing directory to the language specific one as the initialzer is agnostic to the language.
            param_copy = copy.deepcopy(self.params)
            param_copy.rsp_input_dir = os.path.join(self.params.rsp_input_dir, lg)
            if param_copy.rsp_mapping_input_dir:
                param_copy.rsp_mapping_input_dir = os.path.join(param_copy.rsp_mapping_input_dir, lg)

            self.response_set_multi[lg] = ResponseSet(param_copy)
            self.lm_scores_multi[lg] = torch.FloatTensor(self.response_set_multi[lg].lm_scores_list)
            if self.params.use_cuda:
                self.lm_scores_multi[lg] = self.lm_scores_multi[lg].cuda()
            if self.params.fp16:
                if self.params.amp_opt_lvl == 'O2':
                    self.lm_scores_multi[lg] = self.lm_scores_multi[lg].half()

    def set_infer_language(self, language):
        self.infer_language = language

    def set_train_language(self, language):
        self.train_language = language

class SRMLTLModel(SRModel):
    def __init__(self, params):
        self.params = params
        self.create_dataloaders = create_dataloaders
        self.get_dataset = get_dataset
        self.validation_loop = validation_loop
        # self.infer_language = 'en'

    def init_dataloaders(self):
        """
            Overrides the parent data loader function with language specific ones.
            Input directories are expected in the following formats with sub directories for each language:
            - train/language/files
            - valid/language/files
            - gmr/language/files
            - rsp_set/language/files
        """
        self.total_training_size = 0
        if self.params.run_mode == "train":
            self.mr_dataset_train, self.mr_dataloader_train = {}, {}
            for language in self.params.train_langs:
                mr_dataset, mr_dataloader = create_dataloaders(self.params, "train", language)
                self.mr_dataset_train[language] = mr_dataset
                self.mr_dataloader_train[language] = mr_dataloader
                self.total_training_size += len(mr_dataset)

        self.mr_dataset_valid, self.mr_dataloader_valid = {}, {}
        self.mr_dataset_gmr, self.mr_dataloader_gmr = {}, {}
        for language in self.params.test_langs:
            mr_dataset, mr_dataloader = create_dataloaders(self.params, "valid", language)
            self.mr_dataset_valid[language] = mr_dataset
            self.mr_dataloader_valid[language] = mr_dataloader

            mr_dataset, mr_dataloader = create_dataloaders(self.params, "gmr", language)
            self.mr_dataset_gmr[language] = mr_dataset
            self.mr_dataloader_gmr[language] = mr_dataloader

    def set_infer_language(self, language):
        """
            During inference, the model needs to know which language to infer. This is used ofr reading the response set files and setting the appropriate language specific parameters.
        """
        self.infer_language = language
        self.train_model_inst.set_infer_language(language)

    def get_language(self):
        """
            Gets the language for a random batch, in proportion to the sampling proportions set in the parameters.
            When the --sample_languages_uniforly is set, overrides sample ranges and randomly selects a language
            Args:
                None
            Returns:
               batch_language from list of train languages
        """
        batch_language = None
        if self.params.sample_languages_uniformly:
            lang_index = random.randint(0, len(self.params.train_langs) - 1)
            return self.params.train_langs[lang_index]

        while batch_language is None:
            random_float = random.random()
            for key, value in self.sample_range_for_languages.items():
                if value[0] <= random_float <= value[1]:
                    batch_language = key
                    break

            if batch_language is None or batch_language not in self.sample_range_for_languages.keys():
                print("SystemLog: Got invalid sampling language name, re-sampling....")
        return batch_language

    def get_train_batch(self, iterator, language):
        """Simple iterator to get a batch in a specific language"""
        while True:
            batch = next(iterator[language], False)
            if batch == False: #create a new iter
                iterator[language] = iter(self.mr_dataloader_train[language])
            elif batch != None:
                break
        return batch

    def build_sample_ratio(self):
        """
            Compute normalized sample range for languages in train_langs and effective training volume.
            When the --sample_languages_uniforly is set,
                - The sampling ratios are assumed to be set to 1.0 so effective epoch size is computed correctly
                - the sampling ranges computed here are not used.

            Args:
                sample_ratio: sample ratio for different language, E.G.: en#3_pt#3_es#4, default 1.0.
                mr_dataset_train: dict which contains train data for different languages
                train_langs: training languages, E.G.: en#es.
            Returns:
                sample_range_for_languages: dict contains normalized sample range for different language,
                            E.G.: {"en":[0,0.3], "es":[0.3, 0.6], "pt":[0.6, 1]}.
                total_items_in_epoch: effective data for one epoch.

                sampling strategy:
                Suppose: sample_ratio="en#0.5_es#1_pt#2"
                        outlook data set: en 100M es 40M pt 5M
                compute acutual #data for training 1 epoch: en=100*0.5=50M es=40M pt=10M
                normalized sample prob: en=50/(50+40+10)=0.5 es=0.4 pt=0.1
                then generate a random number in range(0,1) [0.0,0.5) go with en, [0.5,0.9) go with es, [0.9, 1.0) go with pt.
        """
        LEFT_VALID_BOARDER, RIGHT_VALID_BOARDER = 0.0, 1.0

        if self.params.local_rank == 0 and self.params.node_rank == 0:
            print('SystemLog: --------Bulid Sample Ratio ------------------')
            print("SystemLog: Sample Ratio=%s"%self.params.sample_ratio)
        self.sample_range_for_languages = collections.OrderedDict()
        self.total_items_in_epoch = 0

        if self.params.sample_ratio is not None:
            for language in self.params.train_langs:
                if language not in self.mr_dataset_train.keys():
                    continue
                train_cnt = len(self.mr_dataset_train[language])
                effective_nums = int(train_cnt * float(self.params.sample_ratio[language]))
                self.total_items_in_epoch += effective_nums
                self.sample_range_for_languages[language] = [effective_nums]
                if self.params.local_rank == 0 and self.params.node_rank == 0:
                    print('SystemLog: effective %s data volumn for one epoch: %d items' % (language, effective_nums))

        for language in self.params.train_langs:
            if language not in self.sample_range_for_languages.keys():
                effective_nums = len(self.mr_dataset_train[language])
                self.sample_range_for_languages[language] = [effective_nums]
                self.total_items_in_epoch += effective_nums

        language_keys = list(self.sample_range_for_languages.keys())
        for index, language in enumerate(language_keys):
            ratio = round(float(self.sample_range_for_languages.get(language)[0]) / self.total_items_in_epoch, 4)
            if index == 0:
                range_list = [LEFT_VALID_BOARDER, ratio]
            elif index == len(language_keys)-1:
                range_list = (self.sample_range_for_languages.get(language_keys[index - 1])[1], RIGHT_VALID_BOARDER)
            else:
                range_list = (self.sample_range_for_languages.get(language_keys[index - 1])[1], self.sample_range_for_languages.get(language_keys[index - 1])[1] + ratio)
            assert LEFT_VALID_BOARDER <= all(range_list) <= RIGHT_VALID_BOARDER, "Invalid Sample Ratio Range."
            self.sample_range_for_languages[language] = range_list

    def train(self):
        self.init_train() # might need new definition here
        # ---------- print some stats at the beginning for each language -----------------
        # for language in self.params.test_langs:
        #     self.set_infer_language(language)
        #     _, best_mrr_new, best_loss_new, best_precision_new = self.validation_loop(self.params, self.train_model_inst, self.mr_dataloader_valid[language], self.mr_dataset_valid[language], self.mr_dataloader_gmr[language], self.tokenizer, self.metrics_logger_multi[language], 0, 0)
        if self.params.local_rank == 0:
            print('SystemLog: ---------------------------------------------------------')
        if self.params.local_rank == 0:
            total = sum([len(v) for k, v in self.mr_dataset_train])
            for language in self.params.train_langs:
                print('SystemLog: Starting training loop with %d items per epoch in language [%s]'%(len(self.mr_dataset_train[language]), language))
        #----------------reset set model variables for training ----------------
        start_time = time.time()
        max_training_time_reached = False
        self.start_epoch = self.metrics_logger.epochs
        self.train_model_inst.set_run_mode(run_mode="train")
        self.train_model.train()
        self.train_steps = 0
        self.global_step = 1
        self.best_tracked_metric = 0
        self.best_tracked_metric_multi = {}
        for lang in self.params.test_langs:
            self.best_tracked_metric_multi[lang] = 0

        self.tracked_metric = "Sentence_ROUGE_weight_f"
        self.train_model.zero_grad()
        self.optimizer.zero_grad()
        self.num_validations_since_last_metric_improvement = 0
        next_validation_run_time = time.time() + (self.params.validation_freq_minutes * 60)
        self.build_sample_ratio()
        # ---------- bookkeeping for MLTL loop -------------
        print('SystemLog: Starting training loop with %d items per epoch' % self.total_items_in_epoch)
        for epoch in range(self.start_epoch, self.params.max_epochs):
            if self.params.distributed_data_parallel:
                for language in self.params.train_langs:
                    self.mr_dataloader_train[language].sampler.set_epoch(epoch)
            # is this needed to be done at each epoch?
            iterator = {}
            i_batch, n_sentences = 0, 0
            for language in self.params.train_langs:
                iterator[language] = iter(self.mr_dataloader_train[language])

            # total training steps have been set to such that it takes into accound the number of items in each distributed sampler
            while n_sentences < self.total_training_size: # according to total training data
                current_language_for_batch = self.get_language()
                self.train_model_inst.set_train_language(current_language_for_batch)
                sample_batched = self.get_train_batch(iterator, current_language_for_batch)
                if  len(sample_batched['rsp_batch_tuple'][0]) > 1:
                    n_sentences += self.params.batch_size
                    # --------take train step and update model parameters--------
                    output, loss_dict = self.train_step(i_batch, sample_batched, epoch)
                    self.train_steps += 1
                    # --------Run validation save checkpoints --------
                    if self.params.validation_freq != -1 and self.train_steps % self.params.validation_freq == 0:
                        for language in self.params.test_langs:
                            self.set_infer_language(language)
                            self.best_tracked_metric = self.best_tracked_metric_multi[language]
                            _, best_mrr_new, best_loss_new, best_precision_new = self.validation_loop(self.params, self.train_model_inst, self.mr_dataloader_valid[language], self.mr_dataset_valid[language], self.mr_dataloader_gmr[language], self.tokenizer, self.metrics_logger_multi[language], epoch, i_batch)
                            # --------- Update the tracked metric ---------
                            self.update_metrics_and_save_step(epoch, i_batch, self.metrics_logger_multi[language])
                            self.best_tracked_metric_multi[language] = self.best_tracked_metric
                # -------------- Break if max training time has elapsed --------------
                i_batch += 1
                time_elapsed_minutes = (time.time() - start_time) / 60
                if self.params.max_training_time_minutes > 0 and time_elapsed_minutes > self.params.max_training_time_minutes:
                    print("SystemLog: Max training time elapsed. Ending training")
                    max_training_time_reached = True
                    break
                # ---------- Early Stopping Criteria -----------
                if self.num_validations_since_last_metric_improvement == self.params.max_metrics_plateau_for_early_stopping:
                    print("SystemLog: Early stopping due to no metrics improvement in last %s validation runs"%self.num_validations_since_last_metric_improvement)
                    return

                if max_training_time_reached:
                    break

    def compute_validation_metrics(self):
        """
            Function to evaluate metrics on GMR and validation set.
            Overrides parent class function for looping over languages.
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
            for language in self.params.test_langs:
                self.set_infer_language(language)
                self.params.lm_alpha = self.params.lm_alpha_multi[language]
                _, _, _, _ = self.validation_loop(self.params, self.train_model_inst, self.mr_dataloader_valid[language], self.mr_dataset_valid[language], self.mr_dataloader_gmr[language], self.tokenizer, self.metrics_logger_multi[language], 0, 0)

            if self.params.local_rank == 0 and self.params.node_rank == 0:
                print('SystemLog: ---------------- Final Metrics for different Languages ------------------')
                print('SystemLog: ---------------------------------------------------------')
                for language in self.params.test_langs:
                    print('SystemLog: ---------------- Final Metrics for [%s] ------------------'%language)
                    self.metrics_logger_multi[language].print_validation_metrics()
            return self.metrics_logger_multi

    def eval(self):
        print('SystemLog: ---------------------------------------------------------')
        with torch.no_grad():
            self.init_validation()
            if self.params.model_input_dir:
                # model_input_directory is not required when loading from tnlr (Bert/TNLR matching model). Make sure the model_input_directory is required depending on the model type being evaluated.
                old_params, self.train_model, _, _, _ = load_checkpoint(self.params, self.train_model)
            else:
                print("SystemLog: !!! ============================================================ !!!")
                print("SystemLog: !!! WARNING: model_input_dir is not provided for Compute Metrics !!!")
                print("SystemLog: model_input_directory is not usually required when loading from tnlr")
                print("SystemLog: !!! ============================================================ !!!")
            self.train_model = self.train_model.to(self.params.device)

            self.train_model.eval()
            self.train_model.set_run_mode(run_mode='valid')
            self.train_model.set_run_mode(run_mode='eval')
            for language in self.params.test_langs:
                print("SystemLog: predicting on", language)
                self.set_infer_language(language)
                self.params.lm_alpha = self.params.lm_alpha_multi[language]

                input_folder = os.path.join(self.params.valid_input_dir, language)
                get_line_items = self.mr_dataset_valid[language].datasets[0].get_line_items
                all_lines = []
                for file_idx, filename in enumerate(os.listdir(input_folder)):
                    filepath = os.path.join(input_folder, filename)
                    with open(filepath, 'r') as f:
                        all_lines.extend(f.readlines())

                n_batches = len(all_lines) // self.params.batch_size_infer
                collate_funct = CollateMRSequence(self.params)
                model_responses_path = os.path.join(self.params.eval_output_dir, language + '_' + OUTPUT_RESPONSES_FILE_NAME)
                with open(model_responses_path, 'w') as f:
                    for batch_idx in range(n_batches):
                        start_idx = batch_idx * self.params.batch_size_infer
                        end_idx = start_idx + self.params.batch_size_infer
                        print('SystemLog: Processed {} messages.'.format(start_idx))
                        raw_batch, msg_batch = [], []
                        for line in all_lines[start_idx:end_idx]:
                            msg = get_line_items(line)
                            if len(msg[0]) > 0 and len(msg[1]) > 0:
                                raw_batch.append(line)
                                msg_batch.append(msg)
                        batched_messages = collate_funct(msg_batch)
                        top_k_vals_list, top_k_ids_list, _ = self.train_model(batched_messages)
                        candidate_responses_list, _, _, _ = self.train_model.response_set.deduplicate_and_map_responses(top_k_ids_list, top_k_vals_list)
                        assert len(raw_batch) == len(candidate_responses_list)
                        for line, candidate_responses in zip(raw_batch, candidate_responses_list):
                            message = line.split('\t')[self.params.msg_col]
                            gold_response = line.split('\t')[self.params.rsp_col]
                            top_3_responses = '\t'.join(candidate_responses)
                            print(message, '\t', gold_response, '\t', top_3_responses, file=f)
        print('SystemLog: ---------------------------------------------------------')

    def create_metrics_logger(self, dataset_len, valid_log_name=None):
        """
            Creates metrics loggers per language.
            Also create one for tracking the overall training stats across languages (train mode)
        """
        self.metrics_logger_multi = {}
        # ------- create metrics_logger to track overall train metrics --------
        self.metrics_logger = None
        if self.params.run_mode == 'train':
            self.metrics_logger = super().create_metrics_logger(self.total_training_size)
           
        default_lm_alpha = self.params.lm_alpha
        # ------- create metrics_logger for each language for validation -------
        for language in self.params.test_langs:
            # ------- set language specific lm_alpha values when initializing metrics logger -------
            self.params.lm_alpha = self.params.lm_alpha_multi[language]
            dataset_len = len(self.mr_dataset_valid[language])
            valid_log_name = "valid.log.%d.%d.%s.txt"%(self.params.node_rank, self.params.local_rank, language)
            self.metrics_logger_multi[language] = super().create_metrics_logger(dataset_len, valid_log_name)
        #  reset lm_alpha values to default
        self.params.lm_alpha = default_lm_alpha
        return self.metrics_logger

    def setup_distributed_data_parallel(self):

        """ Setup DDP wrapper along with dataloaders.
        Loops of language specific dataloader in MLTL mode
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
        total_len_per_worker = 0
        for language in self.params.train_langs:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.mr_dataset_train[language])
            self.mr_dataloader_train[language] = DataLoader(self.mr_dataset_train[language], batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=CollateMRSequence(self.params), pin_memory=False, sampler=train_sampler)
            total_len_per_worker += len(train_sampler)
        print("SystemLog: Number of items in the train sampler in GPU %d = %d" % (self.params.local_rank, total_len_per_worker))
        self.metrics_logger.set_dataset_len(total_len_per_worker)
        self.total_training_size = total_len_per_worker
