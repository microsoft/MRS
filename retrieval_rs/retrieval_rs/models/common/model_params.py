"""
All the general session parameters and model parameters are defined here.
If you're adding a new model architecture, please subclass an existing model parameters
@author: Budhaditya Deb, Shashank Jain
"""

import argparse
from operator import itemgetter
import os
import sys
# Setting system path
SR_DIR = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir, os.pardir))
print('SystemLog: Current retrieval_rs DIR is %s' % SR_DIR)
sys.path.insert(0, SR_DIR)
from retrieval_rs.utils.config_utils import arg_bool


def build_argparser():
    # TODO: arg_bool from the utils should be used for boolean parameters
    # TODO: Add Function DocString
    # TODO: Every model class should have its own build_argparser with only the parameters relevant to it, and a global function like this just for the common arguments
    parser = argparse.ArgumentParser(description='Model parameters')

    parser.add_argument('--aml', required=False, default=False, action='store_true', help='Set flag to run on AML in DDP mode to set loal rank')
    parser.add_argument('--active_ratio_gamma', required=False, type=float, default=0.0, help='Weight of Active Ratio scores for response ranking')
    parser.add_argument('--actr_delta', required=False, type=float, default=0.0, help='Weight of ACTR scores for response ranking')
    parser.add_argument('--amp_opt_lvl', required=False, default="O1", help='opt_level for Apex-Amp')
    parser.add_argument('--architecture', required=False, default="matching_model", help='Model architecture name: matching_model, bert_matching_model')
    parser.add_argument('--batch_size', required=False, type=int, default=1100, help='Batch size for training')
    parser.add_argument('--batch_size_infer', required=False, type=int, default=1100, help='Batch size for inference')
    parser.add_argument('--batch_size_validation', required=False, type=int, default=1100, help='Batch size for validation')
    parser.add_argument('--ctr_beta', required=False, type=float, default=0.0, help='Weight of CTR scores for response ranking')
    parser.add_argument('--decay_rate', required=False, type=float, default=0.99, help='Decay rate for lr decay.')
    parser.add_argument('--decay_step', required=False, type=int, default=1000, help='Decay step for lr decay.')
    parser.add_argument('--distributed_data_parallel', dest='distributed_data_parallel', required=False, default=False, action='store_true', help='Whether to use distributed data parallel training')
    parser.add_argument('--distributed_rouge', required=False, default=False, action='store_true', help='whether to distribute rouge calculation or not')
    parser.add_argument('--do_lower_case', required=False, default=False, action='store_true', help='Whether to lower case the input text, True for uncased models, False for cased models.')
    parser.add_argument('--dummy', dest='dummy', required=False, default=False, action='store_true', help='Whether to use dummy')
    parser.add_argument('--emb_dim', required=False, type=int, help='Embedding dimension')
    parser.add_argument('--eval_input_dir', required=False, default=None, help='path of input  folder consisting of evaluation data')
    parser.add_argument('--eval_output_dir', required=False, default=None, help='path of output  folder consisting of evaluation data')
    parser.add_argument('--fp16', required=False, default=False, action='store_true', help='Whether to use 16-bit float precision instead of 32-bit')
    parser.add_argument('--generic_vocab_input_dir', required=False, default=None, help='vocab folder for generic tokenizer, it must contain a vocab.txt file, it is used by rouge evaluation.')
    parser.add_argument('--gradient_accumulation_steps', required=False, type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--gmr_input_dir', required=False, default=None, help='path of input  folder consisting of golden mr pairs')
    parser.add_argument('--infer_batches', required=False, type=int, default=20, help='number of batches for inference')
    parser.add_argument('--infiniband', required=False, default=False, action='store_true', help='Flag for whether to set environment variables for Infiniband')
    parser.add_argument('--learning_rate', required=False, type=float, default=1.0, help='Learning rate')
    parser.add_argument('--lm_alpha', required=False, type=float, default=1.4, help='Weight of Language Model Score for responses')
    parser.add_argument('--load_from', required=False, type=str, default="bert", help="which model to load: BERT or TNLR as different loading functions are needed")
    parser.add_argument('--model_language_class', required=False, type=str, default="mono_lingual", help="Set as eith mono_lingual or multi_lingual")
    parser.add_argument('--local_rank', type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--loss_scale', required=False, type=float, default=0,
                        help='Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True. 0: dynamic loss scaling. Positive power of 2: static loss scaling value.')
    parser.add_argument('--manual_seed', required=False, type=int, default=1234, help='Manual seed for reproducibility')
    parser.add_argument('--max_epochs', required=False, type=int, default=100, help='Max epochs')
    parser.add_argument('--max_golden_resp_length_to_trim_start_for_rouge', required=False, type=int, default=10, help='For ROUGE, if golden response is longer than this, trim the end. Set to -1 for no trimming.')
    parser.add_argument('--max_training_time_minutes', required=False, type=int, default=-1, help='Max training time in minutes. The training will end after this time has elapsed.')
    parser.add_argument('--max_msg_len', required=False, type=int, default=100, help='Max message length')
    parser.add_argument('--max_rsp_len', required=False, type=int, default=30, help='Max respone length')
    parser.add_argument('--model_file', required=False, type=int, default=-1, help='Model file number')
    parser.add_argument('--model_input_dir', required=False, default=None, help='path of input model directory')
    parser.add_argument('--model_output_dir', required=False, default=None, help='path of output model directory')
    parser.add_argument('--msg_col', required=False, type=int, default=0, help='Msg column')
    parser.add_argument('--msg_eval_col', required=False, type=int, default=0, help='Msg eval column. This generally refers to the unfiltered form of message.')
    parser.add_argument('--num_layers', required=False, type=int, help='Number of RNN layers in LSTM matching model')
    parser.add_argument('--num_output_responses', required=False, type=int, default=3, help='Number of output responses')
    parser.add_argument('--num_workers', required=False, type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--nccl_socket_ifname', required=False, type=str, default="eth0", help="Which network interface to use (setting it to `global` uses the default interface for the cluster)")
    parser.add_argument('--optimizer', required=False, default='adadelta', help='optimizer of matchin model: adadelta, adam')
    parser.add_argument('--pretrained_model_file', required=False, default=None, help='Pretrained BERT/TNLR model file name without path.')
    parser.add_argument('--pretrained_model_path', required=False, default=None, help='path of input folder consisting of pretrained BERT/TNLR model file and configuration.')
    parser.add_argument('--remove_tee_tags', required=False, default=False, action='store_true', help='Whether to remove TEE tags used for email pre-processing')
    parser.add_argument('--remove_unk_tokens', required=False, default=False, action='store_true', help='Whether to remove UNK tokens for vocab id converting')
    parser.add_argument('--rnn_hidden_dim', required=False, type=int, help='Rnn hidden dimension')
    parser.add_argument('--rsp_text_col', required=False, type=int, default=1, help='Response text column')
    parser.add_argument('--rsp_lm_col', required=False, type=int, default=0, help='Response lm column')
    parser.add_argument('--rsp_cluster_col', required=False, type=int, default=2, help='Response cluster column')
    parser.add_argument('--rsp_input_dir', required=False, default=None, help='path of response set folder')
    parser.add_argument('--rsp_mapping_input_dir', required=False, default=None, help='path of response mapping file folder (formatting evaluation response output)')
    # TODO: What's the difference between rsp_col and rsp_text_col??? -- I believe one is index in the response_set and another is in the MR-Pairs set?
    # -- If so, we should make it clearer in the names
    parser.add_argument('--rsp_col', required=False, type=int, default=1, help='Response column')
    parser.add_argument('--rsp_ctr_col', required=False, type=int, default=3, help='Column for CTR value of a response in response set file')
    parser.add_argument('--rsp_actr_col', required=False, type=int, default=4, help='Column for ACTR value of a response in response set file')
    parser.add_argument('--rsp_activeratio_col', required=False, type=int, default=5, help='Column for ActiveRatio value of a response in response set file')
    parser.add_argument('--run_mode', required=False, default="train", help='Running mode of the model')
    parser.add_argument('--save_each_epoch', required=False, default=True, help='True to save model after each epoch. False to only do this in the final epoch.')
    parser.add_argument('--save_freq', required=False, type=int, default=-1,
                        help='Model Checkpointing Frequency (Set it to -1 to switch it off, otherwise the model will be saved after these many number of train steps))')
    parser.add_argument('--show_comp_exception', dest='show_comp_exception', required=False, default=False, action='store_true', help='Whether to show compliant exception')
    parser.add_argument('--steps_per_print', required=False, type=int, default=50, help='Steps per print')
    parser.add_argument('--sweep_lm', required=False, default=False, action='store_true', help='Whether to sweep lm_alpha in each validation.')
    parser.add_argument('--sweep_lm_span', required=False, type=int, default=2, help='Parameter to decide the span for LM alpha sweep.')
    parser.add_argument('--sweep_lm_window', required=False, type=float, default=0.5, help='Parameter to decide the window for LM alpha sweep.')
    parser.add_argument('--tokenizer', required=False, default=None, help='generic, wordpiece or sentencepiece')
    parser.add_argument('--topk_for_rouge', required=False, type=int, default=3, help='topk_for_rouge in the rouge computation while training')
    parser.add_argument('--total_training_steps', required=False, type=int, default=1000000, help='Total training steps for lr decay.')
    parser.add_argument('--tune_parameters_lm_alpha_min', required=False, type=float, default=0, help='Min bound for sweeping lm_alpha with run_mode == tune_parameters')
    parser.add_argument('--tune_parameters_lm_alpha_max', required=False, type=float, default=3, help='Max bound for sweeping lm_alpha (not inclusive) with run_mode == tune_parameters')
    parser.add_argument('--tune_parameters_lm_alpha_step', required=False, type=float, default=0.1, help='Step size for sweeping lm_alpha with run_mode == tune_parameters')
    parser.add_argument('--tune_parameters_ctr_beta_min', required=False, type=float, default=0, help='Min bound for sweeping ctr_beta with run_mode == tune_parameters')
    parser.add_argument('--tune_parameters_sweep_ctr_beta_max', required=False, type=float, default=1, help='Max bound for sweeping ctr_beta (not inclusive) with run_mode == tune_parameters')
    parser.add_argument('--tune_parameters_sweep_ctr_beta_step', required=False, type=float, default=1, help='Step size for sweeping ctr_beta with run_mode == tune_parameters')
    parser.add_argument('--tune_parameters_actr_delta_min', required=False, type=float, default=0, help='Min bound for sweeping actr_delta with run_mode == tune_parameters')
    parser.add_argument('--tune_parameters_actr_delta_max', required=False, type=float, default=1, help='Max bound for sweeping actr_delta (not inclusive) with run_mode == tune_parameters')
    parser.add_argument('--tune_parameters_actr_delta_step', required=False, type=float, default=1, help='Step size for sweeping actr_delta with run_mode == tune_parameters')
    parser.add_argument('--tune_parameters_active_ratio_gamma_min', required=False, type=float, default=0, help='Min bound for sweeping active_ratio_gamma with run_mode == tune_parameters')
    parser.add_argument('--tune_parameters_ratio_gamma_max', required=False, type=float, default=1, help='Max bound for sweeping active_ratio_gamma (not inclusive) with run_mode == tune_parameters')
    parser.add_argument('--tune_parameters_active_ratio_gamma_step', required=False, type=float, default=1, help='Step size for sweeping active_ratio_gamma with run_mode == tune_parameters')
    parser.add_argument('--train_input_dir', required=False, default=None, help='path of input  folder consisting of training data')
    parser.add_argument('--truncate', required=False, default=False, action='store_true', help='Whether to truncate messages and responses based on params msg_max_len and rsp_max_len')
    parser.add_argument('--use_ctr_response_ranking', required=False, default=False, help='whether to use CTR in response rankings')
    parser.add_argument('--use_cuda', required=False, default='True', type=arg_bool, help='Whether to use GPU or not')
    parser.add_argument('--use_horovod', required=False, default=False, help='whether to use horovod for distributed training')
    parser.add_argument('--use_shuffle', dest='use_shuffle', required=False, default=False, action='store_true', help='Whether to use shuffle during training')
    parser.add_argument('--use_multiple_files', dest='use_multiple_files', required=False, default=False, action='store_true', help='Whether to use multiple files in data offset creation')
    parser.add_argument('--use_sentence_start_end', dest='use_multiple_files', required=False, default=True, action='store_false', help='Whether to use multiple files in data offset creation')
    parser.add_argument('--ut_early_stop', dest='ut_early_stop', required=False, default=False, action='store_false', help='True for ut early stop when mrr hit 1.0')
    parser.add_argument('--valid_input_dir', required=False, default=None, help='path of input  folder consisting of validation data')
    parser.add_argument('--validation_batches', required=False, type=int, default=20, help='Validation batch size')
    parser.add_argument('--validation_freq', required=False, type=int, default=-1,
                        help='Validation frequency (Set it to -1 to switch off validation during training, otherwise validation is done after these many number of train steps)')
    parser.add_argument('--validation_freq_minutes', required=False, type=int, default=-1,
                        help='Validation frequency in minutes (Set to -1 to switch off during training)')
    parser.add_argument('--verbose', dest='verbose', required=False, default=False, action='store_true', help='Whether to use verbose logs')
    parser.add_argument('--vocab_input_dir', required=False, default=None,
                        help='vocab folder or vocab.txt path for generic/wordpiece tokenizer, vocab folder and sentencepiece model prefix for sentencepiece tokenizer')
    parser.add_argument('--vocab_sentencepiece_model_prefix', required=False, default=None, help='sentencepiece model prefix for sentencepiece tokenizer')
    parser.add_argument('--warmup_proportion', required=False, type=float, default=0.0002, help='Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.')
    return parser


class HyperParameters(object):
    # TODO: Add Class DocString
    def __init__(self):
        # TODO: It's easy to misalign default values here with the argparser defaults .. figure out a solution
        # TODO: Check to make sure that this class only contains the parameters requires for ALL models
        self.active_ratio_gamma = 0.0
        self.actr_delta = 0.0
        self.aml = False
        self.amp_opt_lvl = "O1"
        self.architecture = "matching_model"
        self.batch_size = 1100
        self.batch_size_infer = 1100
        self.batch_size_validation = 1100
        self.ctr_beta = 0.0
        self.data_parallel = False
        self.decay_step = 1000
        self.decay_rate = 0.99
        self.do_lower_case = False
        self.distributed_data_parallel = False
        self.distributed_rouge = False
        self.dummy = False
        self.emb_dim = 320
        self.eval_input_dir = None
        self.eval_output_dir = None
        self.fp16 = False
        self.generic_vocab_input_dir = None
        self.gradient_aggregation_method = "default"
        self.gradient_accumulation_steps = 1
        self.gradient_clipping_threshold = 500
        self.gmr_input_dir = None
        self.generate_vocab = False
        self.infer_batches = 20
        self.infiniband = False
        self.learning_rate = 1.0
        self.lm_alpha = 1.4
        self.local_rank = -1
        self.loss_scale = 0.0
        self.max_epochs = 20
        self.max_golden_resp_length_to_trim_start_for_rouge = 10
        self.max_training_time_minutes = -1
        self.max_msg_len = 100
        self.max_rsp_len = 30
        self.manual_seed = 1234
        self.model_file = -1
        self.model_input_dir = None
        self.model_output_dir = None
        self.msg_col = 0
        self.msg_eval_col = 0
        self.nccl_socket_ifname = 'eth0'
        self.node_rank = 0
        self.num_gpu = 1
        self.num_layers = 2
        self.num_output_responses = 3
        self.num_workers = 20
        self.optimizer = "adadelta"
        self.pretrained_model_file = None
        self.pretrained_model_path = None
        self.remove_tee_tags = False
        self.rnn_hidden_dim = 300
        self.rsp_text_col = 1
        self.rsp_lm_col = 0
        self.rsp_cluster_col = 2
        self.rsp_ctr_col = 3
        self.rsp_actr_col = 4
        self.rsp_activeratio_col = 5
        self.rsp_input_dir = None
        self.rsp_col = 1
        self.rsp_mapping_input_dir = None
        self.run_mode = "train"
        self.save_each_epoch = True
        self.save_freq = 5000
        self.show_comp_exception = False
        self.steps_per_print = 50
        self.sweep_lm_span = 2
        self.sweep_lm_window = 0.5
        self.tokenizer = None
        self.topk_for_rouge = 3
        self.total_training_steps = 1000000
        self.train_input_dir = None
        self.use_ctr_response_ranking = False
        self.use_cuda = True
        self.use_horovod = False
        self.use_shuffle = False
        self.use_multiple_files = False
        self.use_sentence_start_end = True
        self.ut_early_stop = False
        self.valid_input_dir = None
        self.validation_batches = 20
        self.validation_freq = -1
        self.validation_freq_minutes = -1
        self.verbose = False
        self.vocab_input_dir = None
        self.vocab_size = None
        self.vocab_sentencepiece_model_prefix = None
        self.warmup_proportion = 0.0002
        self.model_language_class = 'mono_lingual'
        self.truncate = False


    def set_args(self, args, verbose=False):
        # TODO: Add Function DocString
        print('SystemLog: ---------------------------------------------------------')
        for k, v in vars(args).items():
            # TODO: This needs to be revisited since at times the default value in argparser might differ from Class's default values
            if v is not None:
                if verbose:
                    print("SystemLog: Setting {}\t{}".format(k, v))
                setattr(self, k, v)

        if self.vocab_input_dir:
            if os.path.isdir(self.vocab_input_dir):
                if self.generic_vocab_input_dir is None:
                    self.generic_vocab_input_dir = self.vocab_input_dir
                # TODO: This filename should be a global Constant
                # TODO: Add a new attribute vocab_file or something instead of conflating the meaning of vocab_dir
                if self.tokenizer in ["sentencepiece", "xlmr"]:
                    if self.vocab_sentencepiece_model_prefix is None:
                        raise ValueError(
                            'When tokenizer is sentencepiece and vocab_input_dir is a directory, vocab_sentencepiece_model_prefix can\'t be empty')
                    self.vocab_input_dir = os.path.join(self.vocab_input_dir, self.vocab_sentencepiece_model_prefix)
                else:
                    self.vocab_input_dir = os.path.join(self.vocab_input_dir, "vocab.txt")

        if self.generic_vocab_input_dir and os.path.isdir(self.generic_vocab_input_dir):
            self.generic_vocab_input_dir = os.path.join(self.generic_vocab_input_dir, "vocab.txt")

        # TODO: These are model specific modifications/checks and shouldn't be in the base class
        # TODO: Always add brackets to avoid any kind of ambiguity or accidental mis-evaluations
        # TODO: Default value of Tokenizer is None (both in class and argparser), so check for None here
        if self.architecture == 'bert_matching_model' and self.tokenizer not in ['wordpiece', 'sentencepiece']:
            raise ValueError('SystemLog: The tokenizer %s should be FullTokenizer for BERT model' % self.tokenizer)

        # TODO: Use brackets here to remove ambiguity from the condition
        # TODO: Add description of what's going on here
        if self.do_lower_case and '-cased' in self.vocab_input_dir or (not self.do_lower_case and '-uncased' in self.vocab_input_dir):
            raise ValueError('SystemLog: The do_lower_case setting %s is inconsistent with vocab %s.' % (self.do_lower_case, self.vocab_input_dir))

        if self.gradient_accumulation_steps < 1:
            raise ValueError('SystemLog: Invalid gradient accumulation steps parameter: %s, should be >= 1.' % self.gradient_accumulation_steps)

        if self.fp16 and self.optimizer == "adadelta":
            raise ValueError("SystemLog: fp16 is only for Adam optimizer currently .. exiting")


def print_parameters(params):
    # TODO: Function already exists in the utils ... should be imported from there
    print('SystemLog: ---------------------------------------------------------')
    print("SystemLog: Printing model parameters")
    param_list = [(k, v) for k, v in vars(params).items()]
    # TODO: Key is not needed since that is the default
    param_list.sort(key=itemgetter(0))
    for k, v in param_list:
        print('SystemLog: %s\t%s' % (k, v))

def save_parameters(params, mode='train'):
    # TODO: Function already exists in the utils ... should be imported from there
    print('SystemLog: ---------------------------------------------------------')
    print("SystemLog: Saving model parameters")
    param_list = [(k, v) for k, v in vars(params).items()]
    # TODO: Key is not needed since that is the default
    param_list.sort(key=itemgetter(0))
    with open(os.path.join(params.model_output_dir, '%s.params.txt'%mode), 'w') as f:
        for k, v in param_list:
            f.write('%s\t%s\n' % (k, v))
