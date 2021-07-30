"""
Code for some common utility functions for training and validation
@author: Budhaditya Deb, Shashank Jain, Lili Zhou
"""

# coding=utf-8
import time
import os
import sys
import gc
import copy
import random
import logging
import numpy as np
#from tensorboardX import SummaryWriter

try:
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
except Exception as ex:
    # TODO: Don't just print the lack of NLTK -- set some flag and use that flag to exit graciously where these imports are being used
    print("SystemLog: nltk not found while importing in utils %s" % ex)

import torch
from torch.utils.data import DataLoader, ConcatDataset
from retrieval_rs.models.common.data_processing import MRDataset, CollateMRSequence
from retrieval_rs.models.common.tokenization import Tokenizer, FullTokenizer, SentencepieceTokenizer, XLMRobertaTokenizerWrapper
from retrieval_rs.models.pytorch_transformers.optimization import BertAdam
from retrieval_rs.metric.rouge_n_ensemble import Sentence_ROUGE_n, Sentence_ROUGE_ensemble_f
from retrieval_rs.metric.rouge import Rouge
from retrieval_rs.constants.weights import SCORE_ERROR, UNIGRAM_WEIGHT, BIGRAM_WEIGHT, TRIGRAM_WEIGHT
from retrieval_rs.constants.file_names import RESPONSE_SET_FILE_NAME
import torch.distributed as dist


def create_trainval_dataloaders(params):
    """Create Datasets and DataLoaders for training and validation
    """
    # ----------------Create Dataset objects and Dataloaders----------------
    mr_dataset_train, tokenizer = get_dataset(params, run_mode="train")
    params.vocab_size = tokenizer.get_vocab_size()
    print("SystemLog: Vocab size used for training is %d" % (params.vocab_size))
    print("SystemLog: Number of items in the train dataset=%d" % len(mr_dataset_train))
    sys.stdout.flush()
    # Collate Function pads the sequences to have a uniform length for the entire batch
    mr_dataloader_train = DataLoader(mr_dataset_train, batch_size=params.batch_size,
                                     shuffle=True, num_workers=params.num_workers, collate_fn=CollateMRSequence(params.architecture))

    mr_dataset_valid, _ = get_dataset(params, run_mode="valid")
    print("SystemLog: Number of items in the valid dataset=%d" % len(mr_dataset_valid))
    mr_dataloader_valid = DataLoader(mr_dataset_valid, batch_size=params.batch_size_validation,
                                     shuffle=False, num_workers=0, collate_fn=CollateMRSequence(params.architecture))

    return mr_dataset_train, mr_dataloader_train, mr_dataset_valid, mr_dataloader_valid


# TODO: Why do we need to return the tokenizer here? -- There's a separate function for that
def get_dataset(params, run_mode="train"):
    """Get the dataset instance for the given input folder
    """
    tokenizer = get_tokenizer(params)
    # Use run_mode to decide input_folder, MR cols, MR max lens.
    msg_col, rsp_col = params.msg_col, params.rsp_col
    max_msg_len, max_rsp_len = params.max_msg_len, params.max_rsp_len
    if run_mode == "train":
        input_folder = params.train_input_dir
    elif run_mode == "valid":
        input_folder = params.valid_input_dir
    elif run_mode == "gmr":
        input_folder = params.gmr_input_dir
        if params.truncate is False:
            max_msg_len, max_rsp_len = np.inf, np.inf
    elif run_mode == "rsp_set":
        # TODO: What's the purpose of this mode?
        input_folder = params.rsp_input_dir
        msg_col, rsp_col = 0, params.rsp_text_col
        # TODO: These values should be global parameters instead of being hard coded like this
        # TODO: Why not just set these values to np.inf like above?
        if params.truncate is False:
            max_msg_len, max_rsp_len = 1000, 1000
    elif run_mode == "eval":
        input_folder = params.eval_input_dir
    elif run_mode == "export":
        # TODO: We should remove this mode from this function since it does nothing anyways
        return None, tokenizer
    else:
        raise ValueError("SystemLog: Invalid run mode %s." % run_mode)

    # We consider each file to be in a separate pytorch dataset. We then use ConcatDataset to combine individual datasets
    datasets = []
    total_file_processed = 0
    # This sorting of file is done to make sure that we get the same file order each time
    for file_idx, filename in enumerate(sorted(os.listdir(input_folder))):
        filepath = os.path.join(input_folder, filename)
        datasets.append(MRDataset(filepath, tokenizer, msg_col=msg_col,
                                  rsp_col=rsp_col, max_msg_len=max_msg_len,
                                  max_rsp_len=max_rsp_len, run_mode=run_mode, architecture=params.architecture, truncate=params.truncate))
        total_file_processed += 1
        if file_idx % 1000 == 0:
            print("SystemLog: %d files processed " % file_idx)
    print("SystemLog: %d files processed in total." % total_file_processed)
    mr_dataset = ConcatDataset(datasets)

    return mr_dataset, tokenizer


# TODO: Maybe a better way is to just pass the list of parameters we want to optimize, instead of the entire model (thus allowing greater flexibility)
# TODO: I think passing the entire params dictionary everywhere is not a great design .. we can/should just pass the values this function needs -- thus making the code more reusable
def get_optimizer(train_model, params):
    """Get the optimizer to train the matching model.
    """
    # TODO: Why are we looking for an entry in a list instead of directly matching the string?
    if params.optimizer in ["adam"]:
        param_optimizer = list(train_model.named_parameters())
        # Set weight decay of bias and LayerNorm.weight to zero.
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if params.fp16:
            print("SystemLog: Loading Apex and building the FusedAdam optimizer.")
            try:
                from apex import amp
                from apex.optimizers import FusedAdam
                from apex.amp import _amp_state
            except:
                raise ImportError(
                    "SystemLog: Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            if params.amp_opt_lvl not in ['O0', 'O1', 'O2', 'O3']:
                raise ValueError("SystemLog: %s amp_opt_level is not supported" % params.amp_opt_lvl)

            print("SystemLog: Using %s opt_level for Amp" % params.amp_opt_lvl)

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=params.learning_rate,
                                  bias_correction=False)
            if params.loss_scale == 0:
                train_model, optimizer = amp.initialize(train_model, optimizer, opt_level=params.amp_opt_lvl, loss_scale="dynamic")
            else:
                train_model, optimizer = amp.initialize(train_model, optimizer, opt_level=params.amp_opt_lvl, loss_scale=params.loss_scale)
            amp._amp_state.loss_scalers[0]._loss_scale = 2**20
        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=params.learning_rate,
                                 warmup=params.warmup_proportion,
                                 t_total=params.total_training_steps)
    else:
        # TODO: These parameters should be made configurable through global parameters/function arguments
        optimizer = torch.optim.Adadelta(train_model.parameters(), lr=params.learning_rate, rho=0.95, eps=1e-08, weight_decay=0)

    return train_model, optimizer


def get_tokenizer(params):
    if params.tokenizer == "wordpiece":
        print("SystemLog: Using wordpiece tokenizer")
        strip_accents = params.do_lower_case
        return FullTokenizer(vocab_path=params.vocab_input_dir, use_sentence_start_end=params.use_sentence_start_end, make_lowercase=params.do_lower_case, strip_accents=strip_accents, remove_tee_tags=params.remove_tee_tags, remove_unk_tokens=params.remove_unk_tokens)
    elif params.tokenizer == "sentencepiece":
        print("SystemLog: Using sentencepiece tokenizer")
        tokenizer = SentencepieceTokenizer(vocab_path=params.vocab_input_dir)
        tokenizer.load_model()
        return tokenizer
    elif params.tokenizer == "xlmr":
        print("SystemLog: Using XLM-R tokenizer")
        tokenizer = XLMRobertaTokenizerWrapper(vocab_path=params.vocab_input_dir, use_sentence_start_end=params.use_sentence_start_end, remove_tee_tags=params.remove_tee_tags)
        return tokenizer
    else:
        print("SystemLog: Using default tokenizer")
        return Tokenizer(vocab_path=params.vocab_input_dir, use_sentence_start_end=params.use_sentence_start_end, make_lowercase=True, strip_accents=True, remove_tee_tags=params.remove_tee_tags, remove_unk_tokens=params.remove_unk_tokens)


# TODO: Split this function into a save_checkpoint, and a save_model function
def save_model(params, trained_model, optimizer, metrics_logger, epoch, mb, update_best_model=True):
    '''Saves model and/or checkpoint
    When the `save_best_model` parameter is set to true, saves just the provided PyTorch model's state_dict
    When the `save_best_model` parameter is set to False, saves the provided PyTorch model's state_dict, as well as saves a checkpoint containing:
    - Params Object
    - Epoch number
    - Model's state dict
    - Optimizer's state dict
    - Metrics_logger Object
    '''
    print('SystemLog: ---------------------------------------------------------')
    if update_best_model:
        print('SystemLog: Epoch %d, batch=%d, updating best model' % (epoch, mb))
        # TODO: These model names should be global constants
        path_to_model = os.path.join(params.model_output_dir, 'model.best.pt')
    else:
        print('SystemLog: Epoch %d, batch=%d, saving model and checkpoint' % (epoch, mb))
        # TODO: These model and checkpoint names/patterns should be global constants
        path_to_model = os.path.join(params.model_output_dir, 'model.%d.pt' % epoch)
        path_to_checkpoint = os.path.join(params.model_output_dir, 'model_checkpoint.tar')
        # TODO: Check if the serializer of metrics_logger is well-defined and optimized
        print("SystemLog: Saving model checkpoint to %s" % path_to_checkpoint)
        torch.save({
            'params': params,
            'epoch': epoch,
            'model_state_dict': trained_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics_logger': metrics_logger
        }, path_to_checkpoint)
        print("SystemLog: Done saving model checkpoint")

    print("SystemLog: Saving model to %s" % path_to_model)
    torch.save(trained_model.state_dict(), path_to_model)
    print("SystemLog: Done saving model")
    print('SystemLog: ---------------------------------------------------------')


def load_checkpoint(params, model, optimizer=None, metrics_logger=None):
    '''Loads the PyTorch model, or the full checkpoint.
    1) This function should only be called when params.model_input_dir is non-None; otherwise an exception is thrown
    2) If the params.run_mode is `Train`, then we try to load the full model checkpoint (which contains metrics_logger, epoch number, optimizer_state etc)
    3) If the params.run_mode is anything else, we try to load just the PyTorch model using the params.model_file value
        a) If params.model_file is -1, then we attempt to load the best model (model.best.pt)
        b) If params.model_file is anything else, then we attempt to load the model with the name model.<params.model_file>.pt
    4) The model and optimizer objects are modified in place, whereas new metrics_logger, and params objects are created and returned
        (this behavior might change in future)
    5) Returned checkpoint is None for the non-training scenarios (i.e. when params.run_mode != train)
    '''
    checkpoint = None
    if params.model_input_dir:
        if params.run_mode == "train":
            # TODO: These checkpoint names should be global constants
            possible_model_file = os.path.join(params.model_input_dir, 'model_checkpoint.tar')
            print("SystemLog: Trying to load the model checkpoint from %s" % possible_model_file)

            checkpoint = torch.load(possible_model_file, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                # Load optimizer
                if checkpoint['optimizer_state_dict'] is not None:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    raise ValueError("SystemLog: The optimizer state is None in the model checkpoint")

            params = checkpoint['params']
            metrics_logger = checkpoint['metrics_logger']
            # Increment epoch number here: If the last checkpoint was x, then the current run should start from x+1
            metrics_logger.epochs += 1
            params.model_file = checkpoint['epoch']

            print('SystemLog: Successfully Loaded checkpoint from {}'.format(possible_model_file))
        else:
            if params.model_file == -1:
                # TODO: These model name patterns should be global constants
                possible_model_file = os.path.join(params.model_input_dir, 'model.best.pt')
                print("SystemLog: Specified model_file is -1 .. trying to load the best model from %s" % possible_model_file)
            else:
                # TODO: These model name patterns should be global constants
                possible_model_file = os.path.join(params.model_input_dir, 'model.%d.pt' % params.model_file)
                print("SystemLog: Trying to load the model with %d model_file from %s" % (params.model_file, possible_model_file))

            if not os.path.exists(possible_model_file):
                raise FileNotFoundError('SystemLog: Could not find model named: %s' % possible_model_file)
            else:
                model.load_state_dict(torch.load(possible_model_file, map_location=torch.device('cpu')))
                print('SystemLog: Successfully Loaded model from {}'.format(possible_model_file))
    else:
        raise ValueError("SystemLog: --model_input_dir has not been offered, which is necessary to load a model/checkpoint .. Exiting")

    # TODO: No need to return model and optimizer -- they are modified in-place
    return params, model, optimizer, metrics_logger, checkpoint


def compute_metrics(rsp_id, top_k_ids, k=3):
    # computes reciprocal rank and precision@k for a golden rsp_id and a ranked list of top_k_ids
    if rsp_id in top_k_ids:
        rank = top_k_ids.index(rsp_id)
        rr = 1.0 / (rank + 1)
    else:
        rank = len(top_k_ids) + 1.0
        rr = 1.0 / rank

    if rsp_id in top_k_ids[0:k]:
        p_at_k = 1.0
    else:
        p_at_k = 0.0
    return rr, p_at_k


def compute_rouge_n(golden_rsp, top_k_responses):
    # computes rouge score for a golden response and a ranked list of top k responses
    rouge_n = []
    for rsp in top_k_responses:
        try:
            rouge_n.append(Rouge().get_scores(rsp, golden_rsp)[0])
        except:
            rouge_n.append(SCORE_ERROR)
    return rouge_n

# TODO: Function name can be improved
def get_file_from_dir(input_path, verbose=True):
    # Returns the input_path as it is if it is a file.
    # Otherwise combines the files present in input_path and returns the combined file
    if not os.path.exists(input_path):
        raise ValueError("SystemLog: Path {} doesn't exist".format(input_path))

    if os.path.isfile(input_path):
        return input_path
    else:
        # TODO: Check what's the use-case of the else condition .. seems very arbitrary.
        all_files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
        if verbose:
            print("SystemLog: Found {} files at {}. Combining into one file.".format(len(all_files), input_path))

        if len(all_files) == 1:
            return all_files[0]
        else:
            # Combine all files into one
            # Create combined dir in the current working directory. This is just a temporary throw away directory after combining various files present
            # in an input folder. Use the current time to create a random directory in current working directory
            # Purpose of this directory is just to read the combined input of all the
            # files present in input_path directory
            # TODO: Avoid creating throwaway directories esp. at current_dir ... even if you do, remove them after you are done
            combined_dir = os.path.join(os.getcwd(), str(int(time.time())))
            if not os.path.exists(combined_dir):
                if verbose:
                    print("SystemLog: Creating combined dir %s" % combined_dir)
                os.makedirs(combined_dir)

            combined_path = os.path.abspath(os.path.join(combined_dir, "combined.txt"))
            if verbose:
                print("SystemLog: Combining following files at %s" % combined_path)
            # TODO: Switch to utf-8 unless there's a very special reason for this
            with open(combined_path, 'w', encoding="ascii", errors="surrogateescape") as outfile:
                for filename in all_files:
                    if verbose:
                        print("SystemLog: " + filename)
                    # TODO: Switch to utf-8 unless there's a very special reason for this
                    with open(filename, 'r', encoding="ascii", errors="surrogateescape") as infile:
                        # TODO: Check if the contents of the subsequent files are printed on the same line or on the newline
                        outfile.write(infile.read())
            return combined_path


def to_cuda(*args):
    """Move tensors to GPU device.
    """
    return [None if x is None else x.cuda() for x in args]


def to_device(*args, device):
    return [None if x is None else x.to(device) for x in args]

def export_response_file(params, rs_path, response_cluster_path, responses_path):
    """Converts response file as per Qas export specifications
    """
    print("SystemLog: Creating response cluster file at %s" % response_cluster_path)
    print("SystemLog: Creating responses file at %s" % responses_path)
    with open(rs_path, 'r', encoding='utf-8') as in_file, \
            open(response_cluster_path, 'w', encoding='utf-8') as fout1, \
            open(responses_path, 'w', encoding='utf-8') as fout2:
        for line in in_file:
            tokens = line.strip().split('\t')
            # TODO: Change these to forward indexes instead of backward
            fout1.write('{0}\t{1}\n'.format(tokens[params.rsp_text_col], tokens[params.rsp_cluster_col]))
            fout2.write('{0}\n'.format(tokens[params.rsp_text_col]))

def set_random_seeds(seed: int):
    """Sets the seed for the random number generators of python, numpy, torch.random.

    Args:
        seed: The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.debug("SystemLog: Set random seed {}".format(seed))


def create_optimizer(model: torch.nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0,
                     momentum: float = 0.0) -> torch.optim.Optimizer:
    """Creates the optimizer for the model.

    Args:
        model: The Module object representing the network.
        optimizer_name: The name of the optimizer to use. Only Adadelta, Adam, SGD, and RMSprop are supported.
        learning_rate: The initial learning rate.
        weight_decay: Weight decay value (L2 penalty)
        momentum: Momentum factor.

    Returns:
        The optimizer object.

    Raises:
        ValueError: If the optimizer is unknown.
    """
    if optimizer_name.lower() == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum,
                                    weight_decay=weight_decay)
    elif optimizer_name.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), learning_rate, weight_decay=weight_decay,
                                        momentum=momentum)
    else:
        raise ValueError("SystemLog: The optimizer must be Adadelta, Adam, SGD, or RMSprop "
                         "(optimizer: {})".format(optimizer_name))

    return optimizer


def create_criterion(loss_name: str) -> torch.nn.Module:
    # TODO: Move to models/common_model_lib/losses.py
    """Creates the loss criterion.

    Args:
        loss_name: The name of the loss function. Only CrossEntropy and NLL are supported.

    Returns:
        The criterion object.

    Raises:
        ValueError: If the loss function name is unknown.
    """
    if loss_name.lower() == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_name.lower() == "nll":
        criterion = torch.nn.NLLLoss()
    else:
        raise ValueError("SystemLog: The loss function must CrossEntropy or NLL "
                         "(loss function: {})".format(loss_name))

    return criterion

def create_lr_scheduler(lr_scheduler_name: str, optimizer: torch.optim.Optimizer,
                        lr_gamma: float, max_epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
    """Creates a learning rate scheduler to use for training.

    Args:
        lr_scheduler_name: The name of the scheduler to create. Only Exponential and Linear are supported.
        optimizer: The optimizer object used for training.
        lr_gamma: Multiplicative decay factor.
        max_epochs: The maximum number of training epochs that the model using this scheduler will run for.

    Returns:
        The learning rate scheduler.

    Raises:
        ValueError: If the requested learning rate scheduler is unknown.
    """
    if lr_scheduler_name.lower() == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)
    elif lr_scheduler_name.lower() == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - epoch / max_epochs)
    else:
        raise ValueError("SystemLog: Unknown learning rate scheduler {}".format(lr_scheduler_name))

    return lr_scheduler

def get_path_to_pretrained_model(params):
    """
        Sets the pretrained model path used for Bert/TNLR based models to
        - Initialize MAtchingModel from pretrained Bert/TNLR files
        - Initialize MCVAE model from matching or bert_matching model
        - Initialize BertModel and BertTNLR from pretrained model file
        Args:
            params.pretrained_model_path: name of directory from which to initialize model
            params.pretrained_model_file: name of file from which to initialize model
            params.pretrained_model_number: if file name is not provided use model # to set path
        Returns:
            path_to_model: absolute path of the pretrained model file
    """
    if params.pretrained_model_path:
        if params.pretrained_model_file is not None:
            path_to_model = os.path.join(params.pretrained_model_path, params.pretrained_model_file)
        else:
            if params.pretrained_model_number != -1:
                path_to_model = os.path.join(params.pretrained_model_path, 'model.%d.pt' % params.pretrained_model_number)
            else:
                path_to_model = os.path.join(params.pretrained_model_path, 'model.best.pt')

        if os.path.exists(path_to_model):
            print('SystemLog: Loading pretrained model from [%s]' % path_to_model)
        else:
            raise ValueError("SystemLog: Pretrained model path [%s] does not exist."% path_to_model)
        return path_to_model
    else:
        raise ValueError("SystemLog: Pretrained model path not set.")

def _all_gather(tensor):
    # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather
    output = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(tensor=tensor, tensor_list=output)
    return torch.cat(output, dim=0)
