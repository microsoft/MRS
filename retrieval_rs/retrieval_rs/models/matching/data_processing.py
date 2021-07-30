"""
    Minor changes to the dataset objects and functions to handle extra inputs and some refactoring to convert outputs to dicts instead of lists
    These would be merged with common/data_processing.py in subsequent PRs
    @Author: Budhaditya Deb
"""

import sys
import os
import pdb
import copy

import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import numpy as np

from retrieval_rs.models.common.utils import get_tokenizer
from retrieval_rs.models.common.tokenization import Vocab, Tokenizer

def get_dataset(params, run_mode="train", language=None):
    """Get the dataset instance for the given input folder
    """
    if params.local_rank == 0:
        print("SystemLog: ------------------------------------------")
        print("SystemLog: Creating dataset for %s in language [%s]"%(run_mode, language))
    tokenizer = get_tokenizer(params)
    params.vocab_size = tokenizer.get_vocab_size()
    # Use run_mode to decide input_folder, MR cols, MR max lens.
    msg_col, rsp_col = params.msg_col, params.rsp_col
    max_msg_len, max_rsp_len = params.max_msg_len, params.max_rsp_len
    architecture = params.architecture
    truncate = params.truncate
    rsp_label_col = params.rsp_label_col

    if run_mode == "train":
        input_folder = params.train_input_dir
    elif run_mode == "valid":
        input_folder = params.valid_input_dir
        # TODO: perhaps add a separate test dataset for such purposes. Or inporporate getting the labels inside the datareader itself
        rsp_label_col = -1
        # Specifically for categorical CVAE model where I have turned off the loss precision computation since it requires the response labels.
    elif run_mode == "gmr":
        input_folder = params.gmr_input_dir
    elif run_mode == "rsp_set":
        # TODO: What's the purpose of this mode?
        input_folder = params.rsp_input_dir
        msg_col, rsp_col = 0, params.rsp_text_col
        if params.truncate is False:
            max_msg_len, max_rsp_len = 1000, 1000
    elif run_mode == "eval":
        input_folder = params.eval_input_dir
    elif run_mode == "export":
        # TODO: We should remove this mode from this function since it does nothing anyways
        return None, tokenizer
    else:
        raise ValueError("Invalid run mode %s." % run_mode)
    # We consider each file to be in a separate pytorch dataset. We then use ConcatDataset to combine individual datasets
    datasets = []
    total_file_processed = 0
    if language:
        input_folder = os.path.join(input_folder, language)

    for file_idx, filename in enumerate(os.listdir(input_folder)):
        filepath = os.path.join(input_folder, filename)
        datasets.append(MRDataset(filepath, tokenizer,
                                    msg_col=msg_col,
                                    rsp_col=rsp_col,
                                    rsp_label_col=rsp_label_col,
                                    max_msg_len=max_msg_len,
                                    max_rsp_len=max_rsp_len,
                                    run_mode=run_mode,
                                    architecture=architecture,
                                    truncate=truncate,
                                    params=params))
        total_file_processed += 1
        if file_idx % 10 == 0:
            if params.local_rank == 0:
                print("SystemLog: %d files processed " % file_idx)
    mr_dataset = ConcatDataset(datasets)
    if params.local_rank == 0:
        print("SystemLog: %d files processed in total." % total_file_processed)
        print("SystemLog: Number of items in the %s dataset=%d" %(run_mode, len(mr_dataset)))
        print("SystemLog: ------------------------------------------")
    return mr_dataset

def create_dataloaders(params, run_mode='train', language=None):
    """Create Datasets and DataLoaders for training and validation
    """
    # ----------------Create Dataset objects and Dataloaders----------------
    mr_dataset = get_dataset(params, run_mode=run_mode, language=language)
    if params.local_rank == 0:
        print("SystemLog: Vocab size used for training is %d" % (params.vocab_size))
        print("SystemLog: Number of items in the %s dataset=%d" % (run_mode, len(mr_dataset)))
        print('SystenLog: -----', mr_dataset[0])
        sys.stdout.flush()
    if run_mode == 'train':
        shuffle = True
    else:
        shuffle = False
    mr_dataloader = DataLoader(mr_dataset, batch_size=params.batch_size,
                                    shuffle=shuffle, num_workers=params.num_workers,
                                    collate_fn=CollateMRSequence(params))
    return mr_dataset, mr_dataloader

class MRDataset(Dataset):
    """
        Mostly similar the original dataset with extra addition for
        response label which can be used for classifier training
    """
    def __len__(self):
        return len(self.data_offset_map)

    def __init__(self, input_file, tokenizer,
                        msg_col=0, rsp_col=1, rsp_label_col=-1,
                        max_msg_len=100, max_rsp_len=30,
                        run_mode=None,
                        architecture="matching_model",
                        truncate=False,
                        params=None):

        self.architecture = architecture
        self.data_offset_map = []
        self.fh = open(input_file, 'r', encoding='utf-8')
        self.input_file = input_file
        self.max_msg_len = max_msg_len
        self.max_rsp_len = max_rsp_len
        self.msg_col = msg_col
        self.params = params
        self.rsp_col = rsp_col
        self.rsp_label_col = rsp_label_col
        self.run_mode = run_mode
        self.tokenizer = tokenizer
        self.truncate = truncate
        self.create_file_offset()
        if self.params.aml == False:
            self.fh.close()

        # for evaluation mode define a generic tokenizer to use for computing overlap metrics such as ROUGE
        if self.run_mode in ['valid', 'gmr']:
            self.generic_tokenizer = Tokenizer(vocab_path=self.params.generic_vocab_input_dir, use_sentence_start_end=False)

    def truncate_input(self, tokens, max_num_tokens):
        # Truncate input sequence
        # Inputs: tokens is a sequence of tokens/ids returned from tokenizer, max_num_tokens is the maximum number of tokens to return
        # Outputs: None. Input tokens truncated max_num_tokens in place
        if max_num_tokens < 1:
            raise ValueError("SystemLog: Please set max_num_tokens >=1 when setting truncation")
        while len(tokens) > max_num_tokens:
            tokens.pop()

    def create_tuple_for_raw_text_input(self, list_of_messages):
        # Function to generate a msg/rsp tuple for feeding in to matching model
        # this takes a list of strings as input rather than reading from a file
        # use for ad-hoc evaluations of test messages after validation,
        # TODO: Eval mode doesn't require to create MRDataset. Discuss what's the best way to create input tuples for eval mode

        batch = []
        rsp_ids = [0, 0]
        for message in list_of_messages:
            message = self.tokenizer.tokenize(message)
            msg_ids = self.tokenizer.convert_tokens_to_ids(message)
            batch.append([msg_ids, rsp_ids, ' '.join(message), ""])
        # TODO: Don't use private functions like this + from the way it is used here, it seems that this can just be a function call instead of creating an object then callng this function
        return CollateMRSequence(self.architecture).__call__(batch)

    def __getitem__(self, idx):
        try:
            if self.params.aml == True:
                # use the open file handler
                offset = self.data_offset_map[idx]
                self.fh.seek(offset, 0)
                line = self.fh.readline()
            else:
                # open the file handle before seeking
                with open(self.input_file, 'r', encoding='utf-8') as fh:
                    offset = self.data_offset_map[idx]
                    fh.seek(offset, 0)
                    line = fh.readline()
            return self.get_line_items(line)
        except Exception:
            return ([], [], [], [])

    def get_line_items(self, line):
        """
            Extract message/reply token ids and response_labels from the given line
        """
        columns = line.rstrip().split("\t")
        msg_ids = [0, 0]
        rsp_ids = [0, 0]
        rsp_label = [-1] # this should throw error later on if not set and used
        required_columns_in_file = max([self.msg_col, self.rsp_col, self.rsp_label_col])
        # this returns both columns but only converts msg or rsp based on mode
        if len(columns) > required_columns_in_file:
            raw_msg = columns[self.msg_col]
            raw_rsp = columns[self.rsp_col]
            if self.run_mode in ['train', 'valid', 'gmr']:
                if self.msg_col != -1:
                    message = self.tokenizer.tokenize(columns[self.msg_col])
                    msg_ids = self.tokenizer.convert_tokens_to_ids(message)
                if self.rsp_col != -1:
                    reply = self.tokenizer.tokenize(columns[self.rsp_col])
                    rsp_ids = self.tokenizer.convert_tokens_to_ids(reply)
                if self.rsp_label_col >= 0:
                    rsp_label = int(columns[self.rsp_label_col])
                if self.run_mode in ['gmr', 'valid']:
                    """
                        During validation, this is used for computing ROUGE Metrics which requires generic tokenizer
                    """
                    raw_rsp = ' '.join(self.generic_tokenizer.tokenize(columns[self.rsp_col]))
            if self.run_mode == 'rsp_set':
                if self.rsp_col != -1:
                    reply = self.tokenizer.tokenize(columns[self.rsp_col])
                    rsp_ids = self.tokenizer.convert_tokens_to_ids(reply)

            if self.run_mode == 'eval':
                if self.msg_col != -1:
                    message = self.tokenizer.tokenize(columns[self.msg_col])
                    msg_ids = self.tokenizer.convert_tokens_to_ids(message)

            # ----------- Truncate ------------
            if self.truncate:
                if self.tokenizer.use_sentence_start_end:
                    msg_ids_copy = copy.deepcopy(msg_ids[1:-1])
                    self.truncate_input(msg_ids_copy, self.max_msg_len-2)
                    msg_ids = [msg_ids[0]] + msg_ids_copy + [msg_ids[-1]]

                    rsp_ids_copy = copy.deepcopy(rsp_ids[1:-1])
                    self.truncate_input(rsp_ids_copy, self.max_rsp_len-2)
                    rsp_ids = [rsp_ids[0]] + rsp_ids_copy + [rsp_ids[-1]]
                else:
                    self.truncate_input(msg_ids, self.max_msg_len)
                    self.truncate_input(rsp_ids, self.max_rsp_len)

            if len(msg_ids) <= self.max_msg_len and len(rsp_ids) <= self.max_rsp_len:
                return (msg_ids, rsp_ids, raw_msg, raw_rsp, rsp_label)
            # TODO Return Dict here
        return ([], [], [], [], [])

    def create_file_offset(self):
        """
            This function is used to populate data offset map.
            data_offset_map[idx] tells how many bytes from the start of the file we should move
            the file pointer so that it points to line idx in the file
        """
        with open(self.input_file, 'rb') as fh:
            self.data_offset_map.append(0)  # set the first offset to zero position for a new file
            for _ in fh:
                # Checks whether we have reached the end of the file or not
                # fh.fileno returns the integer id of file_descriptor,
                # fstat returns info about the file, and
                # st_size gets the file_size in bytes
                if not fh.tell() == os.fstat(fh.fileno()).st_size:
                    # Adds the current byte offset to the map
                    self.data_offset_map.append(fh.tell())

class CollateMRSequence():
    """
        Used for collating tuples of the form (msg_vec, rsp_vec) from the output of MRDataset.
        Creates a padded tensor for message and reply indexes
        Removes zero sequences
        Add masks for BERT matching model.
        Outputs a dict instead of a list
    """
    def __init__(self, params):
        self.params = params
        self.architecture = params.architecture
        self.txt_encoder_type = params.txt_encoder_type

    # TODO: __call__ function has a special meaning inside a class .. I'm not sure if we are overloading that function by design here
    def __call__(self, batch):
        """
            Args:
                batch: a list of lists from MRDataset.get_line_items()
                batch: (msg_ids, rsp_ids, raw_msg, raw_rsp, rsp_label)
            Returns:
                batch_dict: a dictionary of lists: {'msg_batch_tuple', 'rsp_batch_tuple', 'batch_messages', 'batch_replies', 'batch_rsp_labels'}
        """
        batch = [b for b in batch if len(b[0]) > 0 and len(b[1]) > 0]

        if len(batch) < 1:
            return None
        # TODO: Process for Dict once the output from get_item becomes a dict
        batch_msg_ids, batch_rsp_ids, batch_messages, batch_replies, batch_rsp_labels = list(zip(*batch))
        batch_msg_lengths = [len(msg_ids) for msg_ids in batch_msg_ids]
        max_msg_len = np.max(batch_msg_lengths)
        batch_msg_ids_padded = np.zeros([len(batch), max_msg_len], dtype=np.int32)
        batch_rsp_lengths = [len(rsp_ids) for rsp_ids in batch_rsp_ids]
        max_rsp_len = np.max(batch_rsp_lengths)
        batch_rsp_ids_padded = np.zeros([len(batch), max_rsp_len], dtype=np.int32)

        for index in range(len(batch)):
            batch_msg_ids_padded[index, :batch_msg_lengths[index]] = batch_msg_ids[index]
            batch_rsp_ids_padded[index, :batch_rsp_lengths[index]] = batch_rsp_ids[index]

        batch_msg_ids_padded = torch.LongTensor(batch_msg_ids_padded)
        batch_rsp_ids_padded = torch.LongTensor(batch_rsp_ids_padded)
        batch_msg_lengths = torch.LongTensor(batch_msg_lengths)
        batch_rsp_lengths = torch.LongTensor(batch_rsp_lengths)

        if len(batch_rsp_labels) > 0:
            batch_rsp_labels = torch.LongTensor(batch_rsp_labels)


        msg_batch_tuple = (batch_msg_ids_padded, batch_msg_lengths)
        rsp_batch_tuple = (batch_rsp_ids_padded, batch_rsp_lengths)

        if self.txt_encoder_type in ["BertModel", "BertTNLR", "MBert", 'XLMR', 'TULR']:
            # mask padded tokens
            msg_mask = np.zeros([len(batch), max_msg_len], dtype=np.int32)
            rsp_mask = np.zeros([len(batch), max_rsp_len], dtype=np.int32)
            msg_mask = torch.LongTensor(msg_mask)
            rsp_mask = torch.LongTensor(rsp_mask)

            # segment id (0 for first sentence, 1 for second sentence)
            msg_type_ids = np.zeros([len(batch), max_msg_len], dtype=np.int32)
            rsp_type_ids = np.zeros([len(batch), max_rsp_len], dtype=np.int32)
            msg_type_ids = torch.LongTensor(msg_type_ids)
            rsp_type_ids = torch.LongTensor(rsp_type_ids)

            # position id: use default position ids in pretrained model.
            for index in range(len(batch)):
                msg_mask[index, :batch_msg_lengths[index]].fill_(1)
                rsp_mask[index, :batch_rsp_lengths[index]].fill_(1)

            msg_batch_tuple = (batch_msg_ids_padded, msg_type_ids, msg_mask)
            rsp_batch_tuple = (batch_rsp_ids_padded, rsp_type_ids, rsp_mask)

        batch_dict = {}
        batch_dict['msg_batch_tuple'] = msg_batch_tuple
        batch_dict['rsp_batch_tuple'] = rsp_batch_tuple
        batch_dict['batch_messages'] = batch_messages
        batch_dict['batch_replies'] = batch_replies
        batch_dict['batch_rsp_labels'] = batch_rsp_labels

        return batch_dict
