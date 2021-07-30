"""
File contains the text processing functions for training the Matching and M-CVAE models
-   MRDataset Class: Message Reply dataset reader class which converts a pair of messages and
    responses to indiced for feeding into the models
@authors: Pankaj Gulhane, Budhaditya Deb, Guilherme Ilunga, Shashank Jain, Lili Zhou
"""

# coding=utf-8
import os
import copy
import numpy as np
import pdb
import torch
from torch.utils.data import Dataset

# TODO: We should figure out when to close the open file pointer as well
class MRDataset(Dataset):
    # class for reading m-r pairs in text format and feed to Matching model
    # Input is a directory containing a set of files,
    # Each file should have one or more columns of text
    # Note: columns required are different from training, evaluation and response set
    # Reads message and response pairs, and tokenize to indexes from vocab

    def __init__(self, input_file, tokenizer, msg_col=0, rsp_col=1, max_msg_len=100, max_rsp_len=30, run_mode=None,
                    architecture='matching_model', truncate=False):
        self.architecture = architecture
        self.truncate = truncate
        self.input_file = input_file
        self.msg_col = msg_col
        self.rsp_col = rsp_col
        self.max_msg_len = max_msg_len
        self.max_rsp_len = max_rsp_len
        self.tokenizer = tokenizer
        self.run_mode = run_mode
        # TODO: I think we should open the file only when it is needed, and thus in the __getitem__ function instead while checking for None
        # TODO: This is really bad and can hog up the resources by keeping file pointers alive for hundreds and thousands of files
        self.fh = open(input_file, 'r', encoding='utf-8')
        self.data_offset_map = []
        # Create the file map offset which will be used in each get item call
        self.create_file_offset()

    def __len__(self):
        return len(self.data_offset_map)


    def __getitem__(self, idx):
        """Reads a line from the file given by line index idx with at least two colums of message and response

        Arguments:
            idx -- Integer ID of the line to retrieve

        Returns:
            [List] -- [msg_ids (vocabified msg), rsp_ids (vocabified response), raw_message, raw_response]
        """
        try:
            # TODO: Inform the user when the idx is greater than the length of the map .. currently, we will end up pointing to the last line

            offset = self.data_offset_map[idx]
            self.fh.seek(offset, 0)
            line = self.fh.readline()

            return self.get_line_items(line)

        except Exception:
            # TODO: This is really bad .. we are just blindly eating up the exception here without even knowing the cause of it -- we should make the cause explicit and catch that exception
            print("SystemLog: Possible unicode reading exception in FileMRDataset")
            return ([], [], [], [])

    def truncate_bert_input(self, tokens, max_num_tokens):
        # Truncate input for TNLR/BERT models
        # Inputs: tokens is a sequence of tokens/ids returned from tokenizer, max_num_tokens is the maximum number of tokens to return
        # Outputs: tokens truncated to max_num_tokens in place
        while True:
            total_length = len(tokens)
            if total_length <= max_num_tokens:
                break

            trunc_tokens = tokens
            assert len(trunc_tokens) >= 1

            trunc_tokens.pop()

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

    def get_line_items(self, line):
        """
          Extract message/reply token ids from the given line
        """
        columns = line.rstrip().split("\t")
        msg_ids = [0, 0]
        rsp_ids = [0, 0]
        required_columns_in_file = max([self.msg_col, self.rsp_col])
        # this returns both columns but only converts msg or rsp based on run mode
        if len(columns) > required_columns_in_file:
            if self.run_mode == 'train' or self.run_mode == 'valid' or self.run_mode == 'gmr':
                if self.msg_col != -1:
                    message = self.tokenizer.tokenize(columns[self.msg_col])
                    msg_ids = self.tokenizer.convert_tokens_to_ids(message)
                if self.rsp_col != -1:
                    reply = self.tokenizer.tokenize(columns[self.rsp_col])
                    rsp_ids = self.tokenizer.convert_tokens_to_ids(reply)
            if self.run_mode == 'rsp_set':
                # This mode is used in the encode_responses() function of Inference Models
                if self.rsp_col != -1:
                    reply = self.tokenizer.tokenize(columns[self.rsp_col])
                    rsp_ids = self.tokenizer.convert_tokens_to_ids(reply)
            if self.run_mode == 'eval':
                # TODO: This block can be merged with the "train", "valid" and "gmr" block
                if self.msg_col != -1:
                    message = self.tokenizer.tokenize(columns[self.msg_col])
                    msg_ids = self.tokenizer.convert_tokens_to_ids(message)

            # Truncate
            # TODO: This method will not scale ... we should subclass this class for the BERT use-case
            if self.truncate:
                if self.architecture == "bert_matching_model":
                    if self.tokenizer.use_sentence_start_end:
                        msg_ids_copy = copy.deepcopy(msg_ids[1:-1])
                        self.truncate_bert_input(msg_ids_copy, self.max_msg_len-2)
                        msg_ids = [msg_ids[0]] + msg_ids_copy + [msg_ids[-1]]

                        rsp_ids_copy = copy.deepcopy(rsp_ids[1:-1])
                        self.truncate_bert_input(rsp_ids_copy, self.max_rsp_len-2)
                        rsp_ids = [rsp_ids[0]] + rsp_ids_copy + [rsp_ids[-1]]
                    else:
                        self.truncate_bert_input(msg_ids, self.max_msg_len)
                        self.truncate_bert_input(rsp_ids, self.max_rsp_len)

            if len(msg_ids) <= self.max_msg_len and len(rsp_ids) <= self.max_rsp_len:
                return (msg_ids, rsp_ids, columns[self.msg_col], columns[self.rsp_col])

        # TODO: Count the number of such instances either here or in the caller function (possibly different Counters for different reasons)
        return ([], [], [], [])

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


# TODO: This design of if-else conditions is not scalable -- we should be subclassing the base class for model specific changes
# TODO: Also, why is this not a function instead of a Class?
class CollateMRSequence():
    # Used for collateing tuples of the form (msg_vec, rsp_vec)
    # from the output of MRDataset.
    # Creates a padded tensor for message and reply indexes
    # Removes zero sequences
    # Add masks for BERT matching model.
    def __init__(self, architecture='matching_model'):
        self.architecture = architecture

    # TODO: __call__ function has a special meaning inside a class .. I'm not sure if we are overloading that function by design here
    def __call__(self, batch):
        batch = [b for b in batch if len(b[0]) > 0 and len(b[1]) > 0]
        if len(batch) < 1:
            return None

        batch_msg_ids, batch_rsp_ids, batch_messages, batch_replies = list(zip(*batch))
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

        msg_batch_tuple = (batch_msg_ids_padded, batch_msg_lengths)
        rsp_batch_tuple = (batch_rsp_ids_padded, batch_rsp_lengths)

        if self.architecture in ['bert_matching_model']:
            # mask padded tokens
            msg_mask = np.zeros([len(batch), max_msg_len], dtype=np.int32)
            rsp_mask = np.zeros([len(batch), max_rsp_len], dtype=np.int32)
            msg_mask = torch.LongTensor(msg_mask)
            rsp_mask = torch.LongTensor(rsp_mask)

            # segment id (0 for first sentence, 1 for second sentence)
            # TODO: consider BERT interaction model which need to concat M and R with different segment ids
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

        return msg_batch_tuple, rsp_batch_tuple, batch_messages, batch_replies
