"""
    Code for Matching Model from:
    Diversifying Reply Suggestions using a Matching-Conditional Variational Autoencoder: https://arxiv.org/abs/1903.10630
    Copyright: Microsoft Search, Assistance and Intelligence Team, 2020
    @author: Budhaditya Deb
"""
# coding=utf-8
import importlib
import pdb
import numpy as np
import random
import torch

from retrieval_rs.models.matching.model_params import ModelParams
from retrieval_rs.models.common.utils import get_path_to_pretrained_model
from retrieval_rs.models.common_model_lib.text_encoders import TextEncoder
from retrieval_rs.models.common_model_lib.sr_module import SRModule
from retrieval_rs.models.common_model_lib.model_factory import get_module_class_from_factory

# dummy comment to pop it into the PR

class MatchingModule(SRModule):
    """
        Matching Model as described in:
            Diversifying Reply Suggestions using a Matching-Conditional Variational Autoencoder: https://arxiv.org/abs/1903.10630
        Model with two independent encoders for Msg and Rsp, trained to minimize the dot product distance between the two encoding outputs
        This is defined to generalize the models defined in retrieval_rs/models/matching for MCVAE.
        MCVAE always requires two independent encoders for msg and rsp side.
        Thus original MAtching model with BiLSTM encoders cannot be used at it has a shared embedding layer.
        Args to set:
            --txt_encoder_type
            --pretrained_model_path
            --recon_loss_type
            --train_rsp_encoder
    """
    def __init__(self, params):
        super().__init__(params)
        # ------------ create and initialize the base encoders from pretrained model ------------
        self.create_base_encoders()
        module = importlib.import_module("retrieval_rs.models.common_model_lib.losses")
        self.recon_loss = getattr(module, self.params.recon_loss_type)(self.params)

    def create_base_encoders(self):
        """
            Creates the base encoders (BERT, BiLSTM etc.)
            Loads these from a pretrained Matching model if available (need to pass --load_from parameter)
            Loading from pretrained is usually when loading from a model which was trained with a different class structure
            E.g. BERT, TNLR Matching Models (since these had different class structures)
        """
        print("SystemLog: Creating Base encoders for Matching model")
        self.msg_encoder = TextEncoder(self.params, encoder_side='msg')
        self.rsp_encoder = TextEncoder(self.params, encoder_side='rsp')

        if self.params.load_from == "tnlr": # assume that BERT like model was trained as a matching
            """
                This condition assumes the Matching Model was trained using Class definitions in
                /models/matching/bert_matching_model.py
                Since this does not use the TextEncoder wrapper so it needs to be created.
            """
            if self.params.txt_encoder_type == "BertTNLR":
                matching_model_inst = get_module_class_from_factory('bert_matching_model')
                matching_model = matching_model_inst(self.params)

            if self.params.pretrained_model_path:
                path_to_model = get_path_to_pretrained_model(self.params)
                print("SystemLog: Loading Matching Model (%s, %s) from %s" %(self.params.load_from, self.params.txt_encoder_type, path_to_model))
                matching_model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
                self.msg_encoder = TextEncoder(self.params, encoder_inst=matching_model.msg_encoder)
                self.rsp_encoder = TextEncoder(self.params, encoder_inst=matching_model.rsp_encoder)

        self.msg_enc_out_dim = self.msg_encoder.enc_out_dim
        self.rsp_enc_out_dim = self.rsp_encoder.enc_out_dim

    def train_valid_forward(self, input_batch):
        """
            For pretrained txt encoders like BertModel have not been pretrained for matching we should
            train the msg_encoder but dont neccesaruly need to train the response encoder.
            Args:
                input_batch: as a list of dicts: batch_size[{'msg_batch_tuple', 'rsp_batch_tuple', 'batch_messages', 'batch_replies', 'batch_rsp_labels'}]
            Returns:
                X: Input encoding of size [batch_size, self.msg_enc_out_dim]
                Y: Input encoding of size [batch_size, self.rsp_enc_out_dim]
        """
        X = self.msg_encoder(input_batch['msg_batch_tuple'])
        if self.params.train_rsp_encoder is False: # usually with BERT
            with torch.no_grad():
                Y = self.rsp_encoder(input_batch['rsp_batch_tuple'])
        else:
            Y = self.rsp_encoder(input_batch['rsp_batch_tuple'])
        return (X, Y)

    def eval_forward(self, input_batch):
        """ Used for inference model
            Need to make sure that the infernce graph has been initialized to latest model
            For matching we can do constrained sampling by first getting the topk and then doing the CVAE samples using only the topk.
            The sampling this works with batch size of 1. To do larger batch sizes we encode the messages in batch mode,
            but do the sampling with batch sizes of 1.
            Args:
                input_batch: msg_batch_tuple, batch_messages
            Returns:
                scores_with_lm_pen: [batch_size, top_k]
                scores_with_lm_pen_topk_inds: [batch_size, top_k]
                msg_enc_final: [batch_size, output_encoding_size]
        """
        with torch.no_grad():
            if self.run_mode == 'eval':
                msg_enc_final = self.msg_encoder(input_batch['msg_batch_tuple'], self.enforce_sorted)
            if self.run_mode == 'export_onnx':
                # -------- input_batch is inputed as a tuple for consistency with QAS --------
                msg_enc_final = self.msg_encoder(input_batch, self.enforce_sorted)

            scores = torch.matmul(msg_enc_final, torch.transpose(self.rsp_encodings, 0, 1))
            penalized_lm_scores = self.lm_alpha * self.lm_scores
            scores_with_lm_pen_all = torch.add(scores, penalized_lm_scores)
            scores_with_lm_pen, scores_with_lm_pen_topk_inds = torch.topk(scores_with_lm_pen_all, self.top_k, dim=1)

        return scores_with_lm_pen, scores_with_lm_pen_topk_inds, msg_enc_final

    def loss(self, output):
        """
            SMLoss and SySMLoss returns loss for each item in batch. Here we just compute the mean across the batch
            Args:
                output from forward (X, Y)
                X: Input encoding of size [batch_size, self.msg_enc_out_dim]
                Y: Input encoding of size [batch_size, self.rsp_enc_out_dim]
            Returns:
                loss_dict: {loss, precision}
        """
        loss_dict = self.recon_loss(output)
        recon_loss = loss_dict['loss']
        recon_loss = torch.mean(recon_loss)
        loss_dict['loss'] = recon_loss
        return loss_dict

    def initialize_inference_graph(self, params, enforce_sorted=False):
        self.top_k = int(params.top_k)
        self.lm_alpha = params.lm_alpha
        self.msg_encoder.to(params.device)
        self.rsp_encoder.to(params.device)
        self.rsp_encodings, self.rsp_token_ids = self.encode_responses()
        self.enforce_sorted = enforce_sorted

