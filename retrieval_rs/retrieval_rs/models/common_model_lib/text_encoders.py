"""
    Definitions for Text encoders to be used in different models.
    Supports BiLSTM, Bert, mBert, TNLR and TULR
    @author: Budhaditya Deb
"""
# dummy comment to pop it into the PR

import json
import os
import pdb

import numpy as np
import torch

from transformers import XLMRobertaModel, BertModel, BertConfig
from retrieval_rs.models.common.utils import to_cuda
from retrieval_rs.models.common_model_lib.model_factory import get_module_class_from_factory
from retrieval_rs.models.common.utils import get_path_to_pretrained_model

# TODO: Copied SeqEncoder class from matching.matching_model.py. Remove it later from that file

class TextEncoder(torch.nn.Module):
    """
        Wrapper for different Text encoders used in the Matching and CVAE models. Currently supports
        - BiLSTM: Uses the BiLSTMTextEncoder
        - BertModel, mBERT: Uses pre-trained BertModel from Huggingface repo and checkpoint
        - BertTNLR, TULR: Has TNLR specific changes, mostly with different config for msg and rsp encoders
        Args to set:
            --txt_encoder_type
            --pretrained_model_path
            --pretrained_model_file
    """
    def __init__(self, params, encoder_side='msg', encoder_inst=None):
        super().__init__()
        self.params = params
        if encoder_inst != None:
            """
                This option will typically be used for instantiating the TextEncoder from a Pretrained Matching model. For example while training the MCVAE,
                we copy the encoder from the saved Matching model and simply instantiate it here. This is for consistency w.r.t to the texe_encoder class as seen by MAtching/MCVAE classes
            """
            self.encoder = encoder_inst

            if self.params.txt_encoder_type in ['BertModel', 'BertTNLR']:
                self.enc_out_dim = self.encoder.encoder.layer[0].output.dense.weight.shape[0]
                self.input_names = ['msg_id', 'msg_type_id', 'msg_mask']

        else:
            """
                These options will typically be used for instantiating the TextEncoder for training a Matching/MCVAE Model
                by initializing from a pretrained model.
                For BiLSTM, it assumes no pretrained layers and instantiates it with random parameters. (Can add option later)
                For BertModel, BertTNLR, TULR the encoder is loaded from a pretrained model file.
            """
            if self.params.txt_encoder_type in ['BertTNLR', 'TULR']:
                self.init_bert_tnlr(encoder_side)
            if self.params.txt_encoder_type in ['XLMR']:
                self.init_xlmr()
            if self.params.txt_encoder_type in ['BertModel', 'MBert']:
                self.init_bert()
            if self.params.txt_encoder_type in ['BertModel', 'MBert', 'BertTNLR', 'TULR', 'XLMR']:
                self.init_layers(encoder_side)

    def init_bert_tnlr(self, encoder_side):
        """
            This option is used for the TNLR/TULR checkpoint which can have different number of layers in the msg and rsp side.
            This uses a PT model file.
        """
        config_file = os.path.join(self.params.pretrained_model_path, "config.json")
        print("SystemLog: Loading TNLR-BERT Config from %s" %config_file)
        self.config = json.load(open(os.path.join(self.params.pretrained_model_path, "config.json"), 'r', encoding='utf-8'))

        if encoder_side == 'msg':
            self.txt_config = BertConfig(**self.config["bert_msg_config"])
        if encoder_side == 'rsp':
            self.txt_config = BertConfig(**self.config["bert_rsp_config"])
        self.encoder = BertModel(self.txt_config)
        self.enc_out_dim = self.encoder.encoder.layer[0].output.dense.weight.shape[0]
        self.input_names = ['msg_id', 'msg_type_id', 'msg_mask']

        try:
            pretrained_file = get_path_to_pretrained_model(self.params)
            if self.params.run_mode == "train" and pretrained_file is not None:
                encoder_state_dict = torch.load(pretrained_file, map_location=torch.device("cpu"))
                self.encoder.load_state_dict(encoder_state_dict, strict=False)
        except:
            print("SystemLog: WARNING !!! Pretrained model file for BertTNLR encoder does not exist Initialized to Random.")

    def init_xlmr(self):
        """
            This option is used for the XLMR with HF's default loader.
        """
        print("SystemLog: Loading XLMR model from %s" % self.params.pretrained_model_path)
        self.encoder = XLMRobertaModel.from_pretrained(self.params.pretrained_model_path)
        self.enc_out_dim = self.encoder.encoder.layer[0].output.dense.weight.shape[0]
        self.input_names = ['msg_id', 'msg_type_id', 'msg_mask']

    def init_bert(self):
        """
            Current MBert does not have a PT version so load using HF's default loader with default 12 layers on each side.
            TODO: Create PT versions of the model files before loading.
        """
        print("SystemLog: Loading MBERT model from %s"%self.params.pretrained_model_path)
        self.encoder = BertModel.from_pretrained(self.params.pretrained_model_path)
        self.enc_out_dim = self.encoder.encoder.layer[0].output.dense.weight.shape[0]
        self.input_names = ['msg_id', 'msg_type_id', 'msg_mask']

    def init_layers(self, encoder_side):
        """
            Used for selecting and freezing specific layers in Bert like models
            Args to set:
                - kept_layers_msg
                - kept_layers_rsp
                - freeze_layers
            Examples:
                --kept_layers_msg 0_4_8_12_16_20
                --kept_layers_rsp 0_2_4_6_8_10_12_14_16_18_20_22
                --freeze_layers embedding#transformer_0_1
        """
        if encoder_side == 'msg':
            if self.params.kept_layers_msg is not None:
                cross_layer_init(self.params.kept_layers_msg, self.encoder)
        if encoder_side == 'rsp':
            if self.params.kept_layers_rsp is not None:
                cross_layer_init(self.params.kept_layers_rsp, self.encoder)

        if self.params.freeze_layers is not None:
            freeze_net(self.params.freeze_layers, self.encoder)

    def forward(self, txt_tuple, enforce_sorted=False):
        """
            Args:
                txt_tuple (BiLSTM): [txt_batch_tensor, txt_batch_length]
                txt_tuple (BertModel/MBert,BertTNLR/TULR): [batch_txt_ids_padded, txt_type_ids, txt_mask]
            Returns:
                encoding of dimentions: [batch_size, self.enc_out_dim]
        """

        if self.params.txt_encoder_type in ['BertModel', 'BertTNLR', 'MBert', 'TULR', 'XLMR']:
            msg_ids_padded_tensor, msg_type_ids_tensor, msg_mask_tensor = txt_tuple
            if self.params.use_cuda:
                msg_ids_padded_tensor, msg_type_ids_tensor, msg_mask_tensor = to_cuda(msg_ids_padded_tensor, msg_type_ids_tensor, msg_mask_tensor)
            txt_enc_final = self.encoder(msg_ids_padded_tensor, token_type_ids=msg_type_ids_tensor, attention_mask=msg_mask_tensor)[1]
        return txt_enc_final

def freeze_net(freeze_layers, net):
    """
        freeze part of mbert/bert/tnlr/tulr model network
        Args:
            - freeze_layers: params string with layer names to freeze, e.g. embedding#transformer_0_2 will freeze word
                embedding layer and first 3 layers of transformer encoder
            - net: net contains layers to be freezed
        Returns:
            partially freezed net
    """
    for layer_config in freeze_layers.split('#'):
        if layer_config.lower() == 'embedding':
            for p in net.embeddings.parameters():
                p.requires_grad = False
            print("SystemLog: ------------------Froze bert embedding layer.------------------")
        elif 'transformer' in layer_config:
            transformer_freeze_upper = int(layer_config.split('_')[2])
            transformer_freeze_lower = int(layer_config.split('_')[1])
            assert transformer_freeze_lower <= transformer_freeze_upper
            for n, encoder_layer in enumerate(net.encoder.layer):
                if n >= transformer_freeze_lower and n <= transformer_freeze_upper:
                    for p in encoder_layer.parameters():
                        p.requires_grad = False
            print("SystemLog: ------------------Froze upper [%s] Layers.------------------"%transformer_freeze_upper)
            print("SystemLog: ------------------Froze lower [%s] Layers.------------------"%transformer_freeze_lower)
    return net

def unfreeze_net(net):
    """
    Unfreezing all net to make all params trainable
    Args:
        net: net to unfreeze
    Returns:
        unfreezed net
    """
    print("SystemLog: Unfreezing all net to make all params trainable")
    for p in net.parameters():
        p.requires_grad = True

    return net

def cross_layer_init(kept_layers, net):
    """
        Args:
            param kept_layers: layers chosen to init, e.g. 0_2_4_6_8_10
            param msg_encoder: net to cross-layer init
        Returns:
            cross-layer inited net
    """
    layers = kept_layers.split('_')
    layers = [int(x) for x in layers]
    new_encoder_structure = []

    if (len(net.encoder.layer) < max(layers) + 1) or (min(layers) + 1 < 0):
        raise Exception("Designated layers number out of index.")

    for n, encoder_layer in enumerate(net.encoder.layer):
        if n in layers:
            new_encoder_structure.append(encoder_layer)
    print("SystemLog: ------------------Cross Layers [%s] Initialized.------------------"%layers)


    net.encoder.layer = torch.nn.ModuleList(new_encoder_structure)
    return net
