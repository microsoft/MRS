"""
    Code for Multi-Lingual version of MAtching and MCVAE Model from /models/mcvae/ folder
    @author: Budhaditya Deb
"""
# coding=utf-8
import pdb

import torch
import numpy as np

from retrieval_rs.models.matching.matching_module import MatchingModule
from retrieval_rs.models.sr_mltl.sr_mltl_model import SRMLTLModule

class MatchingMLTLModule(MatchingModule, SRMLTLModule):
    """
        Inherits everything from the base classes and adds additional language specific parameters.
    """
    def __init__(self, params):
        super().__init__(params)

    def initialize_inference_graph(self, params, enforce_sorted):
        self.top_k = params.top_k
        self.lm_alpha = params.lm_alpha_multi[self.infer_language]
        self.lm_scores = self.lm_scores_multi[self.infer_language]
        self.rsp_dataloader = self.rsp_dataloaders_multi[self.infer_language]
        self.response_set = self.response_set_multi[self.infer_language]
        self.enforce_sorted = enforce_sorted
        self.rsp_encodings, self.rsp_token_ids = self.encode_responses()
