"""
    Code for common SR model class.
    @author: Budhaditya Deb
"""

import os
import pdb
import torch
from torch.utils.data import DataLoader

try:
    import apex
except ImportError as e:
    print("SystemLog: Apex not found")

from retrieval_rs.models.common.response_set import ResponseSet
from retrieval_rs.models.common.utils import get_tokenizer
# TODO: Later replace this by a factory method of getting the dataset and collate function
from retrieval_rs.models.matching.data_processing import CollateMRSequence, get_dataset

class SRModule(torch.nn.Module):
    """
        This is used as an abstract class with some basic definitions. All SR model classes derive from this.
        Includes some basic classes such as reading the response set etc.
        These are common to all the models.
        Args to set:
            --rsp_input_dir
            --rsp_mapping_input_dir
            --run_mode
            --lm_alpha
            --pretrained_model_path
            --pretrained_model_number

        Subclasses need to define the following functions
        or override this forward if there is a single forward for different models
            - train_valid_forward
            - eval_forward
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.run_mode = params.run_mode # options are train,, valid, eval, inference
        self.init_response_set()
        self.infer_language = None
        self.output_names = ['scores_with_lm_pen', 'top_k_ids', 'scores']

    def init_response_set(self):
        # TODO: Override for MLTL models to have multiple response sets
        rsp_dataset = get_dataset(self.params, run_mode="rsp_set")
        self.rsp_dataloader = DataLoader(rsp_dataset, batch_size=self.params.batch_size,
                                    shuffle=False, num_workers=0, collate_fn=CollateMRSequence(self.params))
        self.response_set = ResponseSet(self.params)
        self.lm_scores = torch.FloatTensor(self.response_set.lm_scores_list)
        if self.params.use_cuda:
            self.lm_scores = self.lm_scores.cuda()
        if self.params.fp16:
            if self.params.amp_opt_lvl == 'O2':
                self.lm_scores = self.lm_scores.half()

    def forward(self, input_batch):
        """
            Forward operates in different modes for Train/Validation and Eval/Inference
            For eval/inference, it is required that the we initialize the inference graph to the latest version of the model.
            initialize_inference_graph should be called before forwarding the model in eval mode.
            TODO: define the eval function for models which overrides the Module.eval()
        """
        if self.run_mode in ["train", "valid"]:
            return self.train_valid_forward(input_batch)
        if self.run_mode in ['eval', "export_onnx"]:
            if self.params.fp16 and self.params.amp_opt_lvl == 'O1':
                """ for fp16 model we want the inference to use standard FP32
                    However apex.amp.disable_casts() only works with Opt level O1
                """
                with apex.amp.disable_casts():
                    return self.eval_forward(input_batch)
            else:
                return self.eval_forward(input_batch)

    def set_lm_alpha(self, lm_alpha):
        """
            Set the Language model penalty factor used in Inference
        """
        self.lm_alpha = lm_alpha

    def print_model_dict(self):
        print('SystemLog: ---------------------------------------------------------')
        print("SystemLog: Defined Matching Model with state_dict:")
        for param_tensor in self.state_dict():
            print('SystemLog:', param_tensor, "\t", self.state_dict()[param_tensor].size())
        print('SystemLog: ---------------------------------------------------------')

    def set_run_mode(self, run_mode="train", enforce_sorted=False):
        """
            Sets run mode to train/valid or eval.
            For eval, the inference graph needs to be always initialized.
        """
        self.run_mode = run_mode
        self.enforce_sorted = enforce_sorted

        if self.run_mode == "eval" or self.run_mode == "export_onnx":
            self.initialize_inference_graph(self.params, enforce_sorted=self.enforce_sorted)

    def encode_responses(self):
        """
            - Read the responses from the response set
            - Forward through the current trained model rsp encoder side
            - Return the response encoding
        """
        if self.params.local_rank == 0:
            print("SystemLog: ---------------------------------------------------------")
            print("SystemLog: Encoding Response Set for languge [%s]"%self.infer_language)
        rsp_embedding = []
        rsp_token_ids = []
        for _, sample_batched in enumerate(self.rsp_dataloader):
            if sample_batched != None:
                if len(sample_batched['rsp_batch_tuple'][0]) >= 1:
                    with torch.no_grad():
                        rsp_enc_final = self.rsp_encoder(sample_batched['rsp_batch_tuple'])
                    rsp_embedding.extend(rsp_enc_final.tolist())
                    rsp_token_ids.extend(sample_batched['rsp_batch_tuple'][0].tolist())
        if self.params.local_rank == 0:
            print("SystemLog: Completed Encoding %d Responses for languge [%s]"%(len(rsp_embedding), self.infer_language))
            print("SystemLog: ---------------------------------------------------------")
        # --------- strip token id of zeros since the output is padded ---------
        for index in range(len(rsp_token_ids)):
            rsp_token_ids[index] = [id for id in rsp_token_ids[index] if id != 0]

        rsp_embedding = torch.FloatTensor(rsp_embedding)

        if self.params.use_cuda:
            rsp_embedding = rsp_embedding.cuda()

        if self.params.fp16 == True:
            if self.params.amp_opt_lvl == 'O2':
                rsp_embedding = rsp_embedding.half()

        return rsp_embedding, rsp_token_ids
