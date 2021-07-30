"""
    This file is the landing class for initiating runs for training.
    @author: Budhaditya Deb
"""

# coding=utf-8
import os
import sys
import torch
import numpy as np
import random


# ------------- add base directories -------
base_dir = os.path.abspath(os.path.dirname(__file__))
print(base_dir)
SR_DIR = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir, os.pardir))
print('SystemLog: Current retrieval_rs DIR is %s' % SR_DIR)
sys.path.insert(0, SR_DIR)

from retrieval_rs.models.common_model_lib.sr_model import SRModel
from retrieval_rs.models.common_model_lib.model_factory import get_params_class_from_factory, get_model_class_from_factory, module_architecture_dictionary
from retrieval_rs.models.common.model_params import  build_argparser

def set_model_parameters():
    """ Get command line parameters. Overwrite the default parameters if any command line parameters are present
        TODO: We should initialize parameters class based on the architecture. How do we do it?
    """
    print("SystemLog: Parsing arguments")
    # ----------- Get the architecture name -------------
    architecture = sys.argv[sys.argv.index("--architecture")+1]
    if architecture not in module_architecture_dictionary:
        print("SystemLog: Incorrect architecture provided please select from: matching_model, mcvae_model")
        raise ValueError("Incorrect architecture provided please select from: matching_model, mcvae_model")
    # ----------- Create self.params class from model factory -----------
    params_class = get_params_class_from_factory(architecture)
    params = params_class()
    parser = build_argparser() #TODO later do this with the super class Hypeparams
    parser = params.append_argparser(parser)
    args = parser.parse_args()
    # ----------- Assign any non-None values passed from command line -----------
    print("SystemLog: Set parameters to self.params class from args")
    params.set_args(args)
    print("SystemLog: ------------------------------------------")
    sys.stdout.flush()
    return params

def set_random_seed(params):
    # TODO: Add Function DocString
    manual_seed = params.manual_seed
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    # TODO: Does the following belong in this function?
    # Amp needs cudnn enabled
    if not params.fp16:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class SRDriver():
    """
        Driver class to run training, validation and export etc.
    """
    def __init__(self, params):
        print("SystemLog: Starting retrieval_rs Model")
        self.params = params
        # ---------------- Set manual seed for reproducibility ---------------
        if self.params.manual_seed != -1:
            print("SystemLog: Setting manual seed to %d"%self.params.manual_seed)
            set_random_seed(self.params)
        if self.params.model_output_dir:
            os.makedirs(self.params.model_output_dir, exist_ok=True)
            print("SystemLog: created model_output_dir", self.params.model_output_dir)
        self.initialize_cuda()
        """
            TODO: Add initialization from factory method here as more architectures are added
        """
        module_class_inst = get_model_class_from_factory(self.params.model_language_class)
        self.model = module_class_inst(self.params)

    def initialize_cuda(self):
         # ---------------- Set Cuda and default devices ----------------
        torch.cuda.empty_cache()
        if self.params.use_cuda:
            print("SystemLog: Trying to use GPU")
            if torch.cuda.is_available():
                self.params.device = torch.device("cuda")
            else:
                print("SystemLog: GPU is not available. Setting to CPU")
                self.params.device = torch.device("cpu")
                self.params.use_cuda = False
        else:
            self.params.device = torch.device("cpu")

    def run(self):
        # ---------------- Run model in different modes ----------------
        if self.params.run_mode == "train":
            self.model.train()
        elif self.params.run_mode == "compute_metrics":
            return self.model.compute_validation_metrics()
        if self.params.run_mode == "export_onnx":
            self.model.export_onnx()
        if self.params.run_mode == "eval":
            with torch.no_grad():
                self.model.eval() # used for predicting responses to a public set such as  ENRON

if "__main__" == __name__:
    try:
        params = set_model_parameters()
        driver = SRDriver(params)
        driver.run()
    except:
        scrub_exc_msg = True
        print("SystemLog: Raising exception manually")
        raise ValueError("SystemLog: Manual exception found")
