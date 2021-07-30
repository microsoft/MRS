"""
    Code for registering new model sand modules
    @author: Budhaditya Deb
"""

import importlib
import pdb
"""
    Put the list of model architectures here.
    module: the import module with definition of the architecture class
    name: class name for the architecture.
    data_loader_module: import module of the definition for dataloader class used
"""
# TODO: Replace this by a JSON File later?
module_architecture_dictionary = {
                    'matching_model':
                        {"module":"retrieval_rs.models.matching.matching_module",
                        "name":"MatchingModule",
                        "dataloader_module":"retrieval_rs.models.matching.data_processing",
                        "params_module":"retrieval_rs.models.matching.model_params"
                        },
                    'bert_matching_model':
                        {"module":"retrieval_rs.models.matching.bert_matching_model",
                        "name":"MatchingModel",
                        "dataloader_module":"retrieval_rs.models.common.data_processing",
                        "params_module":"retrieval_rs.models.common.model_params"},
                    'matching_mltl_model':
                        {"module":"retrieval_rs.models.sr_mltl.matching_mltl_module",
                        "name":"MatchingMLTLModule",
                        "dataloader_module":"retrieval_rs.models.matching.data_processing",
                        "params_module":"retrieval_rs.models.sr_mltl.model_params"},
                    }

model_class_dictonary = {
                            'mono_lingual':
                                {"module":"retrieval_rs.models.common_model_lib.sr_model",
                                "name":"SRModel"},
                            'multi_lingual':
                                {"module":"retrieval_rs.models.sr_mltl.sr_mltl_model",
                                "name":"SRMLTLModel"}
                        }

def get_params_class_from_factory(architecture):
    if architecture not in module_architecture_dictionary:
        raise ValueError("SystemLog: %s architecture currently not supported. Please provide a supported architecture from %s"%(architecture, list(module_architecture_dictionary.keys())))
    module = module_architecture_dictionary[architecture]["params_module"]
    module_inst = importlib.import_module(module)
    params_class = getattr(module_inst, 'ModelParams')
    return params_class

def get_model_class_from_factory(model_lang_class):
    # TODO MLTL add MLTL SR  option here and get this from model factory
    if model_lang_class not in model_class_dictonary:
        raise ValueError("SystemLog: %s architecture currently not supported. Please provide a supported architecture from %s"%(model_lang_class, list(model_class_dictonary.keys())))
    module = model_class_dictonary[model_lang_class]["module"]
    name = model_class_dictonary[model_lang_class]["name"]
    module_inst = importlib.import_module(module)
    model_class = getattr(module_inst, name)
    return model_class


def get_module_class_from_factory(architecture):
    """
        get model instance using the passed model architecture name in params
    """
    if architecture not in module_architecture_dictionary:
        raise ValueError("SystemLog: %s architecture currently not supported. Please provide a supported architecture")
    module = module_architecture_dictionary[architecture]["module"]
    name = module_architecture_dictionary[architecture]["name"]
    module_inst = importlib.import_module(module)
    model_class = getattr(module_inst, name)
    return model_class

# TODO: Similarly get Data loader, tokenizer etc.
