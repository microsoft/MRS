"""
    Code for Model Paramaters specific to MLTL Models
    @author: Budhaditya Deb
"""
import sys
from retrieval_rs.models.common.model_params import HyperParameters
from retrieval_rs.models.matching.model_params import ModelParams as MCVAEModelParams
import ast
import pdb

class ModelParams(MCVAEModelParams):
    def __init__(self):
        super().__init__()
        self.train_langs = 'en'
        self.test_langs = 'en'
        self.lm_alpha_multi = '0.5'
        self.sample_ratio = '1.0'
        self.sample_languages_uniformly = False

    def append_argparser(self, parser):
        parser = super().append_argparser(parser)
        parser.add_argument("--train_langs", type=str, default='en', help="languages: en,es")
        parser.add_argument("--test_langs", type=str, default='en', help="languages: en,es")
        parser.add_argument('--lm_alpha_multi', required=False, type=str, default='0.5', help='lm alpha for different languages')
        parser.add_argument('--sample_ratio', required=False, type=str, default='1.0', help='sample ratios for different languages')
        parser.add_argument('--sample_languages_uniformly', required=False, default=False, action='store_true', help='Overrides sampling ratios and samples languages uniformly')

        return parser

    def convert_multi_Lingual_args(self):
        # parse params for languages languages
        self.train_langs = [t for t in self.train_langs.split('_')]
        self.test_langs = [t for t in self.test_langs.split('_')]
        lm_alpha_multi = [float(l) for l in self.lm_alpha_multi.split('_')]


        if self.sample_languages_uniformly:
            # override input values as batches are sampled over languages instead of sampling ranges
            sample_ratio = [1.0] * len(self.train_langs)
        else:
            sample_ratio = [float(s) for s in self.sample_ratio.split('_')]

        self.sample_ratio = {}
        if self.run_mode == 'train':
            for index in range(len(self.train_langs)):
                self.sample_ratio[self.train_langs[index]] = sample_ratio[index]

        self.lm_alpha_multi = {}
        for index in range(len(self.test_langs)):
            self.lm_alpha_multi[self.test_langs[index]] = lm_alpha_multi[index]

    def set_args(self, args, verbose=False):
        super().set_args(args, verbose=verbose)
        self.convert_multi_Lingual_args()



