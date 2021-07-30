"""
    Code for Model Paramaters:
    @author: Budhaditya Deb
"""
import sys
from retrieval_rs.models.common.model_params import HyperParameters

class ModelParams(HyperParameters):
    """
        Overrides defaults in Hyperparameters
        Adds additional parameters required for CVAE model
        TODO: check if these are set correctly for the parameters listed for each class
    """
    def __init__(self):
        super().__init__()
        self.add_matching_loss_reg = 0
        self.apply_mmr_ranking_alpha = 0.0
        self.bidirectional = True
        self.clip_grad_norm = 1
        self.cvae_h_dim = 512
        self.cvae_score_type = "recon_max_vote" # recon_max_vote, recon_rank_vote, recon_sum_vote, recon_w_KLd_max_vote
        self.cvae_z_dim = 512
        self.decay_matching_loss = 1.0
        self.decay_rate = 0.999
        self.decay_step = 5000
        self.do_constant_folding_in_onnx = False
        self.dropout = 0.2
        self.elbo_lambda = 1.0
        self.emb_dim = 512
        self.freeze_layers = None
        self.kept_layers_msg = None
        self.kept_layers_rsp = None
        self.learning_rate = 0.0000001
        self.lm_alpha = 1.0
        self.load_from = "Matching" # options are BertModel, Matching, tnlr
        self.max_metrics_plateau_for_early_stopping = 10
        self.num_cvae_samples = 1000
        self.num_f_zx_y_proj_layers = 0
        self.num_layers = 3 # rnn layers
        self.num_msg_fine_tune_layers = 0
        self.num_q_xy_z_proj_layers = 0
        self.pretrained_model_number = -1
        self.recon_loss_type = "SMLoss" # options SMLoss, SymSMLoss
        self.response_deduplication = True
        self.response_mapping = True
        self.rnn_hidden_dim = 512
        self.rsp_label_col = -1
        self.shared_base_encoders = False
        self.top_k = 15
        self.total_training_steps = 1000000
        self.train_msg_encoder = False
        self.train_rsp_encoder = False
        self.txt_encoder_type = 'BiLSTM' # options are BertModel (i.e. pretrained bert), BiLSTM, BertTNLR (pretrained Bert with Matching or load from TNLR)
        self.vocab_size = -1
        self.warmup_proportion = 0.0001

    def append_argparser(self, parser):
        # TODO: should do super and then add these
        parser.add_argument('--add_matching_loss_reg', required=False, type=float, default=0.0, help='Add a matching loss as a regularizer with coefficient')
        parser.add_argument('--apply_mmr_ranking_alpha', required=False, type=float, default=0.0, help='Set this to a vluae > 0.0 to to apply MMR reranking after Matching ')
        parser.add_argument('--cvae_score_type', required=False, default="recon_max_vote", help='Scoring method to estimate posterior probability: recon_max_vote, recon_rank_vote, recon_avg, recon_kld_max_vote, kld_max_vote')
        parser.add_argument('--decay_matching_loss', required=False, type=float, default=1.0, help='Add a matching loss as a regularizer with coefficient')
        parser.add_argument('--do_constant_folding_in_onnx', required=False, default=False, action='store_true', help='Whether to fold constants during ONNX conversion')
        parser.add_argument('--elbo_lambda', required=False, type=float, default=1, help='Temparature constant for KL_D Loss')
        parser.add_argument('--freeze_layers', required=False, default=None, help='Freeze specific layers')
        parser.add_argument('--kept_layers_msg', required=False, default=None, help='Select specific layers on msg side')
        parser.add_argument('--kept_layers_rsp', required=False, default=None, help='Select specific layers on rsp side')
        parser.add_argument('--num_cvae_samples', required=False, type=int, default=1000, help='number of CVAE samples to use for prediction')
        parser.add_argument('--num_f_zx_y_proj_layers', required=False, type=int, default=0, help='number of CVAE decoder layers')
        parser.add_argument('--num_msg_fine_tune_layers', required=False, type=int, default=0, help='number of layers to fine tune the msg encoder')
        parser.add_argument('--num_q_xy_z_proj_layers', required=False, type=int, default=0, help='number of cvae encoder layers')
        parser.add_argument('--pretrained_model_number', required=False, type=int, default=-1, help='The model number of the pretrained matching model file.')
        parser.add_argument('--recon_loss_type', required=False, default="SMLoss", help='Reconstrunction loss in VAE loss: SymSMLoss, SMLoss, CELoss')
        parser.add_argument('--response_deduplication', required=False, default=True, action='store_false', help='Setting this turns OFF response deduplication for prediction.')
        parser.add_argument('--response_mapping', required=False, default=True, action='store_false', help='Setting this turns OFF response mapping before prediction.')
        parser.add_argument('--rsp_label_col', required=False, type=int, default=-1, help='response label column typically available in GMR dataset')
        parser.add_argument('--top_k', required=False, type=int, default=15, help='top k number of predictions returned')
        parser.add_argument('--train_msg_encoder', required=False, default=False, action='store_true', help='Whether to train the msg side encoder or freeze it.')
        parser.add_argument('--train_rsp_encoder', required=False, default=False, action='store_true', help='Whether to train the rsp side encoder or freeze it.')
        parser.add_argument('--txt_encoder_type', required=False, default="BiLSTM", help='Text Encioder Type: BertModel, BertTNLR, BiLSTM')
        return parser
