"""
    Defines wrappers around different losses used for SR.
    @author: Budhaditya Deb
"""
import importlib
import pdb
import numpy as np
import torch

class SMLoss(torch.nn.Module):
    """
        Computes tje Softmax of exp(dot product) potential. The one sided version of the the sym_loss
        For CVAE the recon loss by default is SMLoss instead of the SymSMLoss as we always keep the response side fixed.
        Args to set:
            --recon_loss_type SMLoss
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
    def forward(self, output):
        """
            Args:
                output from forward (X, Y)
                X: Input encoding of size [batch_size, self.msg_enc_out_dim]
                Y: Input encoding of size [batch_size, self.rsp_enc_out_dim]
            Returns:
                loss_dict: {loss[batch_size], precision[batch_size]}
        """
        msg_enc, rsp_enc = output
        # TODO: Try to scale the dot product with sqrt(dimentions of emcoding)
        s_x_y = torch.sum(msg_enc * rsp_enc, dim=1)
        s_x = torch.matmul(msg_enc, torch.transpose(rsp_enc, 0, 1))
        lse_s_x = torch.logsumexp(s_x, 1, keepdim=False)
        loss = lse_s_x.add(-s_x_y)
        precision = self.precision(s_x)
        return {'loss': loss, 'precision': precision}

    def precision(self, s_x):
        """
            precision is defined as the number of of times the argmarx of the score matches the corresponding rsp index or P@1
            Args:
                s_x: dot product of msg and rsp vectors [batch_size]
            Returns:
                precision[batch_size]
        """
        if self.params.use_cuda == True:
            s_x = s_x.cuda()
        max_indexes = torch.argmax(s_x, dim=1)
        target_indexes_pt = torch.LongTensor(np.arange(len(max_indexes)))
        if self.params.use_cuda:
            target_indexes_pt = target_indexes_pt.cuda()
        precision = torch.mean(torch.eq(max_indexes, target_indexes_pt).type(dtype=torch.float32))
        return precision

class SymSMLoss(SMLoss):
    """
        Softmax of exp(dot product) potential. The two sided version of the the SMLoss
        From eq. 1 in https://arxiv.org/abs/1903.10630)
        Args to set:
            --recon_loss_type SymSMLoss
    """
    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def forward(self, output):
        """
            Args:
                output from forward (X, Y)
                X: Input encoding of size [batch_size, self.msg_enc_out_dim]
                Y: Input encoding of size [batch_size, self.rsp_enc_out_dim]
            Returns:
                loss_dict: {loss[batch_size], precision[batch_size]}
        """
        msg_enc, rsp_enc = output
        s_x_y = torch.sum(msg_enc * rsp_enc, dim=1)
        s_x = torch.matmul(msg_enc, torch.transpose(rsp_enc, 0, 1))
        s_y = torch.matmul(rsp_enc, torch.transpose(msg_enc, 0, 1))
        s_x_s_y = torch.cat((s_x, s_y), dim=1)
        lse_s_x_s_y = torch.logsumexp(s_x_s_y, 1, keepdim=False)
        sym_loss = lse_s_x_s_y.add(-s_x_y)
        # -------- compute precision@1 for the m-r pairs --------
        precision = self.precision(s_x)
        return {'loss': sym_loss, 'precision': precision}
