from math import inf
from re import S
import torch
import numpy as np


class MathOps():
    def __init__(self, parent):
        self.parent = parent
        self.feature_dim = parent.feature_dim
    
    @torch.no_grad()
    def compute_batched_activation(self, Z):
        if self.parent.c == 0:
            self.parent.Gamma = torch.empty(Z.shape[0], 0, device=self.parent.device)
            return self.parent.Gamma

        c = self.parent.c
        mask = self.parent.feature_mask
        Zm  = Z[:, mask]
        mum = self.parent.mu[:c][:, mask]

        if self.parent.cond_cov:
            Sinv = self.parent.S_inv[:c][:, mask][:, :, mask]
            diff = Zm.unsqueeze(1) - mum
            tmp  = torch.einsum('bcd,cde->bce', diff, Sinv)
            d2   = (tmp * diff).sum(-1)
        else:
            W        = torch.diagonal(self.parent.S_inv[:c], dim1=1, dim2=2)[:, mask]
            term_x2  = (Zm * Zm) @ W.T
            W_mu_T   = (W * mum).T
            term_xmu = Zm @ W_mu_T
            term_mu2 = (mum * mum * W).sum(dim=1)
            d2       = term_x2 - 2.0 * term_xmu + term_mu2.unsqueeze(0)

        self.parent.Gamma = torch.exp_(-d2 * (1.0 / float(self.parent.feature_dim)))
        return self.parent.Gamma