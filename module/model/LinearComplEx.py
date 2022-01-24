import torch
import torch.nn as nn
from .Model import Model

class LinearComplEx(Model):
    def __init__(self, ent_tot, rel_tot, ent_dim=100, rel_dim=100):
        super(LinearComplEx, self).__init__(ent_tot, rel_tot)

        self.ent_dim = ent_dim
        self.rel_dim = rel_dim

        self.ent_re_embeddings = nn.Embedding(self.ent_tot, self.ent_dim)
        self.ent_im_embeddings = nn.Embedding(self.ent_tot, self.ent_dim)
        self.rel_re_embeddings = nn.Embedding(self.rel_tot, self.rel_dim)
        self.rel_im_embeddings = nn.Embedding(self.rel_tot, self.rel_dim)

        self.re_wrh = nn.Embedding(self.rel_tot, self.rel_dim, max_norm=self.rel_dim, norm_type=1)
        self.im_wrh = nn.Embedding(self.rel_tot, self.rel_dim, max_norm=self.rel_dim, norm_type=1)

        self.re_wrt = nn.Embedding(self.rel_tot, self.rel_dim, max_norm=self.rel_dim, norm_type=1)
        self.im_wrt = nn.Embedding(self.rel_tot, self.rel_dim, max_norm=self.rel_dim, norm_type=1)

        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

        nn.init.zeros_(self.re_wrh.weight)
        nn.init.zeros_(self.im_wrh.weight)
        nn.init.zeros_(self.re_wrt.weight)
        nn.init.zeros_(self.im_wrt.weight)

    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return torch.sum(h_re * t_re * r_re + h_im * t_im * r_re + h_re * t_im * r_im - h_im * t_re * r_im, -1)

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)

        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)

        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)

        wh_re = self.re_wrh(batch_r)
        wh_im = self.im_wrh(batch_r)

        wt_re = self.re_wrt(batch_r)
        wt_im = self.im_wrt(batch_r)

        _h_re = wh_re * h_re
        _h_im = wh_im * h_im

        _t_re = wt_re * t_re
        _t_im = wt_im * t_im 

        score = self._calc(_h_re, _h_im, _t_re, _t_im, r_re, r_im)

        return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)

        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)

        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)

        wh_re = self.re_wrh(batch_r)
        wh_im = self.im_wrh(batch_r)

        wt_re = self.re_wrt(batch_r)
        wt_im = self.im_wrt(batch_r)

        regul = (torch.mean(wh_re ** 2) + torch.mean(wh_im ** 2) + torch.mean(wt_re ** 2) + torch.mean(wt_im ** 2) + torch.mean(h_re ** 2) + torch.mean(h_im ** 2) + torch.mean(t_re ** 2) + torch.mean(t_im ** 2) + torch.mean(r_re ** 2) + torch.mean(r_im ** 2)) / 10

        return regul

    
    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()