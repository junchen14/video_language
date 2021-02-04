from collections import OrderedDict

import numpy as np
import torch
from easydict import EasyDict
from torch import nn
import torch.nn.functional as F
import utils


class XLModel:
    def __init__(self, cfg: EasyDict, use_cuda: bool, use_multi_gpu: bool):
        self.use_cuda = use_cuda
        self.use_multi_gpu = use_multi_gpu
        self.model_list = []
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu")

        # video pooler
        self.net_video_pooler = Transformer(
            cfg.net_video_pooler, cfg.dataset.feature_dim)
        self.net_video_pooler = self.to_device_fn(self.net_video_pooler)
        # video sequencer

        self.net_video_sequencer = Transformer(
            cfg.net_video_sequencer, cfg.net_video_pooler.output_dim)
        self.net_video_sequencer = self.to_device_fn(self.net_video_sequencer)

        # text pooler

        self.net_text_pooler = Transformer(
            cfg.net_text_pooler, cfg.text_encoder.feature_dim)
        self.net_text_pooler = self.to_device_fn(self.net_text_pooler)

        # video pooler

        self.net_text_sequencer = Transformer(
            cfg.net_text_sequencer, cfg.net_text_pooler.output_dim)
        self.net_text_sequencer = self.to_device_fn(self.net_text_sequencer)
        self.model_list = [self.net_video_pooler, self.net_video_sequencer,
                           self.net_text_pooler, self.net_text_sequencer]



    def encode_video(
            self, vid_frames, vid_frames_mask, vid_frames_len,
            clip_num, clip_frames, clip_frames_len, clip_frames_mask):
        # print(vid_frames.shape, vid_frames_mask.shape, vid_frames_len.shape, clip_num.shape, clip_frames.shape,clip_frames_len.shape, clip_frames_mask.shape)

        # compute video context


        '''
        :param vid_frames:
        :param vid_frames_mask:
        :param vid_frames_len:
        :param clip_num:
        :param clip_frames:
        :param clip_frames_len:
        :param clip_frames_mask:
        :return:
        '''

        vid_context = self.net_video_pooler(
            vid_frames, vid_frames_mask, vid_frames_len, None)
        if self.cfg.net_video_sequencer.use_context:
            if self.cfg.net_video_sequencer.name == "rnn":
                vid_context_hidden = vid_context.unsqueeze(0)
                vid_context_hidden = vid_context_hidden.repeat(
                    self.cfg.net_video_sequencer.num_layers, 1, 1)
            elif self.cfg.net_video_sequencer.name == "atn":
                vid_context_hidden = vid_context
            else:
                raise NotImplementedError
        else:
            vid_context_hidden = None
        '''
        first step: use a transformer to learn all the videos frames features
        and do the attention pooling
        '''

        clip_emb = self.net_video_pooler(
            clip_frames, clip_frames_mask, clip_frames_len, None)
        batch_size = len(clip_num)
        max_clip_len = torch.max(clip_num)
        clip_feat_dim = self.cfg.net_video_pooler.output_dim
        clip_emb_reshape = torch.zeros(
            (batch_size, max_clip_len, clip_feat_dim))
        clip_emb_mask = torch.zeros((batch_size, max_clip_len))
        clip_emb_lens = torch.zeros((batch_size,))
        if self.use_cuda:
            clip_emb_reshape = clip_emb_reshape.cuda(non_blocking=True)
            clip_emb_mask = clip_emb_mask.cuda(non_blocking=True)
            clip_emb_lens = clip_emb_lens.cuda(non_blocking=True)
        pointer = 0
        for batch, clip_len in enumerate(clip_num):
            clip_emb_reshape[batch, :clip_len, :] =\
                clip_emb[pointer:pointer + clip_len, :]
            clip_emb_mask[batch, :clip_len] = 1
            clip_emb_lens[batch] = clip_len
            pointer += clip_len

        '''
        second step: feed all the clip featuers into a transformer model, and learn its features.
        and also do the attention pooling
        '''

        # compute video embedding
        vid_emb = self.net_video_sequencer(
            clip_emb_reshape, clip_emb_mask, clip_num, vid_context_hidden)

        '''
        third step: feed the video and clip features into antoher transformer, and learn the interaction between clips and videos
        '''

        return (vid_emb, clip_emb, vid_context,
                clip_emb_reshape, clip_emb_mask, clip_emb_lens)

    def encode_paragraph(
            self, par_cap_vectors, par_cap_mask, par_cap_len,
            sent_num, sent_cap_vectors, sent_cap_mask, sent_cap_len):
        # compute paragraph context

        '''

        :param par_cap_vectors:
        :param par_cap_mask:
        :param par_cap_len:
        :param sent_num:
        :param sent_cap_vectors:
        :param sent_cap_mask:
        :param sent_cap_len:
        :return:
        '''
        par_context = self.net_text_pooler(
            par_cap_vectors, par_cap_mask, par_cap_len, None)
        if self.cfg.net_text_sequencer.use_context:
            if self.cfg.net_text_sequencer.name == "rnn":
                par_gru_hidden = par_context.unsqueeze(0)
                par_gru_hidden = par_gru_hidden.repeat(
                    self.cfg.net_text_sequencer.num_layers, 1, 1)
            elif self.cfg.net_text_sequencer.name == "atn":
                par_gru_hidden = par_context
            else:
                raise NotImplementedError
        else:
            par_gru_hidden = None


        '''
        step1: learn the feature representations of the whole paragraph
        '''

        # compute sentence embedding
        sent_emb = self.net_text_pooler(
            sent_cap_vectors, sent_cap_mask, sent_cap_len, None)
        batch_size = len(sent_num)
        sent_feat_dim = self.cfg.net_text_pooler.output_dim
        max_sent_len = torch.max(sent_num)
        sent_emb_reshape = torch.zeros(
            (batch_size, max_sent_len, sent_feat_dim))
        sent_emb_mask = torch.zeros((batch_size, max_sent_len))
        sent_emb_lens = torch.zeros((batch_size,))
        if self.use_cuda:
            sent_emb_reshape = sent_emb_reshape.cuda(non_blocking=True)
            sent_emb_mask = sent_emb_mask.cuda(non_blocking=True)
            sent_emb_lens = sent_emb_lens.cuda(non_blocking=True)
        pointer = 0
        for batch, sent_len in enumerate(sent_num):
            sent_emb_reshape[batch, :sent_len, :] =\
                sent_emb[pointer:pointer + sent_len, :]
            sent_emb_mask[batch, :sent_len] = 1
            sent_emb_lens[batch] = sent_len
            pointer += sent_len

        '''
        step 2: learn the feature representation of caption for each clip 
        '''

        # compute paragraph embedding
        par_emb = self.net_text_sequencer(
            sent_emb_reshape, sent_emb_mask, sent_num, par_gru_hidden)

        '''
        step 3: learn the feature representation of the whole video
        '''
        return (par_emb, sent_emb, par_context,
                sent_emb_reshape, sent_emb_mask, sent_emb_lens)

    def eval(self):
        for model in self.model_list:
            model.eval()
        torch.set_grad_enabled(False)

    def train(self):
        for model in self.model_list:
            model.train()
        torch.set_grad_enabled(True)

    def to_device_fn(self, model):
        if self.use_multi_gpu:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        return model

    def get_params(self):
        params = []
        for model in self.model_list:
            params_dict = OrderedDict(model.named_parameters())
            _params = []
            for key, value in params_dict.items():
                _params += [{
                    'params': value
                }]
            params.extend(_params)
        return params

    def load_checkpoint(self, ckpt: str):
        state = torch.load(str(ckpt))
        for i, m in enumerate(self.model_list):
            state_dict = state[i]
            if self.use_multi_gpu:
                newer_state_dict = OrderedDict()
                for key, val in state_dict.items():
                    assert not key.startswith("module.")
                    new_key = "module." + key
                    newer_state_dict[new_key] = val
                m.load_state_dict(newer_state_dict)
            else:
                m.load_state_dict(state_dict)
            i += 1

    def save_checkpoint(self, ckpt: str):
        model_states = []
        for m in self.model_list:
            state_dict = m.state_dict()
            if self.use_multi_gpu:
                new_state_dict = OrderedDict()
                for key, val in state_dict.items():
                    assert key.startswith("module.")
                    new_key = key[7:]
                    new_state_dict[new_key] = val
                model_states.append(new_state_dict)
            else:
                model_states.append(state_dict)
        torch.save(model_states, str(ckpt))



























class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]



class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm = False):
        super(PositionwiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):

        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp+ core_out)

        return output




class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm = False):
        super(MultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias = False)
        self.kv_net = nn.Linear(d_model, 2* n_head * d_head, bias =False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h , attn_mask = None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            c = self.layer_norm(c)


        head_q = self.q_net(h)
        head_k , head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)



        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt =0,
                 tgt_len = None, ext_len = None, mem_len = None, pre_lnorm=False):
        super(RelMultiHeadAttn,self).__init__()


        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError














class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias = False)


    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask = None, mems=None):

        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)


        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)


        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output




class RelLearnableMultiHeadAttn(RelMultiHeadAttn):

    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attention_mask=None, mems = None):
        qlen, bsz = w.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]                                   # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))                  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]                                              # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)


        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)


        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output




class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(EncoderLayer,self).__init__()
        self.enc_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)

        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))


    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.enc_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output




class RelLearnableEncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):

        super(RelLearnableEncoderLayer,self).__init__()

        self.enc_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)

        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))


    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):

        output = self.enc_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelPartialLearnableEncoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableEncoderLayer, self).__init__()

        self.enc_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)

        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))


    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.enc_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(EncoderLayer, self).__init__()

        self.enc_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.enc_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output



class RelLearnableEncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableEncoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableEncoderLayer(n_head, d_model, d_head, dropout,
                                         **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

class Video_TransfoXLModel(nn.Module):
    def __init__(self, n_layer, d_in, n_head, d_model, d_head, d_inner, dropout, dropatt, tie_weight= True, d_embed = None,
                 div_val = 1, tie_projts = [False], pre_lnorm = False, tgt_len = None, ext_len = None, mem_len=None,
                 cutoffs = [], adapt_inp=False, same_length=False, attn_type = 0, clamp_len = -1,
                 ):

        super(Video_TransfoXLModel, self).__init__()



        self.d_in = d_in
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head =  n_head
        self.d_head = d_head

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len
        self.attn_type = attn_type


        self.emb_layers = nn.Embedding(d_in, d_embed)






        # if not config.untie_r:
        #     self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        #     self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))

        self.layers = nn.ModuleList()
        if attn_type == 0:
            for i in range(self.n_layer):
                self.layers.append(RelPartialLearnableEncoderLayer(
                    n_head,
                    d_model,
                    d_head,
                    d_inner,
                    dropout,
                    dropatt = dropatt,
                    pre_lnorm = pre_lnorm,
                    tgt_len = tgt_len,
                    ext_len = ext_len,
                    mem_len = mem_len
                    # r_w_bias = None if config.untie_r else self.r_w_bias,
                    # r_r_bias = None if config.untie_r else self.r_r_bias,
                    # layer_norm_epsilon = config.layer_norm_epsilon,
                )
                )
        elif attn_type ==1:
            for i in range(self.n_layer):
                self.layers.append(RelLearnableEncoderLayer(
                    n_head,
                    d_model,
                    d_head,
                    d_inner,
                    dropout,
                    dropatt=dropatt,
                    pre_lnorm=pre_lnorm,
                    tgt_len=tgt_len,
                    ext_len=ext_len,
                    mem_len=mem_len
                    # r_w_bias = None if config.untie_r else self.r_w_bias,
                    # r_r_bias = None if config.untie_r else self.r_r_bias,
                    # layer_norm_epsilon = config.layer_norm_epsilon,
                )
                )
        elif attn_type in [2,3]:
            for i in range(self.n_layer):
                self.layers.append(
                    EncoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt = dropatt,  pre_lnorm = pre_lnorm
                    )
                )


        self.same_length = same_length
        self.clamp_len = clamp_len

        # self.pos_emb = PositionalEmbedding(self.d_model)

        self.init_weights()




    def init_weights(self):
        if self.attn_type ==0:
            self.pos_emb  = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor)
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

        elif self.attn_type == 1:

            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head , self.d_head
            ))

            self.r_w_bias = nn.Parameter(torch.Tensor(
                self.n_layer, self.n_head, self.d_head
            ))

        elif self.attn_type ==2:
            self.pos_emb = PositionalEmbedding(
                self.d_model
            )
        elif self.attn_type == 3:
            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head, self.d_head
            ))


    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len >0:
            mems =[]
            param = next(self.parameters())
            for i in range(self.n_layer +1):
                empty = torch.empty(0, dtype=param.dtype, device= param.device)
                mems.append(empty)

            return mems
        else:
            return None


    def update_mems(self, hids, mems, qlen, mlen):
        if mems is None: return None


        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems



    def _forward(self, dec_inp, mems=None):

        qlen , bsz = dec_inp.size()

        '''
        here we remove the adaptive embeddings
        '''
        token_emb = self.emb_layers(dec_inp)
        mlen = mems[0].size(0) if mems is not None else 0
        attn_mask = torch.zeros(token_emb.shape, dtype=token_emb.dtype, device= token_emb.device)
        klen = mlen + qlen

        # there is no attention mask at all
        hids = []

        if self.attn_type ==0:
            pos_seq = torch.arange(klen -1, -1, -1.0, device=token_emb.device,
                                   dtype=token_emb.dtype)
            if self.clamp_len >0:
                pos_seq.clamp_(max = self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)
            pos_emb = self.drop(pos_emb)
            core_out = self.drop(token_emb)

            hids.append(core_out)

            for i , layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias,
                                 dec_attn_mask = attn_mask, mems=mems_i)

                hids.append(core_out)
        elif self.attn_type ==1:
            core_out = self.drop(token_emb)
            hids.append(core_out)

            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb =self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]

                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                                 r_bias, dec_attn_mask=attn_mask, mems = mems_i)

                hids.append(core_out)

        elif self.attn_type ==2:
            pos_seq = torch.arange(klen-1, -1,-1.0, device= token_emb.device,
                                   dtype= token_emb.dtype)

            if self.clamp_len > 0:
                pos_seq.clamp_(max = self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(token_emb + pos_emb[-qlen:])
            hids.append(core_out)

            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i ==0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask = attn_mask, mems = mems_i)
                hids.append(core_out)


        core_out = self.drop(core_out)
        new_mems = self._update_mems(hids, mems, mlen, qlen)


        return core_out, new_mems



    def forward(self,data, target, *mems):

        if not mems: mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems = mems)

        pred_hid = hidden[-tgt_len:]




        if new_mems is None:
            return [pred_hid]
        else:
            return [pred_hid] + new_mems





























































#
#
# class LayerNormalization(nn.Module):
#     def __init__(self, features_count, epsilon=1e-6):
#         super().__init__()
#         self.gain = nn.Parameter(
#             torch.ones(features_count), requires_grad=True)
#         self.bias = nn.Parameter(
#             torch.zeros(features_count), requires_grad=True)
#         self.epsilon = epsilon
#
#     def forward(self, x):
#         mean = x.mean(dim=-1, keepdim=True)
#         std = x.std(dim=-1, keepdim=True)
#         return self.gain * (x - mean) / (std + self.epsilon) + self.bias
#
#
# def build_pooler(input_dim, cfg: EasyDict) -> nn.Module:
#     if cfg.pooler == "atn":
#         pooler = AtnPool(
#             input_dim, cfg.atn_pool_dim, cfg.atn_pool_heads, cfg.dropout)
#     elif cfg.pooler == "avg":
#         pooler = AvgPool()
#     else:
#         raise ValueError(f"unknown pooler {cfg.pooler}")
#     return pooler
#
#
#
# # need to change
# class Transformer(nn.Module):
#     def __init__(self, cfg: EasyDict, feature_dim: int):
#         super().__init__()
#
#         self.input_norm = LayerNormalization(feature_dim)
#         self.input_fc = None
#         input_dim = feature_dim
#
#         #  input_fc is to linear transformer the feature into the hidden states
#         if cfg.input_fc:
#             self.input_fc = nn.Sequential(
#                 nn.Linear(feature_dim, cfg.input_fc_output_dim), nn.GELU())
#             input_dim = cfg.input_fc_output_dim
#         self.embedding = PositionalEncoding(
#             input_dim, cfg.dropout, max_len=1000)
#
#         #  a tranformer layer to learn the interaction among different features
#         self.tf = TransformerEncoder(
#             cfg.num_layers, input_dim, cfg.num_heads, input_dim,
#             cfg.dropout)
#
#         self.use_context = cfg.use_context
#         if self.use_context:
#             self.tf_context = TransformerEncoder(
#                 cfg.atn_ctx_num_layers, input_dim, cfg.atn_ctx_num_heads,
#                 input_dim, cfg.dropout)
#
#
#         # the feature pooling layer
#         self.pooler = build_pooler(input_dim, cfg)
#
#         init_network(self, 0.01)
#
#     def forward(self, features, mask, lengths, hidden_state):
#         features = self.input_norm(features)
#         if self.input_fc is not None:
#             features = self.input_fc(features)
#         features = self.embedding(features)
#         features = self.tf(features, features, features, mask)
#         add_after_pool = None
#         if self.use_context:
#             hidden_state = hidden_state.unsqueeze(1)
#             ctx = self.tf_context(
#                 hidden_state, features, features, mask)
#             add_after_pool = ctx.squeeze(1)
#         pooled = self.pooler(features, mask, lengths)
#         if add_after_pool is not None:
#             pooled = torch.cat([pooled, add_after_pool], dim=-1)
#         return pooled
#
# # need to change
# class PositionalEncoding(nn.Module):
#     def __init__(self, dim, dropout_prob=0., max_len=1000):
#         super().__init__()
#         pe = torch.zeros(max_len, dim).float()
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         dimension = torch.arange(0, dim).float()
#         div_term = 10000 ** (2 * dimension / dim)
#         pe[:, 0::2] = torch.sin(position / div_term[0::2])
#         pe[:, 1::2] = torch.cos(position / div_term[1::2])
#         self.register_buffer('pe', pe)
#         self.dropout = nn.Dropout(p=dropout_prob)
#         self.dim = dim
#
#     def forward(self, x, step=None):
#         if step is None:
#             x = x + self.pe[:x.size(1), :]
#         else:
#             x = x + self.pe[:, step]
#         x = self.dropout(x)
#         return x
#
# # need to change
# class TransformerEncoder(nn.Module):
#     def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob):
#         super().__init__()
#         self.d_model = d_model
#         assert layers_count > 0
#         self.encoder_layers = nn.ModuleList(
#             [TransformerEncoderLayer(
#                 d_model, heads_count, d_ff, dropout_prob)
#                 for _ in range(layers_count)])
#
#     def forward(self, query, key, value, mask):
#         batch_size, query_len, embed_dim = query.shape
#         batch_size, key_len, embed_dim = key.shape
#         mask = (1 - mask.unsqueeze(1).expand(batch_size, query_len, key_len))
#         mask = mask == 1
#         sources = None
#         for encoder_layer in self.encoder_layers:
#             sources = encoder_layer(query, key, value, mask)
#         return sources
#
# #need to change
# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, heads_count, d_ff, dropout_prob):
#         super(TransformerEncoderLayer, self).__init__()
#         self.self_attention_layer = Sublayer(
#             MultiHeadAttention(heads_count, d_model, dropout_prob), d_model)
#         self.pointwise_feedforward_layer = Sublayer(
#             PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)
#         self.dropout = nn.Dropout(dropout_prob)
#
#     def forward(self, query, key, value, sources_mask):
#         sources = self.self_attention_layer(query, key, value, sources_mask)
#         sources = self.dropout(sources)
#         sources = self.pointwise_feedforward_layer(sources)
#         return sources
#
#
# class Sublayer(nn.Module):
#     def __init__(self, sublayer, d_model):
#         super(Sublayer, self).__init__()
#         self.sublayer = sublayer
#         self.layer_normalization = LayerNormalization(d_model)
#
#     def forward(self, *args):
#         x = args[0]
#         x = self.sublayer(*args) + x
#         return self.layer_normalization(x)
#
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, heads_count, d_model, dropout_prob):
#         super().__init__()
#         assert d_model % heads_count == 0,\
#             f"model dim {d_model} not divisible by {heads_count} heads"
#         self.d_head = d_model // heads_count
#         self.heads_count = heads_count
#         self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
#         self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
#         self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
#         self.final_projection = nn.Linear(d_model, heads_count * self.d_head)
#         self.dropout = nn.Dropout(dropout_prob)
#         self.softmax = nn.Softmax(dim=3)
#         self.attention = None
#
#     def forward(self, query, key, value, mask=None):
#         batch_size, query_len, d_model = query.size()
#         d_head = d_model // self.heads_count
#         query_projected = self.query_projection(query)
#         key_projected = self.key_projection(key)
#         value_projected = self.value_projection(value)
#         batch_size, key_len, d_model = key_projected.size()
#         batch_size, value_len, d_model = value_projected.size()
#         query_heads = query_projected.view(
#             batch_size, query_len, self.heads_count, d_head).transpose(1, 2)
#         key_heads = key_projected.view(
#             batch_size, key_len, self.heads_count, d_head).transpose(1, 2)
#         value_heads = value_projected.view(
#             batch_size, value_len, self.heads_count, d_head).transpose(1, 2)
#         attention_weights = self.scaled_dot_product(
#             query_heads, key_heads)
#         if mask is not None:
#             mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
#             attention_weights = attention_weights.masked_fill(
#                 mask_expanded, -1e18)
#         attention = self.softmax(attention_weights)
#         attention_dropped = self.dropout(attention)
#         context_heads = torch.matmul(
#             attention_dropped, value_heads)
#         context_sequence = context_heads.transpose(1, 2)
#         context = context_sequence.reshape(
#             batch_size, query_len, d_model)
#         final_output = self.final_projection(context)
#         return final_output
#
#     def scaled_dot_product(self, query_heads, key_heads):
#         key_heads_transposed = key_heads.transpose(2, 3)
#         dot_product = torch.matmul(
#             query_heads, key_heads_transposed)
#         attention_weights = dot_product / np.sqrt(self.d_head)
#         return attention_weights
#
#
# class PointwiseFeedForwardNetwork(nn.Module):
#     def __init__(self, d_ff, d_model, dropout_prob):
#         super(PointwiseFeedForwardNetwork, self).__init__()
#         self.feed_forward = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.Dropout(dropout_prob),
#             nn.GELU(),
#             nn.Linear(d_ff, d_model),
#             nn.Dropout(dropout_prob))
#
#     def forward(self, x):
#         return self.feed_forward(x)
#
#
# class AvgPool(nn.Module):
#     def forward(self, features, mask, lengths):
#         _ = mask
#         len_div = lengths.unsqueeze(-1).float()
#         result_sum = torch.sum(features, dim=1)
#         result = result_sum / len_div
#         return result
#
#
# class AtnPool(nn.Module):
#     def __init__(
#             self, d_input, d_attn, n_heads, dropout_prob):
#         super().__init__()
#         self.d_head = d_attn // n_heads
#         self.d_head_output = d_input // n_heads
#         self.num_heads = n_heads
#
#         def init_(tensor_):
#             tensor_.data = (utils.truncated_normal_fill(
#                 tensor_.data.shape, std=0.01))
#
#         w1_head = torch.zeros(n_heads, d_input, self.d_head)
#         b1_head = torch.zeros(n_heads, self.d_head)
#         w2_head = torch.zeros(n_heads, self.d_head, self.d_head_output)
#         b2_head = torch.zeros(n_heads, self.d_head_output)
#         init_(w1_head)
#         init_(b1_head)
#         init_(w2_head)
#         init_(b2_head)
#         self.genpool_w1_head = nn.Parameter(w1_head, requires_grad=True)
#         self.genpool_b1_head = nn.Parameter(b1_head, requires_grad=True)
#         self.genpool_w2_head = nn.Parameter(w2_head, requires_grad=True)
#         self.genpool_b2_head = nn.Parameter(b2_head, requires_grad=True)
#         self.activation = nn.GELU()
#         self.dropout1 = nn.Dropout(dropout_prob)
#         self.dropout2 = nn.Dropout(dropout_prob)
#         self.dropout3 = nn.Dropout(dropout_prob)
#         self.softmax = nn.Softmax(dim=2)
#         self.softmax_temp = 1
#         self.genpool_one = nn.Parameter(torch.ones(1), requires_grad=False)
#
#     def extra_repr(self) -> str:
#         strs = []
#         for p in [self.genpool_w1_head, self.genpool_b1_head,
#                   self.genpool_w2_head, self.genpool_b2_head]:
#             strs.append(f"pool linear {p.shape}")
#         return "\n".join(strs)
#
#     def forward(self, features, mask, lengths):
#         _ = lengths
#         batch_size, seq_len, input_dim = features.shape
#         b1 = torch.matmul(
#             features.unsqueeze(1),
#             self.genpool_w1_head.unsqueeze(0))
#         b1 += self.genpool_b1_head.unsqueeze(1).unsqueeze(0)
#         b1 = self.activation(self.dropout1(b1))
#         b1 = torch.matmul(
#             b1, self.genpool_w2_head.unsqueeze(0))
#         b1 += self.genpool_b2_head.unsqueeze(1).unsqueeze(0)
#         b1 = self.dropout2(b1)
#         b1.masked_fill_((mask == 0).unsqueeze(1).unsqueeze(-1), -1e19)
#         smweights = self.softmax(b1 / self.softmax_temp)
#         smweights = self.dropout3(smweights)
#         smweights = smweights.transpose(1, 2).reshape(
#             -1, seq_len, input_dim)
#         pooled = (features * smweights).sum(dim=1)
#         return pooled
#
#
# def init_weight_(w, init_gain=1):
#     w.copy_(utils.truncated_normal_fill(w.shape, std=init_gain))
#
#
# def init_network(net: nn.Module, init_std: float):
#     for key, val in net.named_parameters():
#         if "weight" in key or "bias" in key:
#             init_weight_(val.data, init_std)
