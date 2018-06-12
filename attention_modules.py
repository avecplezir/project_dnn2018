import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from constants import *
import numpy as np
import copy 
import math
    
    
def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 3
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask
    
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)
        
    def forward(self, q, k, v, attn_mask=None):

#         print('q', q.shape)
        attn = torch.bmm(q, k.transpose(1, 2))
    
#         print('attn', attn.shape)
        
        if attn_mask is not None:

            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        if attn_mask is not None:
            return output*q, attn
        else:     
            return output*q, attn
    
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head #N_HEADS, PROJECTION_DIM, D_MODEL
        self.d_k = d_k#PROJECTION_DIM
        self.d_v = d_k#PROJECTION_DIM
        self.d_model = d_model#D_MODEL

        self.w_qs = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_v))

        self.attention = ScaledDotProductAttention(d_model)
        self.proj = nn.Linear(self.n_head*self.d_v, self.d_model)

        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.w_qs)
        nn.init.xavier_normal_(self.w_ks)
        nn.init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.n_head, 1, 1)
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1) 

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return outputs
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
    
#===============================================================================================

# class MultiHeadedAttention(nn.Module):
#     def __init__(self, dropout=0.1):
#         super(MultiHeadedAttention, self).__init__()
        
#         self.d_model = D_MODEL
#         self.h = N_HEADS
#         self.d_k = PROJECTION_DIM
        
#         self.linears = clones(nn.Linear(self.d_model, self.d_k*self.h,  bias=False), 3)
#         self.linear = nn.Linear(self.d_k*self.h, self.d_k*self.h,  bias=False)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)
        
#     def forward(self, query, key, value):

#         initial_x = query
#         nbatches = query.size(0)
        
#         # 1) Do all the linear projections in batch from d_model => h x d_k 
#         query, key, value = \
#             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#              for l, x in zip(self.linears, (query, key, value))]
        
#         # 2) Apply attention on all the projected vectors in batch. 
#         x, self.attn = attention(query, key, value)
        
#         # 3) "Concat" using a view and apply a final linear. 
#         x = x.transpose(1, 2).contiguous() \
#              .view(nbatches, -1, self.h * self.d_k)
            
#         return self.linear(x)
#     #+initial_x
    
# class ScaledDotProductAttention(nn.Module):
#     ''' Scaled Dot-Product Attention '''

#     def __init__(self, attn_dropout=0.1):
#         super(ScaledDotProductAttention, self).__init__()
#         self.dropout = nn.Dropout(attn_dropout)

#     def forward(self, q, k, v):

#         attn = torch.bmm(q, k.transpose(1, 2)) 
#         attn = F.softmax(attn, dim=-1)
#         attn = self.dropout(attn)
#         output = torch.bmm(attn, v)

#         return output*q, attn
    
# class MultiHeadAttention(nn.Module):
#     ''' Multi-Head Attention module '''

#     def __init__(self, dropout=0.1):
#         super(MultiHeadAttention, self).__init__()

#         self.n_head = N_HEADS
#         self.d_k = PROJECTION_DIM
#         self.d_v = PROJECTION_DIM
#         self.d_model = D_MODEL

#         self.w_qs = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_k))
#         self.w_ks = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_k))
#         self.w_vs = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_v))

#         self.attention = ScaledDotProductAttention()
#         self.proj = nn.Linear(self.n_head*self.d_v, self.d_model)

#         self.dropout = nn.Dropout(dropout)

#         nn.init.xavier_normal_(self.w_qs)
#         nn.init.xavier_normal_(self.w_ks)
#         nn.init.xavier_normal_(self.w_vs)

#     def forward(self, q, k, v, attn_mask=None):

#         d_k, d_v = self.d_k, self.d_v
#         n_head = self.n_head

#         residual = q

#         mb_size, len_q, d_model = q.size()
#         mb_size, len_k, d_model = k.size()
#         mb_size, len_v, d_model = v.size()

#         # treat as a (n_head) size batch
#         q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
#         k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
#         v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model

#         # treat the result as a (n_head * mb_size) size batch
#         q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
#         k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
#         v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

#         # perform attention, result size = (n_head * mb_size) x len_q x d_v
#         outputs, attns = self.attention(q_s, k_s, v_s)

#         # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
#         outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1) 

#         # project back to residual size
#         outputs = self.proj(outputs)
#         outputs = self.dropout(outputs)

#         return outputs
    
# class PositionwiseFeedForward(nn.Module):
#     "Implements FFN equation."
#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super(PositionwiseFeedForward, self).__init__()
#         self.w_1 = nn.Linear(d_model, d_ff)
#         self.w_2 = nn.Linear(d_ff, d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         return self.w_2(self.dropout(F.relu(self.w_1(x))))
