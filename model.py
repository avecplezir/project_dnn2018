import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from constants import *
import numpy as np
import copy 

import math
import time
from IPython import display
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import optim
from util import one_hot

criterion_bce_play = nn.BCELoss()  
criterion_bce_replay = nn.BCELoss() 
criterion_mse = nn.MSELoss()  

def compute_loss(y_pred, y_true):
    
    played = y_true[:, :, :, 0]
    
    bce_note = criterion_bce_play(y_pred[:, :, :, 0], y_true[:, :, :, 0])

    replay = played*y_pred[:, :, :, 1] + (1 - played)*y_true[:, :, :, 1]
    
    bce_replay = criterion_bce_replay(replay, y_true[:, :, :, 1])
    
    volume = played*y_pred[:, :, :, 2] + (1 - played)*y_true[:, :, :, 2]
    mse = criterion_mse(volume, y_true[:, :, :, 2] )
    
    return bce_note + bce_replay 
# + mse


def iterate_minibatches(train_data, train_labels, batchsize):
    indices = np.random.permutation(np.arange(len(train_labels)))
    for start in range(0, len(indices), batchsize):
        ix = indices[start: start + batchsize]
        
        if cuda:      
            yield  Variable(torch.FloatTensor(train_data[ix])).cuda(), Variable(torch.FloatTensor(train_labels[ix])).cuda()
        else:
            yield Variable(torch.FloatTensor(train_data[ix])), Variable(torch.FloatTensor(train_labels[ix]))

def pitch_pos_in_f(x):
    """
    Returns a constant containing pitch position of each note
    """
    pos_in = torch.FloatTensor(np.arange(NUM_NOTES)/NUM_NOTES)
    pos_in = pos_in.repeat(x.shape[:-2]+(1,))[:,:,:,None]
    
    return get_variable(pos_in)

def pitch_class_in_f(x):
    """
    Returns a constant containing pitch class of each note
    """

    pitch_class_matrix = np.array([one_hot(n % OCTAVE, OCTAVE) for n in range(NUM_NOTES)])
    pitch_class_matrix = torch.FloatTensor(pitch_class_matrix)
    pitch_class_matrix = pitch_class_matrix.view(1, 1, NUM_NOTES, OCTAVE)
    pitch_class_matrix = pitch_class_matrix.repeat((x.shape[:2]+ (1, 1)))
    
    return get_variable(pitch_class_matrix)

def pitch_bins_f(x):

        bins = [x[:, :, i::OCTAVE, :1].sum(2) for i in range(OCTAVE)]
        bins = torch.cat(bins, dim = -1)
        bins = bins.repeat(NUM_OCTAVES, 1, 1)
        bins = bins.view(x.shape[:2]+(NUM_NOTES, 1))
        
        return bins
    
def beat_f(x):

#     print('x', x.shape)
    beats = torch.FloatTensor(np.array([one_hot(b % NOTES_PER_BAR, NOTES_PER_BAR) for b in range(x.shape[1])]))
#     print('beats', beats.shape)
#     beats = beats.view((1, 1, x.shape[1], NOTES_PER_BAR))
    beats = beats.repeat(((x.shape[0], x.shape[2])+ (1, 1)))
#     print('beats', beats.shape)
    beats = beats.permute(0,2,1,3).contiguous()
#     print('beats', beats.shape)
#     print('beats', beats[0,1,3,:])
#     print('beats', beats[1,2,21,:])

    return get_variable(beats)
    
    
class feature_generation(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()        
        self.padding = nn.ZeroPad2d(((2 * OCTAVE - 1)//2,  math.ceil((2 * OCTAVE - 1)/2), 0, 0))
        self.conv = nn.Conv1d(NOTE_UNITS,  OCTAVE_UNITS, 2 * OCTAVE)
        
    def forward(self, notes):
        initial_shape = notes.shape
        
        # convolution
        notes = notes.contiguous().view((-1,)+ notes.shape[-2:]).contiguous()
        notes = notes.permute(0, 2, 1).contiguous()
        notes = self.padding(notes)
        notes = self.conv(notes)
        notes = nn.Tanh()(notes)
        notes = notes.permute(0, 2, 1).contiguous()
        notes = notes.contiguous().view(initial_shape[:2] + notes.shape[-2:])
        
        pos_in = pitch_pos_in_f(notes)
        class_in = pitch_class_in_f(notes)
        bins = pitch_bins_f(notes)        
        note_features = torch.cat([notes, pos_in, class_in, bins], dim = -1) 
    
        return note_features

def get_variable(x):
    if cuda:      
        return  Variable(x.cuda(), requires_grad=False)
    else:
        return Variable(x, requires_grad=False)

class time_axis(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__() 
        self.n_layers = TIME_AXIS_LAYERS
        self.hidden_size = TIME_AXIS_UNITS
        
        self.attention_layer = MultiHeadAttention() #MultiHeadedAttention()
        self.FF = PositionwiseFeedForward(D_MODEL, 4*D_MODEL)
        self.self_attention = SELF_ATTENTION

        self.input_size = D_MODEL 
        self.use_beat = True
        
        if self.use_beat:       
            self.input_size += BEATS_FEATURES

        self.time_lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, dropout=0.1, 
                                 batch_first=True, )
        self.dropout = nn.Dropout(p=0.2)
        self.generate_features = feature_generation()
        
        
    def forward(self, notes, beats = None):
        
        """
        arg:
            notes - (batch, time_seq, note_seq, note_features)
        
        out: 
            (batch, time_seq, note_seq, hidden_features)
            
        """
        
        notes = self.dropout(notes)
        
        initial_shape = notes.shape
        
        # convolution
        note_features =  self.generate_features(notes)   
        notes = note_features
    
        initial_shape = notes.shape
        
        if self.self_attention:
            # multihead attention
            notes = notes.contiguous().view((-1,)+ notes.shape[-2:]).contiguous()
            notes = self.attention_layer(notes, notes, notes)
            notes = notes.contiguous().view(initial_shape[:2] + notes.shape[-2:])       
            notes = notes + note_features
            # FF
            note_features = self.FF(notes)
            notes = notes + note_features
            
        if self.use_beat:
            if beats is None:
                beats = beat_f(notes)
            else:
                beats = beats.repeat((notes.shape[2], 1,1,1)).permute(1,2,0,3).contiguous()

            notes = torch.cat([notes, beats], dim = -1)

        
        initial_shape = notes.shape
        
        notes = notes.permute(0, 2, 1, 3).contiguous()
        notes = notes.view((-1,)+ notes.shape[-2:]).contiguous()

        out, hidden = self.time_lstm(notes) 
                
        time_output = out.contiguous().view((initial_shape[0],) + (initial_shape[2],) + out.shape[-2:])
        time_output = time_output.permute(0, 2, 1, 3)        
        
        return time_output        
    
def sample_sound2(data_gen):
    size = data_gen.size()
    rand = torch.rand(*size).cuda()
    sample = (rand<data_gen).type(torch.FloatTensor).cuda()
    sample[:,:,2] = sample[:,:,0]
    return sample

def sample_sound(data_gen):
    size = data_gen.size()
    rand = torch.rand(*size).cuda()
    sample = (rand<data_gen).type(torch.FloatTensor).cuda()
    sample[:,:,:,2] = sample[:,:,:,0]
    return sample

class note_axis(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()   
        
        
        self.n_layers = NOTE_AXIS_LAYERS
        self.hidden_size = NOTE_AXIS_UNITS
        # number of time features plus number of previous higher note in the same time momemt
        self.input_size = TIME_AXIS_UNITS + NOTE_UNITS
       
        self.note_lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, dropout=0.1, 
                                 batch_first=True, )
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.logits = nn.Linear(self.hidden_size+NUM_TRACK_FEATURE, NOTE_UNITS) 
        self.to_train = True
        
        self.apply_T = False
        self.temperature = 1
        self.silent_time = 0
        
    def forward(self, notes, chosen, overall_info):
        """
        arg:
            notes - (batch, time_seq, note_seq, time_hidden_features)
        
        out: 
            (batch, time_seq, note_seq, next_notes_features)
            
        """
    
        if self.to_train:
            # Shift target one note to the left.
            chosen = self.dropout(chosen)
            shift_chosen = nn.ZeroPad2d((0, 0, 1, 0))(chosen[:, :, :-1, :]) 
            notes = torch.cat([notes, shift_chosen], dim=-1)

        
        initial_shape = notes.shape    
        note_input = notes.contiguous().view((-1,)+ notes.shape[-2:]).contiguous()
        
        if self.to_train:
            out, hidden = self.note_lstm(note_input) 
            note_output = out.contiguous().view(initial_shape[:2] + out.shape[-2:])
            info = overall_info.expand((note_output.shape[1:3]+overall_info.shape)).permute(2,0,1,3).contiguous()
            note_output = torch.cat([note_output, info], dim =-1)
            logits = self.logits(note_output) 
            next_notes = nn.Sigmoid()(logits)      
            return next_notes, sample_sound(next_notes)
        
        else:
            hidden = (torch.zeros(self.n_layers, note_input.shape[0], self.hidden_size).cuda(), 
                      torch.zeros(self.n_layers, note_input.shape[0], self.hidden_size).cuda()) 
            notes_list = []
            sound_list = []
            sound = torch.zeros(note_input[:,0:1,:].shape[:-1]+(NOTE_UNITS,)).cuda()

            for i in range(NUM_OCTAVES*OCTAVE):
                
                inputs = torch.cat([note_input[:,i:i+1,:], sound], dim = -1)
                out, hidden = self.note_lstm(inputs, hidden) 
                
                info = overall_info.expand(((1,)+overall_info.shape)).permute(1,0,2).contiguous()
                note_output = torch.cat([out, info], dim =-1)
                
                logits = self.logits(note_output) 
                if self.apply_T:
                    next_notes = nn.Sigmoid()(logits/self.temperature)
                else:
                    next_notes = nn.Sigmoid()(logits)
                    
                sound = sample_sound2(next_notes)
                notes_list.append(next_notes)
                sound_list.append(sound)
                
                if self.apply_T:  
                    sounds = torch.cat(sound_list, dim = 1)
#                     print('sounds', sounds.shape)
                    if  ((sounds[-1,:,0] != 0).sum() == 0):
                        self.silent_time += 1
                        if self.silent_time >= NOTES_PER_BAR:
                            self.temperature += 0.1
                    else:
                        self.silent_time = 0
                        self.temperature = 1
                
            out = torch.cat(notes_list, dim = 1)
            sounds = torch.cat(sound_list, dim = 1)
            note_output = out.contiguous().view(initial_shape[:2] + out.shape[-2:])
            sounds = sounds.contiguous().view(initial_shape[:2] + out.shape[-2:])
            
#             if self.apply_T:
                
#                 if  ((sounds[-1,-1,:,0] != 0).sum() == 0):
#                     self.silent_time += 1
#                     if self.silent_time >= NOTES_PER_BAR:
#                         self.temperature += 0.1*10
#                 else:
#                     self.silent_time = 0
#                     self.temperature = 1
            
            return note_output, sounds
    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
def attention(query, key, value):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)
#     p_attn = nn.Sigmoid()(scores) 
    query_new = torch.matmul(p_attn, value)
    
    return query_new*query, p_attn
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        
        self.d_model = D_MODEL
        self.h = N_HEADS
        self.d_k = PROJECTION_DIM
        
        self.linears = clones(nn.Linear(self.d_model, self.d_k*self.h,  bias=False), 3)
        self.linear = nn.Linear(self.d_k*self.h, self.d_k*self.h,  bias=False)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value):

        initial_x = query
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
            
        return self.linear(x)
    #+initial_x
    
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2)) 
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output*q, attn
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = N_HEADS
        self.d_k = PROJECTION_DIM
        self.d_v = PROJECTION_DIM
        self.d_model = D_MODEL

        self.w_qs = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_v))

        self.attention = ScaledDotProductAttention()
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
        outputs, attns = self.attention(q_s, k_s, v_s)

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
    
class track_feature(nn.Module):
    def __init__(self, dropout=0.3):
        super(self.__class__, self).__init__()        
        
        self.overall_information = nn.Conv2d(3, OUT_CHANEL_TRACK, (32, 48), padding=0, stride=16)
        self.l = nn.Linear(OUT_CHANEL_TRACK*7, NUM_TRACK_FEATURE)
        
    def forward(self, notes):
              
        overall_info = notes.permute(0, 3, 1, 2).contiguous()
        overall_info = self.overall_information(overall_info) #F.relu(
#         print('overall_info', overall_info.shape)
# F.relu( nn.LeakyReLU(negative_slope=0.1)
        overall_info = nn.LeakyReLU(negative_slope=0.1)(self.l(overall_info.view((overall_info.shape[0],-1))))
        
        return overall_info     
    
class Generator(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()        
        
        self.time_ax = time_axis() 
        self.note_ax = note_axis()
        #in_ch, out_ch, kernel
        self.overall_information = track_feature()
        
    def forward(self, notes, chosen = None, beat = None):

        overall_info = self.overall_information(notes)
                                                              
        note_ax_output = self.time_ax(notes, beat)
        output = self.note_ax(note_ax_output, chosen, overall_info)
                                                              
        
        return output 
    

def train(generator, X_tr, X_te, y_tr, y_te, batchsize=3, n_epochs = 3, verbose = True):
    
    optimizer = optim.Adam(generator.parameters())
    n_train_batches = math.ceil(len(X_tr)/batchsize)
    n_validation_batches = math.ceil(len(X_te)/batchsize)

    epoch_history = {'train_loss':[], 'val_loss':[]}

    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss = 0
        generator.train(True)    
                    
        try:
            for X, y in tqdm(iterate_minibatches(X_tr, y_tr, batchsize)):

                pred, sound = generator(X, y)
                loss = compute_loss(pred, y) 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().data.numpy()

            train_loss /= n_train_batches
            epoch_history['train_loss'].append(train_loss)

            generator.train(False)
            val_loss = 0
            for X, y in tqdm(iterate_minibatches(X_te, y_te, batchsize)):
                pred, sound = generator(X, y)
                loss = compute_loss(pred, y) 

                val_loss += loss.cpu().data.numpy()

            val_loss /= n_validation_batches
            epoch_history['val_loss'].append(val_loss)
        
        except KeyboardInterrupt:
            return generator, epoch, epoch_history  
            
        # Visualize
        if verbose:
            display.clear_output(wait=True)
            plt.figure(figsize=(16, 6))
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time)) 
            print('current train loss: {}'.format(epoch_history['train_loss'][-1]))
            print('current val loss: {}'.format(epoch_history['val_loss'][-1]))

            plt.title("losses")
            plt.xlabel("#epoch")
            plt.ylabel("loss")
            plt.plot(epoch_history['train_loss'], 'b', label = 'train_loss')
            plt.plot(epoch_history['val_loss'], 'g', label = 'val_loss')
            plt.legend()
            plt.show()

    print("Finished!")
    
    return generator, epoch, epoch_history