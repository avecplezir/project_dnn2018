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

from feature_generation_module import feature_generation, beat_f
from attention_modules import get_attn_subsequent_mask, MultiHeadAttention, PositionwiseFeedForward

def sample_sound2(data_gen):
    size = data_gen.size()
    rand = torch.from_numpy(np.random.random(size)).type(torch.FloatTensor).cuda()
#     rand = torch.rand(*size).cuda()
    sample = (rand<data_gen).type(torch.FloatTensor).cuda()
    sample[:,:,2] = sample[:,:,0]
    return sample

def sample_sound(data_gen):
    size = data_gen.size()
#     rand = torch.rand(*size).cuda()
    rand = torch.from_numpy(np.random.random(size)).type(torch.FloatTensor).cuda()
    sample = (rand<data_gen).type(torch.FloatTensor).cuda()
    sample[:,:,:,2] = sample[:,:,:,0]
    return sample

class time_axis(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__() 
        self.n_layers = TIME_AXIS_LAYERS
        self.hidden_size = TIME_AXIS_UNITS
        
        self.attention_note_axis = ATTENTION_NOTE_AXIS
        self.attention_note_layer = MultiHeadAttention(N_HEADS, PROJECTION_DIM, D_MODEL) 
        self.FF_note = PositionwiseFeedForward(D_MODEL, 4*D_MODEL)             
        self.attention_time_axis = ATTENTION_TIME_AXIS
        self.attention_time_layer = MultiHeadAttention(4, int(self.hidden_size//4), self.hidden_size) 
        self.FF_time = PositionwiseFeedForward(self.hidden_size, 4*self.hidden_size)  
       
        self.input_size = D_MODEL 
        self.use_beat = True
        
        if self.use_beat:       
            self.input_size += BEATS_FEATURES

        self.time_lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, dropout=0.1, 
                                 batch_first=True, )
        self.dropout = nn.Dropout(p=0.2)
        self.generate_features = feature_generation()
        self.beat_embedding =nn.Embedding(num_embeddings=16, embedding_dim = 16)

        
        
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
        
        if self.attention_note_axis:
            # multihead attention
            notes = notes.contiguous().view((-1,)+ notes.shape[-2:]).contiguous()
            notes = self.attention_note_layer(notes, notes, notes)
            notes = notes.contiguous().view(initial_shape[:2] + notes.shape[-2:])       
            notes = notes + note_features
            # FF
            note_features = self.FF_note(notes)
            notes = notes + note_features
            
        if self.use_beat:
            if beats is None:
                beats = beat_f(notes)
            else:
                beats = beats.repeat((notes.shape[2], 1,1)).permute(1,2,0).contiguous()
#                 print('pytorch', beats[0,-1,1])         
            beats = self.beat_embedding(beats)
            notes = torch.cat([notes, beats], dim = -1)

        
        initial_shape = notes.shape
        
        notes = notes.permute(0, 2, 1, 3).contiguous()
        notes = notes.view((-1,)+ notes.shape[-2:]).contiguous()

        lstm_out, hidden = self.time_lstm(notes) 
        
        if self.attention_time_axis:
            attn_mask = get_attn_subsequent_mask(lstm_out)
#             print('attn_mask', attn_mask.shape)
            notes = self.attention_time_layer(lstm_out, lstm_out, lstm_out, attn_mask=attn_mask)      
            lstm_out = notes + lstm_out
            # FF
            notes = self.FF_time(lstm_out)
            lstm_out = notes + lstm_out
                
        time_output = lstm_out.contiguous().view((initial_shape[0],) + (initial_shape[2],) + lstm_out.shape[-2:])
        time_output = time_output.permute(0, 2, 1, 3)        
        
        return time_output        

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
                    
            out = torch.cat(notes_list, dim = 1)
            sounds = torch.cat(sound_list, dim = 1)
            
            if self.apply_T:  
                if  ((sounds[-1,:,0] != 0).sum() == 0):
                    self.silent_time += 1
                    if self.silent_time >= NOTES_PER_BAR:
                        self.temperature += 0.1
                else:
                    self.silent_time = 0
                    self.temperature = 1 
                                        
            note_output = out.contiguous().view(initial_shape[:2] + out.shape[-2:])
            sounds = sounds.contiguous().view(initial_shape[:2] + out.shape[-2:])
                        
            return note_output, sounds
        
class track_feature(nn.Module):
    def __init__(self, dropout=0.3):
        super(self.__class__, self).__init__()        
        
        self.overall_information = nn.Conv2d(3, OUT_CHANEL_TRACK, (32, 48), padding=0, stride=16)
        self.l = nn.Linear(OUT_CHANEL_TRACK*7, NUM_TRACK_FEATURE)
        
    def forward(self, notes):
              
        overall_info = notes.permute(0, 3, 1, 2).contiguous()
        overall_info = self.overall_information(overall_info)
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

#=================================================================================================
# import torch, torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from constants import *
# import numpy as np

# from attention_modules import get_attn_subsequent_mask, ScaledDotProductAttention, MultiHeadedAttention, MultiHeadAttention, PositionwiseFeedForward
# from feature_generation_module import feature_generation, beat_f

# def sample_sound2(data_gen):
#     size = data_gen.size()
#     rand = torch.rand(*size).cuda()
#     sample = (rand<data_gen).type(torch.FloatTensor).cuda()
#     sample[:,:,2] = sample[:,:,0]
#     return sample

# def sample_sound(data_gen):
#     size = data_gen.size()
#     rand = torch.rand(*size).cuda()
#     sample = (rand<data_gen).type(torch.FloatTensor).cuda()
#     sample[:,:,:,2] = sample[:,:,:,0]
#     return sample

# class time_axis(nn.Module):
#     def __init__(self):
#         super(self.__class__, self).__init__() 
#         self.n_layers = TIME_AXIS_LAYERS
#         self.hidden_size = TIME_AXIS_UNITS
        
              
#         self.attention_note_axis = ATTENTION_NOTE_AXIS
#         self.attention_note_layer = MultiHeadAttention(N_HEADS, PROJECTION_DIM, D_MODEL) #MultiHeadedAttention()
#         self.FF_note = PositionwiseFeedForward(D_MODEL, 4*D_MODEL)             
#         self.attention_time_axis = ATTENTION_TIME_AXIS
#         self.attention_time_layer = MultiHeadAttention(4, (self.hidden_size//4), self.hidden_size) #MultiHeadedAttention()
#         self.FF_time = PositionwiseFeedForward(self.hidden_size, 4*self.hidden_size)  

#         self.input_size = D_MODEL 
#         self.use_beat = True
        
#         if self.use_beat:       
#             self.input_size += BEATS_FEATURES

#         self.time_lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, dropout=0.1, 
#                                  batch_first=True, )
#         self.dropout = nn.Dropout(p=0.2)
#         self.generate_features = feature_generation()
        
#         self.beat_embedding =nn.Embedding(num_embeddings=16, embedding_dim = 16)

        
        
#     def forward(self, notes, beats = None):
        
#         """
#         arg:
#             notes - (batch, time_seq, note_seq, note_features)
        
#         out: 
#             (batch, time_seq, note_seq, hidden_features)
            
#         """
        
#         notes = self.dropout(notes)
        
#         initial_shape = notes.shape
        
#         # convolution
#         note_features =  self.generate_features(notes)   
#         notes = note_features
    
#         initial_shape = notes.shape
        
#         if self.attention_note_axis:
#             # multihead attention
#             notes = notes.contiguous().view((-1,)+ notes.shape[-2:]).contiguous()
#             notes = self.attention_note_layer(notes, notes, notes)
#             notes = notes.contiguous().view(initial_shape[:2] + notes.shape[-2:])       
#             notes = notes + note_features
#             # FF
#             note_features = self.FF_note(notes)
#             notes = notes + note_features
            
#         if self.use_beat:
#             if beats is None:
#                 beats = beat_f(notes)
#             else:
#                 beats = beats.repeat((notes.shape[2], 1,1,1)).permute(1,2,0,3).contiguous()

# #             beats = self.beat_embedding(beats)
# #             print('beats', beats.shape)
#             notes = torch.cat([notes, beats], dim = -1)

        
#         initial_shape = notes.shape
        
#         notes = notes.permute(0, 2, 1, 3).contiguous()
#         notes = notes.view((-1,)+ notes.shape[-2:]).contiguous()

#         lstm_out, hidden = self.time_lstm(notes) 
        
#         if self.attention_time_axis:
#             attn_mask = get_attn_subsequent_mask(lstm_out)
# #             print('attn_mask', attn_mask.shape)
#             notes = self.attention_time_layer(lstm_out, lstm_out, lstm_out, attn_mask=attn_mask)      
#             lstm_out = notes + lstm_out
#             # FF
#             notes = self.FF_time(lstm_out)
#             lstm_out = notes + lstm_out
                
#         time_output = lstm_out.contiguous().view((initial_shape[0],) + (initial_shape[2],) + lstm_out.shape[-2:])
#         time_output = time_output.permute(0, 2, 1, 3)        
        
#         return time_output        
    

# class note_axis(nn.Module):
#     def __init__(self):
#         super(self.__class__, self).__init__()   
        
        
#         self.n_layers = NOTE_AXIS_LAYERS
#         self.hidden_size = NOTE_AXIS_UNITS
#         # number of time features plus number of previous higher note in the same time momemt
#         self.input_size = TIME_AXIS_UNITS + NOTE_UNITS
       
#         self.note_lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, dropout=0.1, 
#                                  batch_first=True, )
        
#         self.dropout = nn.Dropout(p=0.2)
        
#         self.logits = nn.Linear(self.hidden_size+NUM_TRACK_FEATURE, NOTE_UNITS) 
#         self.to_train = True
        
#         self.apply_T = False
#         self.temperature = 1
#         self.silent_time = 0
        
#     def forward(self, notes, chosen, overall_info):
#         """
#         arg:
#             notes - (batch, time_seq, note_seq, time_hidden_features)
        
#         out: 
#             (batch, time_seq, note_seq, next_notes_features)
            
#         """
    
#         if self.to_train:
#             # Shift target one note to the left.
#             chosen = self.dropout(chosen)
#             shift_chosen = nn.ZeroPad2d((0, 0, 1, 0))(chosen[:, :, :-1, :]) 
#             notes = torch.cat([notes, shift_chosen], dim=-1)

        
#         initial_shape = notes.shape    
#         note_input = notes.contiguous().view((-1,)+ notes.shape[-2:]).contiguous()
        
#         if self.to_train:
#             out, hidden = self.note_lstm(note_input) 
#             note_output = out.contiguous().view(initial_shape[:2] + out.shape[-2:])
#             info = overall_info.expand((note_output.shape[1:3]+overall_info.shape)).permute(2,0,1,3).contiguous()
#             note_output = torch.cat([note_output, info], dim =-1)
#             logits = self.logits(note_output) 
#             next_notes = nn.Sigmoid()(logits)      
#             return next_notes, sample_sound(next_notes)
        
#         else:
#             hidden = (torch.zeros(self.n_layers, note_input.shape[0], self.hidden_size).cuda(), 
#                       torch.zeros(self.n_layers, note_input.shape[0], self.hidden_size).cuda()) 
#             notes_list = []
#             sound_list = []
#             sound = torch.zeros(note_input[:,0:1,:].shape[:-1]+(NOTE_UNITS,)).cuda()

#             for i in range(NUM_OCTAVES*OCTAVE):
                
#                 inputs = torch.cat([note_input[:,i:i+1,:], sound], dim = -1)
#                 out, hidden = self.note_lstm(inputs, hidden) 
                
#                 info = overall_info.expand(((1,)+overall_info.shape)).permute(1,0,2).contiguous()
#                 note_output = torch.cat([out, info], dim =-1)
                
#                 logits = self.logits(note_output) 
#                 if self.apply_T:
#                     next_notes = nn.Sigmoid()(logits/self.temperature)
#                 else:
#                     next_notes = nn.Sigmoid()(logits)
                    
#                 sound = sample_sound2(next_notes)
#                 notes_list.append(next_notes)
#                 sound_list.append(sound)
                
#                 if self.apply_T:  
#                     sounds = torch.cat(sound_list, dim = 1)
# #                     print('sounds', sounds.shape)
#                     if  ((sounds[-1,:,0] != 0).sum() == 0):
#                         self.silent_time += 1
#                         if self.silent_time >= NOTES_PER_BAR:
#                             self.temperature += 0.1
#                     else:
#                         self.silent_time = 0
#                         self.temperature = 1
                
#             out = torch.cat(notes_list, dim = 1)
#             sounds = torch.cat(sound_list, dim = 1)
#             note_output = out.contiguous().view(initial_shape[:2] + out.shape[-2:])
#             sounds = sounds.contiguous().view(initial_shape[:2] + out.shape[-2:])
            
#             return note_output, sounds
    
    
# class track_feature(nn.Module):
#     def __init__(self, dropout=0.3):
#         super(self.__class__, self).__init__()        
        
#         self.overall_information = nn.Conv2d(3, OUT_CHANEL_TRACK, (32, 48), padding=0, stride=16)
#         self.l = nn.Linear(OUT_CHANEL_TRACK*7, NUM_TRACK_FEATURE)
        
#     def forward(self, notes):
              
#         overall_info = notes.permute(0, 3, 1, 2).contiguous()
#         overall_info = self.overall_information(overall_info) #F.relu(
# #         print('overall_info', overall_info.shape)
# # F.relu( nn.LeakyReLU(negative_slope=0.1)
#         overall_info = nn.LeakyReLU(negative_slope=0.1)(self.l(overall_info.view((overall_info.shape[0],-1))))
        
#         return overall_info     
    
# class Generator(nn.Module):
#     def __init__(self):
#         super(self.__class__, self).__init__()        
        
#         self.time_ax = time_axis() 
#         self.note_ax = note_axis()
#         #in_ch, out_ch, kernel
#         self.overall_information = track_feature()
        
#     def forward(self, notes, chosen = None, beat = None):

#         overall_info = self.overall_information(notes)
                                                              
#         note_ax_output = self.time_ax(notes, beat)
#         output = self.note_ax(note_ax_output, chosen, overall_info)
                                                                     
#         return output 