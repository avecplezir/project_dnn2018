import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from constants import *
import numpy as np
import math
from util import one_hot

def get_variable(x):
    if cuda:      
        return  Variable(x.cuda(), requires_grad=False)
    else:
        return Variable(x, requires_grad=False)
           
# def beat_f(x):

#     beats = torch.FloatTensor(np.array([one_hot(b % NOTES_PER_BAR, NOTES_PER_BAR) for b in range(x.shape[1])]))
#     beats = beats.repeat(((x.shape[0], x.shape[2])+ (1, 1)))
#     beats = beats.permute(0,2,1,3).contiguous()

#     return get_variable(beats)

# def pitch_pos_in_f(x):
#     """
#     Returns a constant containing pitch position of each note
#     """
#     pos_in = torch.FloatTensor(np.arange(NUM_NOTES)/NUM_NOTES)
#     pos_in = pos_in.repeat(x.shape[:-2]+(1,))[:,:,:,None]
    
#     return get_variable(pos_in)

# def pitch_class_in_f(x):
#     """
#     Returns a constant containing pitch class of each note
#     """

#     pitch_class_matrix = np.array([one_hot(n % OCTAVE, OCTAVE) for n in range(NUM_NOTES)])
#     pitch_class_matrix = torch.FloatTensor(pitch_class_matrix)
#     pitch_class_matrix = pitch_class_matrix.view(1, 1, NUM_NOTES, OCTAVE)
#     pitch_class_matrix = pitch_class_matrix.repeat((x.shape[:2]+ (1, 1)))
    
#     return get_variable(pitch_class_matrix)

# def pitch_bins_f(x):

#         bins = [x[:, :, i::OCTAVE, :1].sum(2) for i in range(OCTAVE)]
#         bins = torch.cat(bins, dim = -1)
#         bins = bins.repeat(NUM_OCTAVES, 1, 1)
#         bins = bins.view(x.shape[:2]+(NUM_NOTES, 1))
        
#         return bins
            
# class feature_generation(nn.Module):
#     def __init__(self):
#         super(self.__class__, self).__init__()        
#         self.padding = nn.ZeroPad2d(((2 * OCTAVE - 1)//2,  math.ceil((2 * OCTAVE - 1)/2), 0, 0))
#         self.conv = nn.Conv1d(NOTE_UNITS,  OCTAVE_UNITS, 2 * OCTAVE)
        
#     def forward(self, notes):
#         initial_shape = notes.shape
        
#         # convolution
#         notes = notes.contiguous().view((-1,)+ notes.shape[-2:]).contiguous()
#         notes = notes.permute(0, 2, 1).contiguous()
#         notes = self.padding(notes)
#         notes = self.conv(notes)
#         notes = nn.Tanh()(notes)
#         notes = notes.permute(0, 2, 1).contiguous()
#         notes = notes.contiguous().view(initial_shape[:2] + notes.shape[-2:])
        
#         pos_in = pitch_pos_in_f(notes)
#         class_in = pitch_class_in_f(notes)
#         bins = pitch_bins_f(notes)        
#         note_features = torch.cat([notes, pos_in, class_in, bins], dim = -1) 
    
#         return note_features
    
#===================================================================================================

def beat_f(x):

    beats = torch.LongTensor(np.array([b % NOTES_PER_BAR for b in range(x.shape[1])]))
    beats = beats.repeat(((x.shape[0], x.shape[2])+ (1,)))
    beats = beats.permute(0,2,1).contiguous()

    return get_variable(beats)

def pitch_pos_in_f(x):
    """
    Returns a constant containing pitch position of each note
    """
    pos_in = torch.LongTensor(np.arange(NUM_NOTES))
    pos_in = pos_in.repeat(x.shape[:-2]+(1,))
    
    return get_variable(pos_in)

def pitch_class_in_f(x):
    """
    Returns a constant containing pitch class of each note
    """

    pitch_class_matrix = np.array([n % OCTAVE for n in range(NUM_NOTES)])
    pitch_class_matrix = torch.LongTensor(pitch_class_matrix)
    pitch_class_matrix = pitch_class_matrix.repeat((x.shape[:2]+ (1,)))
    
    return get_variable(pitch_class_matrix)

def pitch_bins_f(x):

        bins = [x[:, :, i::OCTAVE, :1].sum(2) for i in range(OCTAVE)]
        bins = torch.cat(bins, dim = -1)
        bins = bins.repeat(NUM_OCTAVES, 1, 1)
        bins = bins.view(x.shape[:2]+(NUM_NOTES, 1))
        
        return bins
    
    
class feature_generation(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()        
        self.padding = nn.ZeroPad2d(((2 * OCTAVE - 1)//2,  math.ceil((2 * OCTAVE - 1)/2), 0, 0))
        self.conv = nn.Conv1d(NOTE_UNITS,  OCTAVE_UNITS, 2 * OCTAVE)
        
        self.pos_embedding =nn.Embedding(num_embeddings=NUM_NOTES, embedding_dim = 20)
        self.pitch_class_embedding =nn.Embedding(num_embeddings=OCTAVE, embedding_dim = OCTAVE)
        
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
#         print('pos_in', pos_in.shape)
        class_in = pitch_class_in_f(notes)
#         print('class_in', class_in.shape)
        bins = pitch_bins_f(notes)  
#         print('bins', bins.shape)

        pos_in = self.pos_embedding(pos_in)
        class_in = self.pitch_class_embedding(class_in)
        #         print('pos_in', pos_in.shape)
        #         print('class_in', class_in.shape)
        #         print('notes', notes.shape)
        
        note_features = torch.cat([notes, pos_in, class_in, bins], dim = -1) 
        
#         print('note_features',note_features.shape)
    
        return note_features

