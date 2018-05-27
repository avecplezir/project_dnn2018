import torch,torch.nn as nn

def EB(sound):
    play = sound[:,:,:,0]
    num_notes = play.sum(dim=-1)
    is_empty = (num_notes==0).float()
    return is_empty.mean()*100
    
def UPC(sound):
    play = sound[:,:,:,0]
    batch_size,time_scale,note_num = play.size()
    octave_play = play.view(batch_size,time_scale,12,note_num//12)
    pitch_play = octave_play.sum(dim = -1)>0
    num_pitches = pitch_play.sum(dim=-1)
    return num_pitches.float().mean()

def num_notes(play,replay):
    play = play*(1-replay)
    batch_size,time_scale,note_num = play.size()
    shifted_play = torch.zeros(batch_size,time_scale,note_num).cuda()
    shifted_play[:,:-1,:] = play[:,1:,:]
    cond_change = shifted_play-play
    return (cond_change>0).float().sum()

def QN(sound, qualified_len = 3):
    batch_size,time_scale,note_num,_ = sound.size()
    play = sound[:,:,:,0]
    replay = sound[:,:,:,1]
    shifted_play = torch.zeros(batch_size,time_scale,note_num).cuda()
    
    shifted_play[:,:time_scale-qualified_len,:] = play[:,qualified_len:,:]
    qualified_play = play*shifted_play
    
    general_num_notes = num_notes(play,replay)
    num_qualified_notes = num_notes(qualified_play,replay)
    return num_qualified_notes/general_num_notes*100
    