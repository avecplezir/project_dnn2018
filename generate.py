import numpy as np
from collections import deque
import midi
import argparse

from constants import *
# from util import *
from dataset import unclamp_midi, compute_beat
from tqdm import tqdm
from midi_util import midi_encode

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MusicGeneration:
    """
    Represents a music generation
    """
    def __init__(self, default_temp=1):
        self.notes_memory = deque([np.zeros((NUM_NOTES, NOTE_UNITS)) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.beat_memory = deque([np.zeros(NOTES_PER_BAR) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        
        # The next note being built
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        self.silent_time = NOTES_PER_BAR

        # The outputs
        self.results = []
        # The temperature
        self.default_temp = default_temp
        self.temperature = default_temp

    def build_time_inputs(self):
        return (
            np.array(self.notes_memory),
#             np.array(self.beat_memory),
        )

    def build_note_inputs(self, note_features):
        # Timesteps = 1 (No temporal dimension)
        return (
            np.array(note_features),
            np.array([self.next_note]),
        )

    def choose(self, prob, n):
        vol = prob[n, -1]
        prob = apply_temperature(prob[n, :-1], self.temperature)

        # Flip notes randomly
        if np.random.random() <= prob[0]:
            self.next_note[n, 0] = 1
            # Apply volume
            self.next_note[n, 2] = vol
            # Flip articulation
            if np.random.random() <= prob[1]:
                self.next_note[n, 1] = 1

    def end_time(self, t):
        """
        Finish generation for this time step.
        """
        # Increase temperature while silent.
        if np.count_nonzero(self.next_note) == 0:
            self.silent_time += 1
            if self.silent_time >= NOTES_PER_BAR:
                self.temperature += 0.1
        else:
            self.silent_time = 0
            self.temperature = self.default_temp

        self.notes_memory.append(self.next_note)
        # Consistent with dataset representation
        self.beat_memory.append(compute_beat(t, NOTES_PER_BAR))
        self.results.append(self.next_note)
        # Reset next note
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        return self.results[-1]

def apply_temperature(prob, temperature):
    """
    Applies temperature to a sigmoid vector.
    """
    # Apply temperature
    if temperature != 1:
        # Inverse sigmoid
        x = -np.log(1 / prob - 1)
        # Apply temperature to sigmoid function
        prob = 1 / (1 + np.exp(-x / temperature))
    return prob

def process_inputs(ins):
    ins = list(zip(*ins))
    ins = [np.array(i) for i in ins]
    return ins

def generate(models, num_bars, Attention = False):
    print('Generating with no styles:')

    models.train(False) 
    time_model, note_model = models.time_ax, models.note_ax
    generations = [MusicGeneration()]

    for t in tqdm(range(NOTES_PER_BAR * num_bars)):
        # Produce note-invariant features
        ins = process_inputs([g.build_time_inputs() for g in generations])[0]
#         print('ins', ins.shape)
        g = generations[0]
        
        if cuda:      
            ins = Variable(torch.FloatTensor(ins)).cuda()
        else:
            ins = Variable(torch.FloatTensor(ins))
            
        # Pick only the last time step
        note_features = time_model(ins)
#         print('note_features', note_features.shape)
        note_features = note_features[:, -1:, :]
#         print('note_features', note_features.shape)

        # Generate each note conditioned on previous       
        for n in range(NUM_NOTES):
            if cuda:      
                current_note = Variable(torch.FloatTensor([[g.next_note]])).cuda()
            else:
                current_note = Variable(torch.FloatTensor([[g.next_note]]))
             
#             print('current_note', current_note.shape)
            predictions = note_model(note_features, current_note, to_train=True)
#             print('predictions', predictions.shape)

            predictions = predictions.cpu().data.numpy()
            for i, g in enumerate(generations):
                # Remove the temporal dimension
                g.choose(predictions[i][-1], n)

        # Move one time step
        yield [g.end_time(t) for g in generations]

def write_file(name, results):
    """
    Takes a list of all notes generated per track and writes it to file
    """
    results = zip(*list(results))

    for i, result in enumerate(results):
        fpath = os.path.join(SAMPLES_DIR, name + '_' + str(i) + '.mid')
        print('Writing file', fpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        mf = midi_encode(unclamp_midi(result))
        midi.write_midifile(fpath, mf)


def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--bars', default=4, type=int, help='Number of bars to generate')
    args = parser.parse_args()

    models = build_or_load()

    write_file('output', generate(models, args.bars))

if __name__ == '__main__':
    main()
