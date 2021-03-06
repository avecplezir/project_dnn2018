import os

# Define the musical styles
genre = [
    #'baroque',
    'classical',
    #'romantic'
]

# where to look for songs to train nn on
styles = [
       [
        'data/test',
    ],
]

#=============================
#self attention - number of heads, projection dim
cuda = True

ATTENTION_NOTE_AXIS = True
ATTENTION_TIME_AXIS = False
PROJECTION_DIM = 97 #94
N_HEADS = 3
D_MODEL = 97 #78 #94 #78
BEATS_FEATURES = 16
NUM_TRACK_FEATURE = 30
OUT_CHANEL_TRACK = 20

#=============================

NUM_STYLES = sum(len(s) for s in styles)

# MIDI Resolution
DEFAULT_RES = 96
MIDI_MAX_NOTES = 128
MAX_VELOCITY = 127

# Number of octaves supported
NUM_OCTAVES = 4
OCTAVE = 12

# Min and max note (in MIDI note number)
MIN_NOTE = 36
MAX_NOTE = MIN_NOTE + NUM_OCTAVES * OCTAVE
NUM_NOTES = MAX_NOTE - MIN_NOTE

# Number of beats in a bar
BEATS_PER_BAR = 4
# Notes per quarter note
NOTES_PER_BEAT = 4
# The quickest note is a half-note
NOTES_PER_BAR = NOTES_PER_BEAT * BEATS_PER_BAR

# Training parameters
BATCH_SIZE = 32
SEQ_LEN = 8 * NOTES_PER_BAR

# Hyper Parameters
OCTAVE_UNITS = 64

NOTE_UNITS = 3
TIME_AXIS_UNITS = 256
NOTE_AXIS_UNITS = 128

TIME_AXIS_LAYERS = 2
NOTE_AXIS_LAYERS = 2

# Move file save location
OUT_DIR = '/data/i.anokhin/music_out'
MODEL_DIR = os.path.join(OUT_DIR, 'models')
SAMPLES_DIR = os.path.join(OUT_DIR, 'samples')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
