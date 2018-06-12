import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from constants import *
import numpy as np
import math
import time
from IPython import display
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import optim

criterion_bce_play = nn.BCELoss()  
criterion_bce_replay = nn.BCELoss() 
criterion_mse = nn.MSELoss()  

def compute_loss(y_pred, y_true):
    
    y_pred = y_pred[:, 32:, :, :]
    y_true = y_true[:, 32:, :, :]
    
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