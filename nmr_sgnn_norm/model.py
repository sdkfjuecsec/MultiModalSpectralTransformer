"""Training / inference loop shared by both SGNN variants.

Keeps one epoch loop (L1 loss on standardised shifts, ReduceLROnPlateau + a hand
written early-stopping rule) and one inference routine that performs Monte-Carlo
dropout to get a predictive mean plus a per-molecule wall-clock timing.
"""

import numpy as np
import time

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from nmr_sgnn_norm.util import MC_dropout
from sklearn.metrics import mean_absolute_error

        
def training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path, n_forward_pass = 5, cuda = torch.device('cuda:0')):
    """Train ``net`` and return it reloaded from its best checkpoint.

    Shifts are standardised with the *training* set statistics before the
    forward pass, the model is evaluated with MC-dropout MAE every epoch, the
    best state dict is written to ``model_path`` and training stops after 50
    epochs without improvement. ``n_forward_pass`` is forwarded to ``inference``
    (fewer passes than at test time keeps validation cheap).
    """
    # The requested device is immediately replaced by the device the parameters
    # already live on (the caller moved the net with .cuda()).
    cuda = next(net.parameters()).device


    optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=1e-10)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-7, verbose=True)

    max_epochs = 500
    val_log = np.zeros(max_epochs)
    
    for epoch in range(max_epochs):
        # training
        net.train()
        start_time = time.time()
        for batchidx, batchdata in enumerate(train_loader):

            inputs, n_nodes, numHshifts, shift_test, shift, masks = batchdata
            
            # Standardise only the annotated atoms; unannotated ones are dropped
            # by `masks` anyway, so their raw value never reaches the loss.
            shift = (shift[masks] - train_y_mean) / train_y_std
            
            inputs = inputs.to(cuda)
            n_nodes = n_nodes.to(cuda)
            shift = shift.to(cuda)
            masks = masks.to(cuda)
            
            pred = net(inputs, n_nodes, masks)
            # Mean absolute error, i.e. L1 on standardised shifts.
            loss = torch.abs(pred - shift).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Rescale the last mini-batch loss back to ppm for the log line.
            train_loss = loss.detach().item() * train_y_std
    
        # validation
        # NOTE (known issue, left unchanged): for target '1H' this uses shift[mask],
        # i.e. the *mean* shift of every annotated carbon, while the test metric in
        # train.py repeats the prediction per proton. Validation MAE and test MAE
        # therefore measure different quantities, so early stopping / the LR
        # scheduler optimise something other than the reported test number.
        val_y = np.hstack([inst[-2][inst[-1]] for inst in iter(val_loader.dataset)])
        val_y_pred, _ = inference(net, val_loader, train_y_mean, train_y_std, n_forward_pass = n_forward_pass)
        val_loss = mean_absolute_error(val_y, val_y_pred)
        
        val_log[epoch] = val_loss
        if epoch % 10 == 0: print('--- validation epoch %d, processed %d, train_loss %.3f, current MAE %.3f, best MAE %.3f, time elapsed(min) %.2f' %(epoch, val_loader.dataset.__len__(), train_loss, val_loss, np.min(val_log[:epoch + 1]), (time.time()-start_time)/60))
        
        lr_scheduler.step(val_loss)
        
        # earlystopping
        # Keep the best checkpoint; give up once 50 epochs passed without beating it.
        if np.argmin(val_log[:epoch + 1]) == epoch:
            torch.save(net.state_dict(), model_path) 
        
        elif np.argmin(val_log[:epoch + 1]) <= epoch - 50:
            break

    print('training terminated at epoch %d' %epoch)
    # Restore the best weights so the caller evaluates the best epoch, not the last.
    net.load_state_dict(torch.load(model_path))
    
    return net
    

def inference(net, test_loader, train_y_mean, train_y_std, n_forward_pass = 30, cuda = torch.device('cuda:0')):
    """Predict shifts for ``test_loader`` with Monte-Carlo dropout.

    Returns
    -------
    tuple
        ``(mean_prediction, time_per_mol)`` — the dropout-averaged prediction
        (back-transformed to ppm) and the average seconds spent per molecule.
    """
    net.eval()
    # As in `training`, the caller-side device wins over the argument.
    cuda = next(net.parameters()).device

    # net.eval() above disables dropout, so this call flips the dropout modules
    # back to train mode: repeated forward passes then sample different sub-nets
    # (MC dropout) instead of all being identical.
    MC_dropout(net)
    tsty_pred = []
    
    start_time = time.time()
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
            inputs = batchdata[0].to(cuda)
            n_nodes = batchdata[1].to(cuda)
            masks = batchdata[-1].to(cuda)

            mean_list = []

            for _ in range(n_forward_pass):
                mean = net(inputs, n_nodes, masks)
                mean_list.append(mean.cpu().numpy())
            
            tsty_pred.append(np.array(mean_list).transpose())

    # Undo the standardisation once for the whole (atoms x forward-pass) matrix.
    tsty_pred = np.vstack(tsty_pred) * train_y_std + train_y_mean
    
    time_per_mol = (time.time() - start_time) / test_loader.dataset.__len__()

    # Average over the MC-dropout samples (axis 1 = forward pass).
    return np.mean(tsty_pred, 1), time_per_mol
