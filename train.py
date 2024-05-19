import argparse
import os
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm 
import torch
import numpy as np
from torch import optim, nn

from dataset import MyDataset, MyDataLoader, load_data
import custom_metrics as c_metrics
import comGNN 
import sys 
sys.path.append('..')



def do_compute(model, batch, device):
    batch = batch.to(device)
    wd_preds, loss, valid_seq_ind, non_zero_loss = model(batch)
    wd_targets = batch.cell_d[:, 1:, 0]
    bin_targets = batch.bin[:, 1:]
    
    out_bin = bin_targets
    
    scale = batch.scale[0] 
    wd_preds = wd_preds * scale 
    wd_targets = wd_targets * scale 
    
    return  loss, (wd_targets, wd_preds), valid_seq_ind, non_zero_loss


def run_batch(model, optimizer, data_loader, epoch_i, desc, device):
        loss_wd = 0
        label_list, pred_list, mask_list = [], [], []
        loss_nz = 0
        
        for batch in tqdm(data_loader, desc= f'{desc} Epoch {epoch_i}'):
            loss, (elev_targets, elev_preds),  valid_seq_ind, non_zero_loss  = do_compute(model, batch, device)

            grand_loss = non_zero_loss + loss  
            if model.training:
                optimizer.zero_grad()
                grand_loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 1.)
                optimizer.step()

            loss_wd += float(loss)
            loss_nz += float(non_zero_loss) 
            
            with torch.no_grad():
                
                elev_targets = elev_targets.cpu().numpy().T
                assert elev_targets.ndim == 2
                

                elev_preds = elev_preds.cpu().numpy().T
                assert elev_preds.ndim == 2, elev_preds.shape

                label_list.append(elev_targets)
                pred_list.append(elev_preds)
                

                mask = valid_seq_ind.cpu().numpy().T
                assert np.any(mask)
                mask_list.append(mask)
        
        loss_wd /= len(data_loader)
        loss_nz /= len(data_loader)
        
        label_list = np.concatenate(label_list, axis=1)
        pred_list = np.concatenate(pred_list, axis=1)
        mask_list = np.concatenate(mask_list, axis=1)
        b_mask = np.full_like(label_list, fill_value=True).astype('bool')
        
        mask_regr = mask_list & b_mask 
        mse = c_metrics.MSE_score(label_list, pred_list, mask_regr)
        nse = c_metrics.NSE_score(label_list, pred_list, mask_regr)
        p_r2 = c_metrics.pearson(label_list, pred_list, mask_list)
        
        return loss_wd, mse, nse, p_r2, loss_nz


def print_metrics(loss_wd, mse, nse, p_r2, loss_nz):
    print('loss_wd:', f"{loss_wd:.4f}", 'loss_nz:', f"{loss_nz:.4f}", "mse:", [f"{i:.4f}" for i in mse], "nse:", [f"{i:.4f}" for i in nse], "p_r2:", [f"{i:.4f}" for i in p_r2])


def train(train_data_loader, val_data_loader):
    
    index = [19]
    for epoch_i in range(1, args.n_epochs+1):
        
        start = time.time()
        model.train()
        ## Training
        train_loss, train_mse, train_nse, train_p_r2, train_loss_nz = run_batch(model, optimizer, train_data_loader, epoch_i,  'train', args.device)
        model.eval()
        with torch.no_grad():
            if val_data_loader:
                val_loss , val_mse, val_nse,  val_p_r2, val_loss_nz = run_batch(model, optimizer, val_data_loader, epoch_i, 'val', args.device)
        
        model.train()
        
        if train_data_loader:
            print(f'\n#### Epoch {epoch_i} time {time.time() - start:.4f}s')
            print_metrics(train_loss, train_mse[index], train_nse[index], train_p_r2[index], train_loss_nz)

        if val_data_loader:
            print('#### Validation')
            print_metrics(val_loss, val_mse[index], val_nse[index], val_p_r2[index], val_loss_nz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    data_source = '.'
    parser.add_argument('--time_steps', type=int, default=40, help='Total number of time steps.')
    parser.add_argument('--drop', type=float, default=0.3, help='Dropout probability.')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--offset', type=int, default=10)
    parser.add_argument('--n_conv_layers', type=int, default=2)
    parser.add_argument('--valid_data', type=str, default='sims_harvey.npz')
    parser.add_argument('--train_data', type=str, default='whiteoak_harvey.npz')

    args = parser.parse_args()

    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.time_stamp = f'{datetime.now()}'.replace(':', '_')
    
    train_dataset, val_dataset = load_data(args.train_data, args.valid_data, args.offset, args.time_steps)
    train_loader = MyDataLoader(train_dataset,  shuffle=True) 
    valid_loader = MyDataLoader(val_dataset) 
    sample = train_dataset[0]
    
    model = comGNN.ComGNN(args)
    args.model_name = model.__class__.__name__
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(args.model_name)
    
    model.to(device=args.device)

    print(f'Training on {args.device}.')
    print(f'Starting  at', args.time_stamp)

    print(args)
    print(f'Train on {len(train_dataset)}, Validating on {len(val_dataset)}')
    train(train_loader, valid_loader)
