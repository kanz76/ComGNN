import torch 
import wandb 
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset, DataLoader 
from torch_geometric.data import Data, Batch
from tqdm import tqdm 
import copy 
from sklearn import preprocessing
import json 


def get_data(data_list, T, training=False):
    print(f'Working with {len(data_list)} datasets')
    
    data_collection_list = []
    for data, offset in data_list:
        cell_dyn = data[f'cell_feats'][:, offset:][:, :T+1] # [# of nodes x (T + 1) x 3]
        seq_len = cell_dyn.shape[1]
        
        true_seq = np.full(T, fill_value=True)

        if seq_len <= T: # +1 because we always predict the next time step
            cell_dyn_pad_s = list(cell_dyn.shape)
            cell_dyn_pad_s[1] = T + 1
            cell_dyn_padded = np.zeros(cell_dyn_pad_s)
            cell_dyn_padded[:, :cell_dyn.shape[1]] = cell_dyn

            true_seq[-1] = False  # The last entry is always False because there is no output for it
            padded_seq = np.full(T, fill_value=False)
            padded_seq[:cell_dyn.shape[1]] = true_seq
            cell_dyn = cell_dyn_padded
            true_seq = padded_seq

        sample_dict = {
            'static': data['static'],
            'edges': data['edges'],
            'seq': true_seq,
            'cell_dyn': cell_dyn,
        }
        
        if training:
            step = 40
            temp_sample = [
                {'static': sample_dict['static'],
                 'edges': sample_dict['edges'],
                 'seq': sample_dict['seq'][(i-step):i],
                 'cell_dyn': sample_dict['cell_dyn'][:, (i-step):i + 1] 
                
                } for i in list(range(0, T+step, step))[1:]
            ]
            sample_dict = temp_sample

        if training: 
            data_collection_list.extend(sample_dict)
        else:
            data_collection_list.append(sample_dict)
    
    return data_collection_list


def npz_load_batch(file_name, offset):
    return [(np.load(file_name, allow_pickle=True),offset)]


def load_data(train_file, valid_file, offset, time_steps):

    print('Loading training set...')
    train_data = npz_load_batch(f'./data/{train_file}', offset=offset)
    train_data = get_data(train_data, T=time_steps, training=True)
    
    print('Loading val set...')
    val_data = npz_load_batch(f'./data/{valid_file}', offset=offset)
    val_data = get_data(val_data, T=time_steps)
    
    print('Training dataset...')
    train_dataset = MyDataset(train_data)
    print('Validation dataset...')
    val_dataset = MyDataset(val_data)

    return train_dataset, val_dataset


class MyDataset(Dataset):

    def __init__(self, data):
        self.data = self.build_graph_data(data)
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    def collate(self, batch):
        return Batch.from_data_list(batch, follow_batch=['cell_d'])
    
    def build_graph_data(self, data):
        data_list = []
        
        for d in tqdm(data, desc='Building data'):
            cell_dyn = d['cell_dyn']
            cell_s = d['static'] 
            temp_seq = np.full((cell_dyn.shape[0], d['seq'].shape[0]), fill_value=False) 
            temp_seq[:,] = d['seq']
            cell_dyn[...,1] = cell_s[...,0][...,None] + cell_dyn[...,0]

            scaler, param = self.get_normal_method(cell_s[...,0], method='std')
            cell_s[...,0] = self.transform_data(scaler, cell_s[..., 0])

            scale_factor = 10
            cell_dyn[...,0] = self.log_transform(cell_dyn[..., 0]) 
            cell_dyn[...,0] = cell_dyn[..., 0] / scale_factor 
            cell_dyn[...,1] = cell_dyn[...,0] + cell_s[...,0][...,None]
            cell_dyn[...,2] = self.log_transform(cell_dyn[..., 2])
            cell_dyn[...,2] = cell_dyn[..., 2] / scale_factor
            bin = (cell_dyn[..., 0] > 0)
            

            dem_scale = 1
            print('Scale is', param['scale'])
            sample = CustomGraph(cell_d=torch.FloatTensor(cell_dyn) / torch.tensor([dem_scale, dem_scale, dem_scale]).unsqueeze(0).unsqueeze(0), 
                                        cell_s=torch.FloatTensor(cell_s) / torch.tensor([dem_scale, 1]).unsqueeze(0),
                                        edge_index=torch.LongTensor(d['edges'].T),
                                        seq=torch.BoolTensor(temp_seq), 
                                        bin=torch.BoolTensor(bin),
                                        num_nodes=cell_dyn.shape[0], 
                                        scale=torch.tensor([scale_factor]))
            sample.sanity_check() 
            data_list.append(sample)
        
        return data_list
    
    def get_normal_method(self, data, method):
        
        if method == 'std':
            scaler = preprocessing.StandardScaler() 
            scaler.fit(data.reshape(-1, 1)) 
            param = {
                'method': 'standardScaler',
                'mean': scaler.mean_,
                'scale': scaler.scale_,
            }
            return scaler, param 
        elif method == 'min_max':
            feature_range  = (0, 2)
            scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
            scaler.fit(data.reshape(-1, 1))
            param = {
                'method': 'min_max',
                'feature_range': feature_range,
                'scale': scaler.scale_
            }
            return scaler, param
    
    def transform_data(self, scaler, data):
        shape = data.shape 
        return scaler.transform(data.reshape(-1, 1)).reshape(*shape)


    def std_norm(self, data, mean, std):
        data  = data - mean 
        if not np.allclose(std, 0):
            return data / std 
        return data 


    def log_transform(self, data, eps=1e-2):
        return np.log(1 + data/eps)
    
    
class MyDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, batch_size=1, collate_fn=dataset.collate, **kwargs)


class CustomGraph(Data):
    def __init__(self, cell_d, cell_s, edge_index, seq, bin, num_nodes, scale):
        super().__init__() 
        self.cell_d = cell_d 
        self.cell_s = cell_s 
        self.edge_index = edge_index 
        self.bin = bin
        self.seq = seq 
        self.num_nodes = num_nodes
        self.scale = scale
        
    def sanity_check(self):
        assert self.cell_d.shape[0] ==  self.cell_s.shape[0]
        assert self.edge_index[0].max() <= self.cell_d.shape[0] 
