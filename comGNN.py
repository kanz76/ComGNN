import torch 
from torch import nn 
import torch.nn.functional as F
from torch_geometric import nn as pyg_nn 
from layers import MyLinear


class ComGNN(nn.Module):
    def __init__(self, args):
        super().__init__() 
        self.emb_dim = args.emb_dim 
        self.processor = ComGNNBlock(self.emb_dim) 
        
    
    def forward(self, graph):
        cell_d = graph.cell_d 
        cell_wd, elev , rains= cell_d[..., 0], cell_d[...,1], cell_d[..., 2]
        cell_wd = cell_wd.unsqueeze(-1)
        rains = rains.unsqueeze(-1)
        elev = elev.unsqueeze(-1)
        
        cell_s = graph.cell_s 
        edge_index = graph.edge_index 
        valid_seq_ind = graph.seq 
        binary = graph.bin 
        binary = binary.unsqueeze(-1)
        
        seq_len = cell_wd.shape[1] - 1
        wd_h = cell_wd[:, 0]
        r_h = rains[:, 0]
        
        out_wd = []
        wd_losses = 0
        int_losses = 0
        non_zero_loss = 0
        req_qt = None 
        for i in range(1, seq_len + 1):
            wd_h, req_qt = self.processor(wd_h, r_h, cell_s, edge_index, req_qt)
            out_wd.append((wd_h).squeeze(-1))

            if torch.any(valid_seq_ind[:, i-1]):
                wd_losses += self.compute_loss(cell_wd[:, i], wd_h)
                non_zero_loss += F.relu(-wd_h).mean()
            r_h = rains[:, i]
        
        out_wd = torch.stack(out_wd, dim=1)
        loss = (wd_losses + int_losses) / 2 
        return out_wd, loss, valid_seq_ind, non_zero_loss

    
    def compute_loss(self, targets, preds):
        loss = torch.where(torch.abs(targets - preds) < 1, torch.abs(targets - preds), (targets - preds) ** 2)
        
        return loss.mean()


class ComGNNBlock(nn.Module):
    def __init__(self, emb_dim):
        super().__init__() 
        self.emb_dim = emb_dim 
        
        self.cell_process = FlowGNNLayerGCN2(self.emb_dim)
        
        self.elev_pred = nn.Sequential(
            MyLinear(self.emb_dim, self.emb_dim),
            nn.Tanh(),
            MyLinear(self.emb_dim, 1),
        )

        self.rain_process = nn.Sequential(
            MyLinear(2, self.emb_dim //2),
            nn.Tanh(),
            MyLinear(self.emb_dim //2, self.emb_dim),
            nn.Tanh(),
            MyLinear(self.emb_dim , self.emb_dim ),
        )    
    
    def forward(self, wd, rain, cell_s, edge_index, req_qt):
        rain_emb = self.rain_process(torch.cat([wd, rain], dim=-1))
        cell_out, req_qt = self.cell_process(wd, cell_s, rain_emb, edge_index, req_qt)
        wd = wd + self.elev_pred(cell_out)
        
        return wd , req_qt 


class FlowGNNLayerGCN2(nn.Module):
    def __init__(self, emb_dim):
        super().__init__() 
        self.cell_in_dim = emb_dim // 2
        self.feat_in_dim = 1
        self.cell_msg = self.cell_in_dim * 1
        self.emb_dim = emb_dim 
        
        self.n_convs = 2
        self.gcn_list = nn.ModuleList(ConservationBlock(self.emb_dim) for _ in range(self.n_convs))
    
    def forward(self, wd, cell_s, rain_emb, edge_index, req_qt):
        dem = cell_s[..., [0]]
        fric = cell_s[..., [1]]
        cell_elev = wd + dem 
        temp_f = cell_elev * wd  

        cell_in = rain_emb 
        cell_out = cell_in 
        if req_qt is None: req_qt = torch.zeros_like(cell_in)
        dev_qt = req_qt
        for i in range(self.n_convs):
            cell_out, dev_qt = self.gcn_list[i](cell_out, edge_index, dem, dev_qt)
            if i + 1 < self.n_convs:
                cell_out  = torch.tanh(cell_out) 
        
        req_qt = dev_qt
        return cell_out, req_qt



class ConservationBlock(nn.Module):
    def __init__(self, emb_dim):
        super().__init__() 
        self.emb_dim = emb_dim 
        
        self.qt_layer = ComputeDevQt(emb_dim)
        self.elev_layer = ComputeWd(emb_dim)
    
    def forward(self, cell_feats, edge_index, dem, dev_qt):
        dev_qt = self.qt_layer(cell_feats, edge_index, dem) + dev_qt
        cell_feats = self.elev_layer(cell_feats, edge_index, dev_qt)
        
        return cell_feats, dev_qt
        
        

class ComputeWd(pyg_nn.MessagePassing):
    def __init__(self, emb_dim):
        super().__init__(aggr='add') 
        self.emb_dim = emb_dim
        self.dev_qt_ln = MyLinear(self.emb_dim, self.emb_dim)
    
    def forward(self, x, edge_index, dev_qt):
        in_dev_qt = self.propagate(dev_qt=dev_qt, edge_index=edge_index)
        
        total_qt = in_dev_qt - dev_qt
        total_qt = self.dev_qt_ln(total_qt)
        
        out = x + total_qt
        return out
        
    def message(self, dev_qt_j):
        return dev_qt_j


class ComputeDevQt(pyg_nn.MessagePassing):
    def __init__(self, emb_dim):
        super().__init__(aggr='add') 
        self.emb_dim = emb_dim
        
        self.cell_ln = nn.Sequential(
            MyLinear(self.emb_dim, self.emb_dim),
            nn.Tanh(),
            MyLinear(self.emb_dim , self.emb_dim)
        )
        self.dem_ln = MyLinear(1, self.emb_dim)
        

        self.cell_gate = nn.Sequential(
            MyLinear(self.emb_dim, self.emb_dim),
            nn.Tanh(),
            MyLinear(self.emb_dim, self.emb_dim),
            nn.Sigmoid()
        )
        
        
    
    def forward(self, x, edges, dem):
        
        dem = self.dem_ln(dem)
        
        self.dim_size = x.shape[0]
        
        edges = edges[[1, 0]]
        dev_qt = self.propagate(x=x, edge_index=edges, dem=dem)
        
        out = dev_qt
        
        return out 
        
    def message(self, x_i, x_j, dem_i, dem_j):
        elev_slope = (x_i + dem_i) - (x_j  + dem_j)
        elev_slope = self.cell_gate(elev_slope)
        
        out = self.cell_ln(x_i) * elev_slope 
        
        assert self.dim_size >= out.shape[0], 'Each node has only one output so the # of nodes should be geq to # edges'
        return out 
    
    def aggregate(self, inputs, index):
        node_feats = inputs 
        node_feats = super().aggregate(node_feats, index, dim_size=self.dim_size)
        
        return node_feats
