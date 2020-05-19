import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class SAGELayer(nn.Module):

    def __init__(self, in_channels, out_channels, aggre_type='gcn', bias=True, ):

        super(SAGELayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self._aggre_type = aggre_type

        self.fc_neigh = nn.Linear(self.in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, block, feat_src, feat_dst, edge_weight):
        """
        Args:
            block:
            feat_src
            feat_dst:
            edge_weight:

        Returns:
        """
        block = block.local_var()
        if self._aggre_type == 'gcn':

            block.srcdata['h'] = feat_src
            block.dstdata['h'] = feat_dst
            block.edata['w'] = edge_weight
            block.update_all(fn.src_mul_edge('h', 'w', 'm'), fn.sum('m', 'neigh'))

            degs = block.in_degrees().to(feat_dst)
            h_neigh = (block.dstdata['neigh'] + block.dstdata['h']) / (degs.unsqueeze(-1) + 1)
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

        if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
        else:
            raise KeyError

        return rst


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels=1, normalize='none'):
        super(GATLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels

        self.weight_n = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_e = nn.Parameter(torch.Tensor(edge_channels, out_channels))

        self.u = nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.v = nn.Parameter(torch.Tensor(out_channels, out_channels))

        self.normalize = normalize

        if normalize == 'bn':
            self.batch_norm = nn.BatchNorm1d(out_channels)
        if normalize == 'ln':
            self.layer_norm = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_n)
        nn.init.xavier_uniform_(self.weight_e)
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    def forward(self, block, feat_src, feat_dst, edge_weight):

        if len(edge_weight.shape) == 1:
            edge_weight = edge_weight.unsqueeze(dim=-1)

        edge_weight = torch.matmul(edge_weight, self.weight_e)

        block.srcdata['h'] = torch.matmul(feat_src, self.u)
        block.dstdata['h'] = torch.matmul(feat_dst, self.v)
        block.edata['weight'] = edge_weight.unsqueeze(dim=1)
        block.update_all(message_func=self.message_fn, reduce_func=fn.sum(msg='aggr', out='aggr_out'))

        aggr_out = torch.matmul(feat_dst, self.u) + block['aggr_out']

        if self.normalize == 'bn':
            aggr_out = aggr_out.permute(0, 2, 1)
            aggr_out = self.batch_norm(aggr_out)
            aggr_out = aggr_out.permute(0, 2, 1)

        elif self.normalize == 'ln':
            aggr_out = self.layer_norm(aggr_out)
        elif self.normalize == 'vn':
            mean = aggr_out.view(aggr_out.size(0), -1). \
                mean(dim=[0, 2], keepdim=True)
            std = aggr_out.view(aggr_out.size(0), -1). \
                std(dim=[0, 2], keepdim=True)
            aggr_out = (aggr_out - mean) / (std + 1e-5)

        else:
            raise TypeError('Norm type error')

        return aggr_out + feat_dst

    def message_fn(self, edges):

        gate = F.sigmoid(edges.src['h'] * edges.dst['h'] * edges.data['w'].unsqueeze(dim=1))

        return {'aggr': edges.src['h'] * gate}


class EGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels=1, norm=None):
        super(EGNNLayer, self).__init__()
        # default aggr = 'add' src to dst

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels

        self.weight_n = nn.Parameter(torch.Tensor(in_channels, out_channels), requires_grad=True)
        self.weight_e = nn.Parameter(torch.Tensor(edge_channels, out_channels), requires_grad=True)

        self.query = nn.Parameter(torch.Tensor(out_channels, out_channels), requires_grad=True)
        self.key = nn.Parameter(torch.Tensor(out_channels, out_channels), requires_grad=True)

        self.linear_att = nn.Linear(3 * self.out_channels, 1)
        self.linear_out = nn.Linear(2 * self.out_channels, out_channels)

        if norm == 'bn':
            self.norm_layer = nn.BatchNorm1d(out_channels)
        elif norm == 'ln':
            self.norm_layer = nn.LayerNorm(out_channels)
        else:
            self.norm_layer = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_n)
        nn.init.xavier_uniform_(self.weight_e)
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.key)

    def forward(self, block, feat_src, feat_dst, edge_weight):
        # x[0], x[1]

        edge_emb = torch.matmul(edge_weight, self.weight_e)
