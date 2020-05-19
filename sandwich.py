import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets import SAGENet
from krnn import KRNN



class GCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, num_nodes, gcn_type, normalize):
        super(GCNBlock, self).__init__()
        # GCNUnit = {'sage': SAGENet, 'gat': GATNet,
        #            'gated': GatedGCNNet, 'egnn': EGNNNet}.get(gcn_type)
        GCNUnit = {'sage': SAGENet}.get(gcn_type)
        self.gcn = GCNUnit(in_channels=in_channels,
                           out_channels=spatial_channels,
                           normalize=normalize)

    def forward(self, blocks, node_feats, edge_feats):
        """

        Args:
            blocks:
            node_feats:
            edge_feats:

        Returns:

        """
        t1 = node_feats.permute(0, 2, 1, 3).contiguous(
        ).view(-1, node_feats.shape[1], node_feats.shape[3])
        t2 = F.relu(self.gcn(blocks, t1, edge_feats))
        out = t2.view(node_feats.shape[0], node_feats.shape[2], t2.shape[1],
                      t2.shape[2]).permute(0, 2, 1, 3)

        return out


class Sandwich(nn.Module):
    def __init__(self, num_nodes, num_features,
                 num_timesteps_input, num_timesteps_output,
                 gcn_type='sage', hidden_size=64, normalize='none', **kwargs):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(Sandwich, self).__init__()

        self.gru1 = KRNN(num_nodes, num_features, num_timesteps_input,
                         num_timesteps_output=None, hidden_size=hidden_size)

        self.gcn = GCNBlock(in_channels=hidden_size,
                            spatial_channels=hidden_size,
                            num_nodes=num_nodes,
                            gcn_type=gcn_type,
                            normalize=normalize)

        self.gru = KRNN(num_nodes, hidden_size, num_timesteps_input,
                        num_timesteps_output, hidden_size)

    def forward(self, blocks, input_nids, seed_nids, node_feat, edge_weights):
        """

        Args:
            blocks:
            input_nids:

            seed_nids:
            node_feat: Input data of shape (batch_size, num_nodes, num_timesteps,num_features=in_channels).
            edge_weights:

        Returns:

        """
        out1 = self.gru1(node_feat, input_nids)
        # out2 = self.gcn(blocks, out1, edge_weights)
        # out3 = self.gru(out2, seed_nids)
        # out3 = out3.squeeze(dim=-1)

        return out1
