import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *


class SAGENet(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, hidden_units=16,normalize=None):
        super(SAGENet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.conv1 = SAGELayer(in_channels, hidden_units)
        # for i in range(1, self.num_layers - 1):
        #     self.layers.append(SAGELayer(hidden_units,hidden_units))
        self.conv2 = SAGELayer(hidden_units, out_channels)
        self.normalize = normalize

    def forward(self, blocks, node_feat, edge_weight):
        # TODO: swap node to dim 0
        node_feat = node_feat.permute(1, 0, 2)

        block1 = blocks[0]
        block2 = blocks[1]

        # block/subgraph node_src_feat, node_dst_feat, edge_weight
        conv1_output = self.conv1(block1, node_feat, node_feat[:block1.number_of_dst_nodes()], edge_weight[0])
        conv1_output = F.leaky_relu(conv1_output)
        conv2_output = self.conv2(block2, conv1_output, conv1_output[:block2.number_of_dst_nodes()], edge_weight[1])
        conv2_output = F.leaky_relu(conv2_output)

        conv2_output = conv2_output.permute(1, 0, 2)
        return conv2_output

class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        self.conv1 = GATLayer(in_channels=in_channels, out_channels=16)
        self.conv2 = GATLayer(in_channels=16, out_channels=out_channels)

    def forward(self, blocks, node_feat, edge_weight):
        node_feat = node_feat.permute(1, 0, 2)

        block1 = blocks[0]
        block2 = blocks[1]

        conv1_output = self.conv1(block1, node_feat, node_feat[:block1.number_of_dst_nodes()], edge_weight[0])
        conv1_output = F.leaky_relu(conv1_output)
        conv2_output = self.conv2(block2, conv1_output, conv1_output[:block2.number_of_dst_nodes()], edge_weight[1])
        conv2_output = F.leaky_relu(conv2_output)

        conv2_output = conv2_output.permute(1, 0, 2)
        return conv2_output





