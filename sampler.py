from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from dgl.sampling import sample_neighbors
from dgl.contrib.sampling import NeighborSampler
import torch
import numpy as np
import multiprocessing
import dgl

NUM_WORKERS = multiprocessing.cpu_count()


def load_subgraph(g, labels, node_feats, seed_nodes, device, input_nodes=None, input_edges=None,
                  edge_feat_name=None):
    """
    Load subgraph and corresponding features onto current device.
    Args:
        g: DGLGraph
        node_feats:
        labels: tensor
        seed_nodes:
        device:
        input_nodes:
        input_edges:
        node_feat_name: str,
        edge_feat_name: str,

    Returns:
        batch_node_feats:
        batch_edge_feats: [edge_feats]

    """
    # TODO
    batch_labels = labels[:, seed_nodes].to(device)
    if edge_feat_name:
        batch_node_feats = node_feats[:, input_nodes].to(device)
        batch_edge_feats = [g.edata[edge_feat_name][edge].to(device) for edge in input_edges]

        return batch_node_feats, batch_edge_feats, batch_labels
    elif node_feats:
        batch_node_feats = node_feats[:, input_nodes].to(device)
        return batch_node_feats, batch_labels
    else:
        raise ValueError


# official recommend
class NSDataLoader(object):
    """
    Neighbour sampler based on dgl.sampling.sample_neighbor and dgl official implementation of GraphSAGE.
    Examples:

    Args:
        graph: DGLGraph
        batch_size: int, batch_size
        num_neighbors: int or list of int, nums of neighbors sampled in each layer.
        device: str, specify device
        node_feat_name: str, optional
        edge_feat_name: str, optional
        shuffle: boolean, optional,
        drop_last: boolean, optional, set to ``True`` to drop the last incomplete batch,
                   if the dataset size is not divisible by the batch size. If ``False`` and
                   the size of dataset is not divisible by the batch size, then the last batch
                   will be smaller. (default: ``False``)
        num_workers: int, optional, default num of cpu workers.
        edge_type: str, optional Indicates the neighbors on different types of edges.
                   * "in": the neighbors on the in-edges.
                   * "out": the neighbors on the out-edges.
        replace: bool, optional, If True, sample with replacement.

    """

    def __init__(self, graph, node_feats, labels, batch_size, sample_size, num_neighbors, device,
                 edge_feat_name=None, shuffle=True,
                 drop_last=False, num_workers=NUM_WORKERS, edge_type='in', replace=True):
        self.graph = graph
        self.node_feats = node_feats
        self.labels = labels
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.num_neighbors = num_neighbors

        self.edge_feat_name = edge_feat_name
        self.device = device
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.replace = replace
        self.edge_type = edge_type
        num_nodes = graph.number_of_nodes()
        self.feats_length = len(node_feats)
        print('Feats length:', self.feats_length)
        self.nodes_ids = torch.from_numpy(np.array(range(num_nodes)))

    def sample_blocks(self, nodes):
        """
        Args:
            nodes: seed nodes
        Returns:
        """
        nodes = torch.LongTensor(np.asarray(nodes))
        blocks = []
        edge_ids = []
        for num in self.num_neighbors:
            frontier = sample_neighbors(self.graph, nodes, num, self.edge_type, replace=self.replace)
            # frontier: DGLHeteroGraph, A sampled subgraph containing only the sampled neighbor edges from ``nodes``.
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, nodes)
            # Obtain the seed nodes for next layer.
            nodes = block.srcdata[dgl.NID]
            edge_ids.insert(0, frontier.edata[dgl.EID][block.edata[dgl.EID]])
            blocks.insert(0, block)
        return blocks, edge_ids

    def make_sampling(self):
        """

        Args:
            nodes_id:
            labels: tensor, all training labels

        Yields:
            blocks: [block_2hop,block_1hop]
            batch_node_feats:
            batch_edge_feats:
            batch_labels:
        """
        dataLoader = DataLoader(dataset=self.nodes_ids,
                                batch_size=self.sample_size,
                                collate_fn=self.sample_blocks,
                                shuffle=True,
                                drop_last=self.drop_last,
                                num_workers=self.num_workers)

        for blocks, edge_ids in dataLoader:
            sub_graph = dict()
            input_nodes = blocks[0].srcdata[dgl.NID]  # The nodes for input lies at the Left side of the first block.
            seed_nodes = blocks[-1].dstdata[dgl.NID]  # The nodes for output lies at the right side of the last block.

            sub_graph['blocks'] = blocks
            sub_graph['input_nids'] = input_nodes
            sub_graph['seed_nids'] = seed_nodes
            sample_node_feats, batch_edge_feats, sample_labels = load_subgraph(g=self.graph,
                                                                               node_feats=self.node_feats,
                                                                               labels=self.labels,
                                                                               seed_nodes=seed_nodes,
                                                                               device=self.device,
                                                                               input_nodes=input_nodes,
                                                                               input_edges=edge_ids,
                                                                               edge_feat_name=self.edge_feat_name)

            sub_graph['edge_weights'] = batch_edge_feats
            # sub_graph['node_feats'] = batch_node_feats
            #
            # yield sub_graph, batch_labels
            indices = np.arange(self.feats_length)
            num_batches = (self.feats_length + self.batch_size - 1) // self.batch_size
            for batch_id in range(num_batches):
                start = batch_id * self.batch_size
                end = (batch_id + 1) * self.batch_size
                sub_graph['node_feats'] = sample_node_feats[indices[start: end]]
                batch_labels = sample_labels[indices[start: end]]
                yield sub_graph, batch_labels, indices[start: end]

# class NeighborSamplingDataLoader(object):
#     """
#     Neighbour sampler based on dgl.contrib.sampling.NeighbourSampler and gcn
#     It do not support sampling different num of neighbours in different hops of neighbour.
#     Examples:
#
#         sampler = NeighborSamplingDataLoader(...)
#         for nf, batch_labels in sampler.make_sampling(device):
#             pred = model(nf)
#             loss = loss_fcn(pred, batch_labels)
#
#     Args：
#         graph: DGLGraph
#         labels: Tensor, labels.
#         batch_size: int, batch_size.
#         num_neighbors: int, num of neighbor sampled in each layer.
#                        Note do not support sampling different num of neighbors in different layers.
#         edge_type:  str, optional Indicates the neighbors on different types of edges.
#                     * "in": the neighbors on the in-edges.
#                     * "out": the neighbors on the out-edges.
#         num_workers: int, optional, default num of cpu workers
#         use_prefetch: boolean, optional, default True
#         add_self_loop:boolean, optional, default True
#
#     Returns：
#         nf: NodeFlow
#         batch_labels: tensor labels of current batch.
#     """
#
#     def __init__(self, graph, labels,
#                  batch_size, num_neighbors, num_hops, seed_nodes=None,
#                  edge_type='in', num_workers=NUM_WORKERS, use_prefetch=True, add_self_loop=True):
#         self.labels = labels
#         self.graph = graph
#         self.sampler = NeighborSampler(g=self.graph,
#                                        batch_size=batch_size,
#                                        expand_factor=num_neighbors,
#                                        num_hops=num_hops,
#                                        neighbor_type=edge_type,
#                                        seed_nodes=seed_nodes,
#                                        num_workers=num_workers,
#                                        prefetch=use_prefetch,
#                                        add_self_loop=add_self_loop)
#
#     def make_sampling(self, device):
#         for nf in self.sampler:
#             nf.copy_from_parent()
#             batch_nids = nf.layer_parent_nid(-1).to(device=device, dtype=torch.long)
#             batch_labels = self.labels[batch_nids]
#             yield nf, batch_labels
