import numpy as np
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import RedditDataset
import torch

dgl.load_backend('pytorch')
data = RedditDataset(self_loop=True)

