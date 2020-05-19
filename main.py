from parse_arguments import parse_arguments
from preprocess import generate_dataset, load_nyc_sharing_bike_data
import dgl
from sandwich import Sandwich
import torch
from sampler import NSDataLoader
import torch.nn as nn
import pandas as pd
import numpy as np


def evaluate(model, val_dataloader,global_steps,tag='test'):
    model.eval()
    pred_list = []
    label_list = []
    with torch.no_grad():
        for sub_graph, batch_labels, rows in val_dataloader.make_sampling():
            batch_blocks = sub_graph['blocks']
            batch_node_feats = sub_graph['node_feats']
            batch_edge_weights = sub_graph['edge_weights']
            batch_input_nids = sub_graph['input_nids']
            batch_seed_nids = sub_graph['seed_nids']

            batch_preds = model(batch_blocks,
                                batch_input_nids,
                                batch_seed_nids,
                                batch_node_feats,
                                batch_edge_weights)

            out_dim = batch_labels.size(-1)

            index_ptr = torch.cartesian_prod(
                torch.arange(rows.size(0)),
                torch.arange(sub_graph['seed_nids'].size(0)),
                torch.arange(out_dim)
            )

            label = pd.DataFrame({
                'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
                'node_idx': sub_graph['seed_nids'][index_ptr[:, 1]].data.cpu().numpy(),
                'feat_idx': index_ptr[:, 2].data.cpu().numpy(),
                'val': batch_labels[index_ptr.t().chunk(3)].squeeze(dim=0).data.cpu().numpy()
            })

            pred = pd.DataFrame({
                'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
                'node_idx': sub_graph['seed_nids'][index_ptr[:, 1]].data.cpu().numpy(),
                'feat_idx': index_ptr[:, 2].data.cpu().numpy(),
                'val': batch_preds[index_ptr.t().chunk(3)].squeeze(dim=0).data.cpu().numpy()
            })

            pred = pred.groupby(['row_idx', 'node_idx', 'feat_idx']).mean()
            label = label.groupby(['row_idx', 'node_idx', 'feat_idx']).mean()


    pred = pd.concat(pred_list, axis=0)
    label = pd.concat(label_list, axis=0)

    pred = pred.groupby(['row_idx', 'node_idx', 'feat_idx']).mean()
    label = label.groupby(['row_idx', 'node_idx', 'feat_idx']).mean()

    loss = np.mean((pred.values - label.values) ** 2)

    print('Global steps-{0}, loss-{1}',format(global_steps,loss))



if __name__ == '__main__':
    args = parse_arguments()
    # loss_criterion = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()} \
    #     .get(args.loss_criterion)
    gcn_type = args.gcn_type
    # gcn_partition = args.gcn_partition
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    num_timesteps_input = args.num_timesteps_input
    num_timesteps_output = args.num_timesteps_output
    early_stop_rounds = args.early_stop_rounds
    num_neighbors = args.num_neighbors
    device = args.device

    A, X, means, stds = load_nyc_sharing_bike_data()

    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)
    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    train_inputs, train_target = generate_dataset(X=train_original_data,
                                                  num_timesteps_input=num_timesteps_input,
                                                  num_timesteps_output=num_timesteps_output)

    val_inputs, val_target = generate_dataset(X=val_original_data,
                                              num_timesteps_input=num_timesteps_input,
                                              num_timesteps_output=num_timesteps_output)
    test_inputs, test_target = generate_dataset(X=test_original_data,
                                                num_timesteps_input=num_timesteps_input,
                                                num_timesteps_output=num_timesteps_output)
    num_features = train_inputs.shape[3]
    print(train_inputs.shape)
    print(train_target.shape)

    print(val_inputs.shape)
    print(val_target.shape)

    A = torch.from_numpy(A)

    sprase_A = A.to_sparse()

    edge_index = sprase_A.indices().numpy()
    edge_weight = sprase_A.values()

    graph = dgl.graph(data=(edge_index[0], edge_index[1]))
    # print(graph)

    # # add self loop
    # dgl.add_self_loop()

    graph.edata['weight'] = edge_weight

    # graph.ndata['train_feat'] = train_inputs
    # graph.ndata['val_feat'] = val_inputs
    # graph.ndata['test_feat'] = test_inputs

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    print('-----------------<Test>----------------')
    print('Input Graph num of nodes:{0}, num of edges: {1}, average degrees:{2}.'.format(num_nodes, num_edges,
                                                                                         num_edges / num_nodes))

    train_dataloader = NSDataLoader(graph=graph, node_feats=train_inputs, labels=train_target, batch_size=batch_size,
                                    sample_size=50, num_neighbors=num_neighbors, device=device,
                                    edge_feat_name='weight')
    val_dataloader = NSDataLoader(graph=graph, node_feats=val_inputs, labels=val_target, batch_size=batch_size,
                                  sample_size=50, num_neighbors=num_neighbors, device=device,
                                  edge_feat_name='weight')

    model = Sandwich(num_nodes=num_nodes,
                     num_features=num_features,
                     num_timesteps_input=num_timesteps_input,
                     num_timesteps_output=num_timesteps_output,
                     gcn_type=gcn_type, normalize='none')
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    global_steps = 0
    evaluate_step = 10
    for sub_graph, batch_labels, rows in train_dataloader.make_sampling():
        model.train()
        batch_blocks = sub_graph['blocks']
        batch_node_feats = sub_graph['node_feats']
        batch_edge_weights = sub_graph['edge_weights']
        batch_input_nids = sub_graph['input_nids']
        batch_seed_nids = sub_graph['seed_nids']
        # print('batch node feature', batch_node_feats.shape)
        # print('input nids:', sub_graph['input_nids'].shape)
        # print('seed nids:', sub_graph['seed_nids'].shape)
        # print('labels',batch_labels.shape)
        batch_preds = model(batch_blocks,
                            batch_input_nids,
                            batch_seed_nids,
                            batch_node_feats,
                            batch_edge_weights)
        loss = loss_fn(batch_preds, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_steps += 1

        if evaluate_step % 10 == 0:
            evaluate(model,val_dataloader,global_steps)
