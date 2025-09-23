# Downstrean evaluation pipeline from LayerDAG
# https://github.com/Graph-COM/LayerDAG/tree/main/src/eval

import os
from io import StringIO
import re
import torch
import networkx as nx


from tpu_tile import TPUTileEvaluator
from src.layerdag_eval.dataset import DAGDataset

sample_dir = 'tpu_tile_samples'
os.makedirs(sample_dir, exist_ok=True)

def parse_graphs(file_name, labels_file):
    import time
    start_time = time.time()

    with open(file_name, 'r') as f:
        data = f.read()

    train_syn_set = DAGDataset(num_categories=47, label=True)
    val_syn_set = DAGDataset(num_categories=47, label=True)

    labels = torch.load(labels_file)

    # Split and iterate over graph blocks
    graph_blocks = data.strip().split('N=')[1:]

    for idx, block in enumerate(graph_blocks):
        lines = block.strip().split('\n')
        n = int(lines[0])

        try:
            x_start = lines.index('X: ') + 1
            e_start = lines.index('E: ')
        except ValueError:
            raise ValueError(f"Malformed graph block at index {idx}")

        # Parse node features (X)
        x_lines = lines[x_start:e_start]
        x_n_list = [int(x) for line in x_lines for x in line.strip().split()]

        # Parse adjacency matrix (E)
        e_lines = lines[e_start + 1:]
        e_values = [int(x) for line in e_lines for x in line.strip().split()]
        e_tensor = torch.tensor(e_values).reshape(n, n)

        # Get edges
        src_list, dst_list = (e_tensor == 1).nonzero(as_tuple=True)
        src_list, dst_list = src_list.tolist(), dst_list.tolist()

        # Build directed graph using networkx for DAG check and topo sort
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(zip(src_list, dst_list))

        if not nx.is_directed_acyclic_graph(G):
            print(f"[Warning] Skipping non-DAG graph at index {idx}")
            continue  # Skip non-DAG graphs

        # Topological sort
        topo_order = list(nx.topological_sort(G))

        x_n_list = [x_n_list[i] for i in topo_order]

        # Mapping from old to new indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(topo_order)}
        src_list = torch.tensor(
            [old_to_new[i] for i in src_list], dtype=torch.long
        )
        dst_list = torch.tensor(
            [old_to_new[i] for i in dst_list], dtype=torch.long
        )

        if len(src_list) > 0:
            assert max(src_list) < len(x_n_list)
            assert max(dst_list) < len(x_n_list)
        assert len(src_list) == len(dst_list)

        # Label
        y_list = [labels[idx].item()] if idx < len(labels) else [0]

        # Store
        dataset = train_syn_set if idx < 5040 else val_syn_set
        dataset.add_data(
            torch.tensor(src_list),
            torch.tensor(dst_list),
            torch.tensor(x_n_list),
            torch.tensor(y_list)
        )

    print(f"[INFO] Parsed {len(train_syn_set)} train and {len(val_syn_set)} val graphs in {time.time() - start_time:.2f}s")
    return train_syn_set, val_syn_set


evaluator = TPUTileEvaluator()

# Load the dataset
graphs_file = 'graphs.txt' # Path to your graphs file
labels_file = 'labels_tensor.pt' # Path to your labels file
train_syn_set, val_syn_set = parse_graphs(graphs_file, labels_file)

evaluator.eval(train_syn_set, val_syn_set)