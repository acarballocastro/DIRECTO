import os
import torch

from .general import DAGDataset
import networkx as nx


def reformat_data(input_data):
    dict_data = {"src_list": [], "dst_list": [], "x_n_list": [], "y_list": []}
    data = input_data[0]
    slices = input_data[1]
    num_graphs = slices["x"].shape[0] - 1

    for n in range(num_graphs):
        # Node types
        node_features = (
            data.x[slices["x"][n] : slices["x"][n + 1]] - 1
        ).int()  # start at 0
        num_nodes = node_features.shape[0]
        # dict_data["x_n_list"].append(node_features.squeeze())
        # Processing edges
        edge_index = data.edge_index[
            :, slices["edge_index"][n] : slices["edge_index"][n + 1]
        ]
        assert edge_index[0].max() < node_features.shape[0]
        assert edge_index[1].max() < node_features.shape[0]
        assert edge_index[0].min() >= 0
        assert edge_index[1].min() >= 0

        # Build graph for topological sorting
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)

        try:
            topo_order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            raise ValueError(f"Graph {n} is not a DAG; topological sort is impossible.")

        # Reorder node features and remap edge indices
        node_features = node_features[topo_order]

        # Mapping from old to new indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(topo_order)}
        src = torch.tensor(
            [old_to_new[i] for i in edge_index[0].tolist()], dtype=torch.long
        )
        dst = torch.tensor(
            [old_to_new[i] for i in edge_index[1].tolist()], dtype=torch.long
        )

        assert src.max() < node_features.shape[0]
        assert dst.max() < node_features.shape[0]
        assert src.min() >= 0
        assert src.min() >= 0

        dict_data["x_n_list"].append(node_features.squeeze())
        dict_data["src_list"].append(src)
        dict_data["dst_list"].append(dst)
        dict_data["y_list"].append(None)

        # dict_data["src_list"].append(edge_index[0])
        # dict_data["dst_list"].append(edge_index[1])
        # # Processing labels
        # label = None
        # dict_data["y_list"].append(label)
    return dict_data


def to_dag_dataset(data_dict, num_categories):
    if data_dict["y_list"][0] is not None:
        label = True
    else:
        label = False
    dataset = DAGDataset(
        num_categories=num_categories, label=False
    )  # Doing unconditional

    src_list = data_dict["src_list"]
    dst_list = data_dict["dst_list"]
    x_n_list = data_dict["x_n_list"]
    y_list = data_dict["y_list"]

    num_g = len(src_list)
    for i in range(num_g):
        dataset.add_data(src_list[i], dst_list[i], x_n_list[i], None)  # , y_list[i]
    # We leave y behind to have equal conditions for all models

    return dataset


def get_tpu_tile():
    root_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(root_path, f"../../../data/tpu_tile/raw")

    train_path = os.path.join(root_path, "train.pt")
    val_path = os.path.join(root_path, "val.pt")
    test_path = os.path.join(root_path, "test.pt")

    print(f"Loading TPU Tile dataset...")
    # Load the pre-processed TPU Tile dataset, where for each kernel graph, we
    # average the normalized runtime over multiple compiler configurations.
    train_set = torch.load(train_path)
    val_set = torch.load(val_path)
    test_set = torch.load(test_path)

    num_categories = torch.cat(train_set["x_n_list"]).max().item() + 1
    train_set = to_dag_dataset(train_set, num_categories)
    val_set = to_dag_dataset(val_set, num_categories)
    test_set = to_dag_dataset(test_set, num_categories)

    return train_set, val_set, test_set


def get_synthetic(dataset_name):
    root_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(
        root_path, f"../../../../data/synthetic_{dataset_name}/processed"
    )

    train_path = os.path.join(root_path, "train.pt")
    val_path = os.path.join(root_path, "val.pt")
    test_path = os.path.join(root_path, "test.pt")

    print(f"Loading {dataset_name} dataset...")

    train_set = torch.load(train_path)
    val_set = torch.load(val_path)
    test_set = torch.load(test_path)

    # Adapting to the required format

    num_categories = train_set[0].x.shape[-1]
    train_set = to_dag_dataset(reformat_data(train_set), num_categories)
    val_set = to_dag_dataset(reformat_data(val_set), num_categories)
    test_set = to_dag_dataset(reformat_data(test_set), num_categories)

    return train_set, val_set, test_set
