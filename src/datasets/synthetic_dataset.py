import os
import pathlib
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

import networkx as nx
from scipy.spatial import Delaunay


class SyntheticDataset(InMemoryDataset):
    """
    Class for generating and processing synthetic graph data.
    """

    def __init__(
        self,
        split,
        root,
        num_nodes,
        degree,
        p_threshold,
        graph_type,
        num_graphs=1000,
        w_min=0.5,
        w_max=0.5,
        acyclic=True,
        transform=None,
        pre_transform=None,
    ):
        self.split = split
        self.num_nodes = num_nodes
        self.degree = degree
        self.p_threshold = p_threshold
        self.graph_type = graph_type
        self.num_graphs = num_graphs
        self.w_min = w_min
        self.w_max = w_max
        self.acyclic = acyclic
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # No raw files needed

    @property
    def processed_file_names(self):
        return [f"{self.split}.pt"]

    def process(self):
        """
        Generate synthetic data and convert to PyTorch Geometric format.
        """
        data_list = []

        for _ in range(self.num_graphs):
            adj = self.generate_structure(
                self.num_nodes,
                self.degree,
                self.p_threshold,
                self.graph_type,
                self.w_min,
                self.w_max,
                self.acyclic,
            )
            edge_index = torch.tensor(np.vstack(np.nonzero(adj)), dtype=torch.long)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)  # One-hot encoded node features
            y = torch.zeros([1, 0]).float()  # No label information

            data = torch_geometric.data.Data(
                x=X, edge_index=edge_index, edge_attr=edge_attr, y=y
            )
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    @staticmethod
    def generate_structure(
        num_nodes, degree, p_threshold, graph_type, w_min, w_max, acyclic
    ):
        """
        Generate a synthetic graph structure as an adjacency matrix.

        Parameters:
        - num_nodes: Number of nodes in the graph.
        - degree: Average degree of nodes.
        - graph_type: Type of graph to generate ('erdos-renyi', 'barabasi-albert', 'full').
        - w_min, w_max: Minimum and maximum absolute edge weights.
        - acyclic: If True, ensures the generated graph is acyclic (DAG).

        Returns:
        - adj_matrix: The adjacency matrix of the generated graph.
        """
        if num_nodes < 2:
            raise ValueError("Graph must have at least 2 nodes")

        w_min, w_max = abs(w_min), abs(w_max)
        if w_min > w_max:
            raise ValueError(
                f"Minimum weight must be <= maximum weight: {w_min} > {w_max}"
            )

        if graph_type == "er":  # erdos-renyi
            # p_threshold = float(degree) / (num_nodes - 1)
            num_nodes = np.random.randint(20, 80)
            p_edge = (np.random.rand(num_nodes, num_nodes) < p_threshold).astype(float)
            # Set diagonal to zero
            np.fill_diagonal(p_edge, 0)
            edge_flags = (
                np.tril(p_edge, k=-1) if acyclic else p_edge
            )  # Force DAG if acyclic=True

        elif graph_type == "ba":  # barabasi-albert
            # Known as Price's model in network theory when acyclic
            # Dynamically setting m to allow graphs with different number of nodes
            if not acyclic:
                raise ValueError("Barabasi-Albert model only generates acyclic graphs")

            m = max(1, int(round(np.log2(num_nodes))))  # m = 6 for 64 nodes
            # m = int(round(degree / 2))
            edge_flags = np.zeros([num_nodes, num_nodes])
            bag = [0]
            for i in range(1, num_nodes):
                dest = np.random.choice(bag, size=m)
                for j in dest:
                    edge_flags[i, j] = 1
                bag.append(i)
                bag.extend(dest)

        elif graph_type == "sbm":
            # Random number of communities (between 2 and 5) and community sizes
            n_communities = np.random.randint(2, 6)
            community_sizes = np.random.random_integers(20, 40, size=n_communities)

            # intra- and inter-community probabilities
            probs = (
                np.ones((n_communities, n_communities)) * 0.05
            )  # Low inter-community probability
            np.fill_diagonal(probs, 0.3)  # Higher intra-community probability

            G = nx.stochastic_block_model(community_sizes, probs, directed=True)
            adj_matrix = nx.to_numpy_array(G)

            if acyclic:
                raise ValueError("SBM model only generates cyclic graphs")
                # Enforce DAG by keeping only forward edges
                # adj_matrix = np.triu(adj_matrix, k=1)  # Keep only upper-triangular part (no backward edges)
            return adj_matrix

        elif graph_type == "planar":
            points = np.random.rand(num_nodes, 2)
            tri = Delaunay(points)  # Delaunay triangulation

            adj_matrix = np.zeros([num_nodes, num_nodes])
            for t in tri.simplices:
                edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]  # Triangle edges
                if not acyclic:
                    for u, v in edges:
                        if torch.rand(1).item() < 0.5:  # 50% directionality chance
                            adj_matrix[u, v] = 1  # Add edge u -> v
                        else:
                            adj_matrix[v, u] = 1  # Add edge v -> u

                else:
                    raise ValueError("Planar model only generates cyclic graphs")
                    # DAG: Only allow edges from lower-indexed to higher-indexed nodes
                    # for u, v in edges:
                    #     if u < v:
                    #         adj_matrix[u, v] = 1
                    #     else:
                    #         adj_matrix[v, u] = 1
            return adj_matrix

        elif graph_type == "full":
            edge_flags = (
                np.tril(np.ones([num_nodes, num_nodes]), k=-1)
                if acyclic
                else np.ones([num_nodes, num_nodes])
            )

        else:
            raise ValueError(f"Unknown graph type {graph_type}")

        # Randomly permute edges
        perms = np.random.permutation(np.eye(num_nodes, num_nodes))
        edge_flags = perms.T @ edge_flags @ perms

        # Generate random edge weights (optional)
        # edge_weights = np.random.uniform(low=w_min, high=w_max, size=[num_nodes, num_nodes])
        # edge_weights[np.random.rand(num_nodes, num_nodes) < 0.5] *= -1

        adj_matrix = edge_flags  # Weighting removed for simplicity; can be added back
        return adj_matrix


class SyntheticDataModule(AbstractDataModule):
    """
    Data module for synthetic DAG datasets compatible with PyTorch Geometric.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_nodes = cfg.dataset.num_nodes
        self.degree = cfg.dataset.degree
        self.p_threshold = cfg.dataset.p_threshold
        self.graph_type = cfg.dataset.graph_type
        self.w_min = cfg.dataset.w_min
        self.w_max = cfg.dataset.w_max
        self.acyclic = cfg.dataset.acyclic
        self.num_graphs_train = cfg.dataset.num_graphs_train
        self.num_graphs_val = cfg.dataset.num_graphs_val
        self.num_graphs_test = cfg.dataset.num_graphs_test

        if self.acyclic:
            self.datadir = cfg.dataset.datadir + "_" + cfg.dataset.graph_type + "_dag/"
        else:
            self.datadir = cfg.dataset.datadir + "_" + cfg.dataset.graph_type + "/"
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {
            "train": SyntheticDataset(
                num_nodes=self.num_nodes,
                degree=self.degree,
                p_threshold=self.p_threshold,
                graph_type=self.graph_type,
                num_graphs=self.num_graphs_train,
                split="train",
                root=root_path,
                w_min=self.w_min,
                w_max=self.w_max,
                acyclic=self.acyclic,
            ),
            "val": SyntheticDataset(
                num_nodes=self.num_nodes,
                degree=self.degree,
                p_threshold=self.p_threshold,
                graph_type=self.graph_type,
                num_graphs=self.num_graphs_val,
                split="val",
                root=root_path,
                w_min=self.w_min,
                w_max=self.w_max,
                acyclic=self.acyclic,
            ),
            "test": SyntheticDataset(
                num_nodes=self.num_nodes,
                degree=self.degree,
                p_threshold=self.p_threshold,
                graph_type=self.graph_type,
                num_graphs=self.num_graphs_test,
                split="test",
                root=root_path,
                w_min=self.w_min,
                w_max=self.w_max,
                acyclic=self.acyclic,
            ),
        }

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class SyntheticDatasetInfos(AbstractDatasetInfos):
    """
    Metadata and information about the Synthetic DAG Dataset.
    """

    def __init__(self, datamodule, dataset_config):
        self.name = "synthetic_graphs"
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
