"""Adapted from https://link.springer.com/chapter/10.1007/978-3-031-55088-1_4"""

import graph_tool.all as gt  # need to import first to avoid error

import os
import torch
from collections import Counter, defaultdict
from itertools import product, combinations
import time

import hydra
import logging
import numpy as np
import pickle
from tqdm import tqdm
import torch_geometric

import src.utils as utils
from src.datasets.synthetic_dataset import SyntheticDataModule, SyntheticDatasetInfos
from src.datasets.tpu_tile_dataset import TPUGraphDataModule, TPUDatasetInfos
from src.datasets.visual_genome_dataset import  VisualGenomeDataModule, VisualGenomeDatasetInfos
from src.analysis.directed_utils import SyntheticSamplingMetrics, TPUSamplingMetrics, VisualGenomeSamplingMetrics
from src.datasets.dataset_utils import load_pickle, save_pickle

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="mle")
def main(cfg):
    start_time = time.time()

    if cfg["dataset"].name == "synthetic":
        datamodule = SyntheticDataModule(cfg)
        dataset_infos = SyntheticDatasetInfos(datamodule, cfg["dataset"])
        metrics_dictionary = {
            "er": [
                "in_degree",
                "out_degree",
                "clustering",
                "spectre",
                "wavelet",
                "er",
                "dag",
                "valid",
                "unique",
            ],
            "ba": [
                "in_degree",
                "out_degree",
                "clustering",
                "spectre",
                "wavelet",
                "ba",
                "dag",
                "valid",
                "unique",
            ],
            "sbm": [
                "in_degree",
                "out_degree",
                "clustering",
                "spectre",
                "wavelet",
                "sbm",
                "valid",
                "unique",
            ],
            "planar": [
                "in_degree",
                "out_degree",
                "clustering",
                "spectre",
                "wavelet",
                "planar",
                "valid",
                "unique",
            ],
        }
        sampling_metrics = SyntheticSamplingMetrics(
            datamodule,
            acyclic=cfg.dataset.acyclic,
            metrics_list=metrics_dictionary[cfg.dataset.graph_type],
            graph_type=cfg.dataset.graph_type,
        )
    elif cfg["dataset"].name == "tpu_tile":
        datamodule = TPUGraphDataModule(cfg)
        dataset_infos = TPUDatasetInfos(datamodule, cfg["dataset"])
        sampling_metrics = TPUSamplingMetrics(datamodule)
    
    elif cfg["dataset"].name in ["visual_genome"]:
        datamodule = VisualGenomeDataModule(cfg)
        sampling_metrics = VisualGenomeSamplingMetrics(datamodule)
        dataset_infos = VisualGenomeDatasetInfos(datamodule, cfg["dataset"])

    training_graphs = []
    logging.info(
        "===Converting training dataset to format required by sampling metrics.==="
    )
    for data_batch in tqdm(datamodule.train_dataloader()):
        dense_data, node_mask = utils.to_dense(
            data_batch.x,
            data_batch.edge_index,
            data_batch.edge_attr,
            data_batch.batch,
        )
        dense_data = dense_data.mask(
            node_mask, collapse=True, directed=cfg["dataset"].directed
        ).split(node_mask)
        for graph in dense_data:
            training_graphs.append([graph.X, graph.E])

    # Get distribution of number of nodes in each graph in the dataset
    logging.info("=== Computing probabilities of number of nodes ===")
    num_nodes_tensor = dataset_infos.n_nodes
    num_nodes_distribution = {
        i: prob.item() for i, prob in enumerate(num_nodes_tensor) if prob > 0
    }
    # num_nodes_counter = Counter([graph[0].shape[-1] for graph in training_graphs])
    # num_nodes_distribution = {
    #     num_nodes: counts / num_graphs
    #     for num_nodes, counts in num_nodes_counter.items()
    # }

    # Get distribution of nodes types
    logging.info("=== Computing probabilities of node classes ===")
    node_class_tensor = dataset_infos.node_types
    node_class_distribution = {
        i: float(prob) for i, prob in enumerate(node_class_tensor) if prob > 0
    }
    # node_class_counter = Counter()
    # for nodes, _ in tqdm(training_graphs):
    #     node_class_counter.update(nodes.tolist())
    # total_num_nodes = sum(node_class_counter.values())
    # node_class_distribution = {
    #     node_class: counts / total_num_nodes
    #     for node_class, counts in node_class_counter.items()
    # }

    # Get distribution of number of edges in each graph in the dataset
    # logging.info("=== Computing probabilities of edge classes ===")
    # edge_class_tensor = dataset_infos.edge_types
    # edge_class_distribution = {i: float(prob) for i, prob in enumerate(edge_class_tensor) if prob > 0}

    # Get distribution for edge pairs
    logging.info("=== Computing probabilities of edge pairs ===")
    num_edge_classes = len(dataset_infos.edge_types)

    edge_pair_counts = defaultdict(
        lambda: torch.zeros(num_edge_classes, dtype=torch.float32)
    )

    # Iterate over all training graphs
    for nodes, edges in tqdm(training_graphs):
        num_node_classes = len(node_class_tensor)

        for class_a, class_b in product(range(num_node_classes), repeat=2):
            class_a_nodes = torch.where(nodes == class_a)[0]
            class_b_nodes = torch.where(nodes == class_b)[0]
            # if class_a_nodes.nelement() == 0 or class_b_nodes.nelement() == 0:
            #     continue
            submatrix = edges[class_a_nodes[:, None], class_b_nodes]
            for edge_type in range(num_edge_classes):
                edge_pair_counts[(class_a, class_b)][edge_type] += submatrix.eq(
                    edge_type
                ).sum()

    edge_pair_distribution = {
        pair: (
            counts / counts.sum()
            if counts.sum() > 0
            else torch.ones(num_edge_classes) / num_edge_classes
        )
        for pair, counts in edge_pair_counts.items()
    }

    # SANITY CHECKS: num nodes and node phenotypes probs sum to one and edges prob between 0 and 1
    # print("num nodes probs sum", sum(num_nodes_distribution.values()))
    # print("node phenotypes probs sum", sum(node_phenotypes_distribution.values()))
    # print(
    #     "edge probs all between 0 and 1: ",
    #     all(0 <= prob <= 1 for prob in phenotype_pair_edge_probs.values()),
    # )

    probs_dict = {
        "num_nodes": num_nodes_distribution,
        "nodes": node_class_distribution,
        "edges": edge_pair_distribution,
    }
    ref_metrics_path = os.path.join(
        datamodule.train_dataloader().dataset.root, f"ref_metrics_naive.pkl"
    )
    save_pickle(probs_dict, ref_metrics_path)

    logging.info(f"=== Total training time: {time.time() - start_time} seconds ===")

    logging.info(f"=== Sampling graphs using naive model ===")
    start_time = time.time()

    # Seed the random number generator
    np.random.seed(125)

    # Sample number of nodes for each graph
    num_nodes_sorted = sorted(num_nodes_distribution)
    num_nodes_probs = [
        num_nodes_distribution[num_nodes] for num_nodes in num_nodes_sorted
    ]
    num_nodes_sampled = np.random.choice(
        num_nodes_sorted, size=cfg.dataset_size, p=num_nodes_probs
    )
    # Build each graph
    graph_list = []
    for graph_idx in tqdm(range(cfg.dataset_size)):
        # Sample node classes for nodes of each graph
        num_nodes = num_nodes_sampled[graph_idx]
        probs = torch.tensor(
            list(node_class_distribution.values()), dtype=torch.float32
        )
        node_class = torch.multinomial(probs, num_nodes, replacement=True)

        # Sample edges (with type) for each graph using the edge pair distribution
        edge_class = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
        for node_1_idx, node_2_idx in product(range(num_nodes), repeat=2):
            edge_vertices_class = tuple(
                sorted((node_class[node_1_idx].item(), node_class[node_2_idx].item()))
            )
            edge_probability = edge_pair_distribution[edge_vertices_class]
            edge_class[node_1_idx, node_2_idx] = torch.multinomial(
                edge_probability, 1
            ).item()

        # Build graph
        graph = [node_class, edge_class]
        graph_list.append(graph)

    logging.info(f"=== Total sampling time: {time.time() - start_time} seconds ===")

    logging.info(f"=== Calculating metrics ===")
    # metrics_list=['in_degree', 'out_degree', 'clustering', 'spectre', 'wavelet', 'connected', 'dag', 'valid', 'unique']]

    # defining dummy arguments
    dummy_kwargs = {
        "name": "ref_metrics",
        "current_epoch": 0,
        "val_counter": 0,
        "local_rank": 0,
        "ref_metrics": {"val": None, "test": None},
    }
    metrics_results = sampling_metrics.forward(graph_list, **dummy_kwargs, test=True)

    # Save metrics
    save_pickle(metrics_results, ref_metrics_path)


if __name__ == "__main__":
    main()
