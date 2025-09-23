import torch
import re
import os
import pathlib
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Trainer

import hydra
from omegaconf import DictConfig

from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from src.diffusion.extra_features_directed import ExtraDirectedFeatures

from src.datasets.dataset_utils import load_pickle

def parse_graphs(file_path):
    """
    Parses a text file containing graph information and converts it into a list of tensors.

    Args:
        file_path (str): The path to the input text file.

    Returns:
        list: A list of lists, where each inner list contains two tensors:
              - A tensor of node labels.
              - A tensor representing the adjacency matrix.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    graphs_data = []
    # Split the content by 'N=' to process each graph individually
    graph_sections = content.strip().split('N=')[1:]

    for section in graph_sections:
        lines = section.strip().split('\n')
       
        # Extract N
        n = int(lines[0])
       
        # Find the line where X starts
        x_start_index = -1
        for i, line in enumerate(lines):
            if line.strip() == 'X:':
                x_start_index = i + 1
                break
       
        # Extract X
        x_lines = []
        x_line_index = x_start_index
        while "E:" not in lines[x_line_index]:
            x_lines.append(lines[x_line_index].strip())
            x_line_index += 1
        x_str = " ".join(x_lines)
        x_values = [int(val) for val in x_str.split()]
        x_tensor = torch.tensor(x_values)

        # Find the line where E starts
        e_start_index = -1
        for i, line in enumerate(lines):
            if line.strip() == 'E:':
                e_start_index = i + 1
                break

        # Extract E
        e_lines = lines[e_start_index:]
        e_str = " ".join([line.strip() for line in e_lines])
        e_values = [int(val) for val in re.split(r'\s+', e_str) if val]
        e_tensor = torch.tensor(e_values).reshape(n, n)
       
        graphs_data.append([x_tensor, e_tensor])
       
    return graphs_data

def evaluate_samples(
    sampling_metrics,
    dataset_infos,
    samples,
    labels,
    is_test,
    cfg,
    save_filename="",
):
    print("Computing sampling metrics...")

    to_log = {}
    samples_to_evaluate = cfg.general.final_model_samples_to_generate
    for i in range(cfg.general.num_sample_fold):
        cur_samples = samples[
            i * samples_to_evaluate : (i + 1) * samples_to_evaluate
        ]
        cur_labels = labels[
            i * samples_to_evaluate : (i + 1) * samples_to_evaluate
        ]

        cur_to_log = sampling_metrics.forward(
            cur_samples,
            ref_metrics=dataset_infos.ref_metrics,
            name=f"self.name_{i}",
            current_epoch=0,
            val_counter=-1,
            test=is_test,
            local_rank=0,
        )

        if i == 0:
            to_log = {i: [cur_to_log[i]] for i in cur_to_log}
        else:
            to_log = {i: to_log[i] + [cur_to_log[i]] for i in cur_to_log}

        filename = os.path.join(
            os.getcwd(),
            f"epoch0_res_fold{i}_{save_filename}.txt",
        )
        with open(filename, "w") as file:
            for key, value in cur_to_log.items():
                file.write(f"{key}: {value}\n")

    to_log = {
        i: (np.array(to_log[i]).mean(), np.array(to_log[i]).std())
        for i in to_log
    }

    return to_log

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    pl.seed_everything(cfg.train.seed)

    dataset_config = cfg["dataset"]

    if dataset_config["name"] in ["tpu_tile"]:
        from datasets.tpu_tile_dataset import TPUGraphDataModule, TPUDatasetInfos
        from src.analysis.directed_utils import TPUSamplingMetrics
        from analysis.visualization import DAGVisualization

        datamodule = TPUGraphDataModule(cfg)
        sampling_metrics = TPUSamplingMetrics(datamodule)
        dataset_infos = TPUDatasetInfos(datamodule, dataset_config)
        train_metrics = (
            TrainAbstractMetricsDiscrete()
        )
        visualization_tools = DAGVisualization(cfg)

        if cfg.model.extra_features is not None:
            extra_features = ExtraDirectedFeatures(
                cfg.model.extra_features,
                cfg.model.rrwp_steps,
                cfg.model.restart_prob,
                dataset_info=dataset_infos,
            )
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

    elif dataset_config["name"] in ["visual_genome"]:
        from datasets.visual_genome_dataset import (
            VisualGenomeDataModule,
            VisualGenomeDatasetInfos,
        )
        from src.analysis.directed_utils import VisualGenomeSamplingMetrics
        from analysis.visualization import SceneGraphVisualization

        datamodule = VisualGenomeDataModule(cfg)
        sampling_metrics = VisualGenomeSamplingMetrics(datamodule)
        dataset_infos = VisualGenomeDatasetInfos(datamodule, dataset_config)
        train_metrics = (
            TrainAbstractMetricsDiscrete()
        )
        visualization_tools = SceneGraphVisualization(cfg)

        if cfg.model.extra_features is not None:
            extra_features = ExtraDirectedFeatures(
                cfg.model.extra_features,
                cfg.model.rrwp_steps,
                cfg.model.restart_prob,
                dataset_info=dataset_infos,
            )
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

    # Parse the graphs from the file
    parsed_graphs = parse_graphs(cfg.general.sampled_graphs)

    # Create a list of labels that is tensor([], dtype=torch.int64) repeated as many times as there are graphs
    labels = [torch.tensor([], dtype=torch.int64) for _ in range(len(parsed_graphs))]

    ## Adding reference metrics to dataset_infos
    # defining dummy arguments and loading reference metrics
    ref_metrics_path = datamodule.train_dataloader().dataset.root

    # Check if reference metrics file exists and load it
    if not os.path.exists(os.path.join(ref_metrics_path, "ref_metrics.pkl")):
        # Computing training reference metrics
        dataset_infos.compute_reference_metrics(
            datamodule=datamodule,
            sampling_metrics=sampling_metrics,
        )
    train_metrics = load_pickle(os.path.join(ref_metrics_path, f"ref_metrics.pkl"))
    dataset_infos.ref_metrics = train_metrics

    to_log = evaluate_samples(
        sampling_metrics=sampling_metrics,
        dataset_infos=dataset_infos,
        samples=parsed_graphs,
        labels=labels, 
        save_filename="",
        cfg=cfg,
        is_test=True,
    )

    # Store results
    filename = os.path.join(
        os.getcwd(),
        f"test_epoch{0}_res_{cfg.sample.eta}_{cfg.sample.rdb}.txt",
    )
    with open(filename, "w") as file:
        for key, value in to_log.items():
            file.write(f"{key}: {value}\n")

    print("Finished testing.")

if __name__ == '__main__':
    main()