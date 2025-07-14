import sys
import os
import torch

from pprint import pprint
from tqdm import tqdm

from setup_utils import set_seed
from dataset import load_dataset, DAGDataset

# from eval import TPUTileEvaluator
from model.diffusion import DiscreteDiffusion, EdgeDiscreteDiffusion
from model.layer_dag import LayerDAG

import hydra
import logging
import omegaconf
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base='1.3', config_path='../../../configs', config_name='LayerDAG')
def main(config: DictConfig):
    torch.set_num_threads(config.general.num_threads)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    root_path = os.path.dirname(os.path.abspath(__file__)) + "/outputs"
    load_path = os.path.join(root_path, config.sample.model_path)

    ckpt = torch.load(load_path)

    node_diffusion = DiscreteDiffusion(**ckpt["node_diffusion_config"])
    edge_diffusion = EdgeDiscreteDiffusion(**ckpt["edge_diffusion_config"])

    model = LayerDAG(
        device=device,
        node_diffusion=node_diffusion,
        edge_diffusion=edge_diffusion,
        **ckpt["model_config"]
    )
    pprint(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    set_seed(config.general.seed)

    # Sample graphs
    train_set, val_set, _ = load_dataset('tpu_tile')

    graph_list = sample_tpu_subset(config, device, train_set.dummy_category, model, train_set)

    # Metrics loop
    samples = [graph_list[i : i + config.sample.sample_size] for i in range(0, len(graph_list), config.sample.sample_size)]

    # Save the sampled graphs as pickle files
    with open(os.path.join(root_path, 'sampled_graphs_TPU.pkl'), 'wb') as f:
        torch.save(samples, f)


def sample_tpu_subset(config, device, dummy_category, model, subset):

    num_samples = config.sample.sample_size * config.sample.num_folds
    batch_edge_index, batch_x_n = model.sample(
            device, num_samples, None,
            min_num_steps_n=config.sample.min_num_steps_n,
            max_num_steps_n=config.sample.max_num_steps_n,
            min_num_steps_e=config.sample.min_num_steps_e,
            max_num_steps_e=config.sample.max_num_steps_e)

    graph_list = []
    for i in tqdm(range(num_samples), desc='Sampling graphs'):
        nodes = batch_x_n[i].view(-1).cpu()
        num_nodes = nodes.shape[0]

        edges = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        edge_index = batch_edge_index[i].cpu()
        edges[edge_index[0], edge_index[1]] = 1

        graph_list += [[nodes, edges]]
    
    return graph_list

# def dump_to_file(syn_set, file_name, sample_dir):
#     file_path = os.path.join(sample_dir, file_name)
#     data_dict = {
#         'src_list': [],
#         'dst_list': [],
#         'x_n_list': [],
#         'y_list': []
#     }
#     for i in range(len(syn_set)):
#         src_i, dst_i, x_n_i, y_i = syn_set[i]

#         data_dict['src_list'].append(src_i)
#         data_dict['dst_list'].append(dst_i)
#         data_dict['x_n_list'].append(x_n_i)
#         data_dict['y_list'].append(y_i)

#     torch.save(data_dict, file_path)

# def eval_tpu_tile(args, device, model):
#     sample_dir = 'tpu_tile_samples'
#     os.makedirs(sample_dir, exist_ok=True)

#     evaluator = TPUTileEvaluator()
#     train_set, val_set, _ = load_dataset('tpu_tile')

#     train_syn_set = sample_tpu_subset(args, device, train_set.dummy_category, model, train_set)
#     val_syn_set = sample_tpu_subset(args, device, train_set.dummy_category, model, val_set)

#     evaluator.eval(train_syn_set, val_syn_set)

#     dump_to_file(train_syn_set, 'train.pth', sample_dir)
#     dump_to_file(val_syn_set, 'val.pth', sample_dir)

if __name__ == "__main__":
    # from argparse import ArgumentParser

    # parser = ArgumentParser()
    # parser.add_argument("--model_path", type=str, help="Path to the model.")
    # parser.add_argument("--batch_size", type=int, default=256)
    # parser.add_argument("--num_threads", type=int, default=24)
    # parser.add_argument("--min_num_steps_n", type=int, default=None)
    # parser.add_argument("--min_num_steps_e", type=int, default=None)
    # parser.add_argument("--max_num_steps_n", type=int, default=None)
    # parser.add_argument("--max_num_steps_e", type=int, default=None)
    # parser.add_argument("--seed", type=int, default=0)
    # args = parser.parse_args()

    main()
