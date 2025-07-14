from __future__ import print_function
import os
import sys
import math
import pickle
import pdb
import argparse
import random
from tqdm import tqdm
from shutil import copy
import graph_tool.all as gt
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy.io
from scipy.linalg import qr
import igraph
from random import shuffle
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from util import *
from models import *

# from bayesian_optimization.evaluate_BN import Eval_BN

import wandb
import hydra
import logging
import omegaconf
from omegaconf import DictConfig, OmegaConf

import src.utils as utils
from src.datasets.synthetic_dataset import SyntheticDataModule, SyntheticDatasetInfos
from src.datasets.tpu_tile_dataset import TPUGraphDataModule, TPUDatasetInfos
from src.analysis.directed_utils import SyntheticSamplingMetrics, TPUSamplingMetrics
from src.datasets.dataset_utils import load_pickle, save_pickle

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="DVAE")
def main(cfg: DictConfig):

    # Allow to dynamically add keys
    OmegaConf.set_struct(cfg, False)

    # Start wandb
    setup_wandb(cfg)

    cfg.optimization.cuda = not cfg.optimization.no_cuda and torch.cuda.is_available()
    torch.manual_seed(cfg.optimization.seed)
    if cfg.optimization.cuda:
        torch.cuda.manual_seed(cfg.optimization.seed)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    np.random.seed(cfg.optimization.seed)
    random.seed(cfg.optimization.seed)

    """Prepare data"""
    if cfg.data.name in ["ba_dag", "er_dag"]:
        pdb.set_trace()
        file_root = os.path.abspath(os.path.join(__file__, "../../../../data"))
        data_dir = file_root + "synthetic_" + cfg.data.name + "/processed"
        train_path = os.path.join(data_dir, "train.pt")
        train_data, graph_args = load_pyg_to_igraph(
            train_path, cfg.data.name, cfg.data.nvt
        )

    elif cfg.data.name in ["tpu_tile"]:
        file_root = os.path.abspath(os.path.join(__file__, "../../../../data"))
        data_dir = file_root + cfg.data.name + "/processed"
        train_path = os.path.join(data_dir, "train.pt")
        train_data, graph_args = load_pyg_to_igraph(
            train_path, cfg.data.name, cfg.data.nvt
        )

    cfg.file_dir = os.path.dirname(os.path.realpath(__file__))
    cfg.data.res_dir = os.path.join(
        cfg.file_dir, "results/{}{}".format(cfg.data.name, cfg.data.save_appendix)
    )
    if not os.path.exists(cfg.data.res_dir):
        os.makedirs(cfg.data.res_dir)

    # pkl_name = os.path.join(cfg.data.res_dir, cfg.data.name + '.pkl')

    # # check whether to load pre-stored pickle data
    # if os.path.isfile(pkl_name) and not cfg.data.reprocess:
    #     with open(pkl_name, 'rb') as f:
    #         train_data, test_data, graph_args = pickle.load(f)
    # # otherwise process the raw data and save to .pkl
    # else:
    if cfg.data.name == "final_structures6":
        # determine data formats according to models, DVAE: igraph, SVAE: string (as tensors)
        if cfg.model.name == "DVAE":
            input_fmt = "igraph"
        elif cfg.model.name == "SVAE":
            input_fmt = "string"
        if cfg.data.type == "ENAS":
            train_data, test_data, graph_args = load_ENAS_graphs(
                cfg.data.name, n_types=cfg.data.nvt, fmt=input_fmt
            )
        elif cfg.data.type == "BN":
            train_data, test_data, graph_args = load_BN_graphs(
                cfg.data.name, n_types=cfg.data.nvt, fmt=input_fmt
            )
        # with open(pkl_name, 'wb') as f:
        #     pickle.dump((train_data, test_data, graph_args), f)

    if cfg.data.small_train:
        train_data = train_data[:128]

    """Prepare the model"""
    # model
    model = eval(cfg.model.name)(
        graph_args.max_n,
        graph_args.num_vertex_type,
        graph_args.START_TYPE,
        graph_args.END_TYPE,
        hs=cfg.model.hs,
        nz=cfg.model.nz,
        bidirectional=cfg.model.bidirectional,
    )
    if cfg.model.predictor:
        predictor = nn.Sequential(
            nn.Linear(cfg.model.nz, cfg.model.hs), nn.Tanh(), nn.Linear(cfg.model.hs, 1)
        )
        model.predictor = predictor
        model.mseloss = nn.MSELoss(reduction="sum")
    # optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.optimization.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=10, verbose=True
    )

    model.to(device)

    if cfg.optimization.all_gpus:
        net = custom_DataParallel(model, device_ids=range(torch.cuda.device_count()))
    if cfg.model.load_latest_model:
        load_module_state(model, os.path.join(cfg.data.res_dir, "latest_model.pth"))
    else:
        if cfg.model.continue_from is not None:
            epoch = cfg.model.continue_from
            load_module_state(
                model,
                os.path.join(cfg.data.res_dir, f"model_checkpoint{epoch}_{cfg.optimization.batch_size}.pth"
                ),
            )
            load_module_state(
                optimizer,
                os.path.join(
                    cfg.data.res_dir, f"optimizer_checkpoint{epoch}_{cfg.optimization.batch_size}.pth",
                ),
            )
            load_module_state(
                scheduler,
                os.path.join(
                    cfg.data.res_dir, f"scheduler_checkpoint{epoch}_{cfg.optimization.batch_size}.pth"
                ),
            )

    # plot sample train/test graphs
    if (
        not os.path.exists(os.path.join(cfg.data.res_dir, "train_graph_id0.pdf"))
        or cfg.data.reprocess
    ):
        if not cfg.data.keep_old:
            for data in ["train_data", "test_data"]:
                G = [g for g, y in eval(data)[:10]]
                if cfg.model.name == "SVAE":
                    G = [g.to(device) for g in G]
                    G = model._collate_fn(G)
                    G = model.construct_igraph(
                        G[:, :, : model.nvt], G[:, :, model.nvt :], False
                    )
                for i, g in enumerate(G):
                    name = "{}_graph_id{}".format(data[:-5], i)
                    plot_DAG(g, cfg.data.res_dir, name, data_type=cfg.data.type)

    """Define some train/test functions"""

    def train(epoch):
        model.train()
        train_loss = 0
        recon_loss = 0
        kld_loss = 0
        pred_loss = 0
        shuffle(train_data)
        pbar = tqdm(train_data)
        g_batch = []
        y_batch = []
        for i, (g, y) in enumerate(pbar):
            if cfg.model.name == "SVAE":  # for SVAE, g is tensor
                g = g.to(device)
            g_batch.append(g)
            y_batch.append(y)
            if len(g_batch) == cfg.optimization.batch_size or i == len(train_data) - 1:
                optimizer.zero_grad()
                g_batch = model._collate_fn(g_batch)
                if cfg.optimization.all_gpus:  # does not support predictor yet
                    loss = net(g_batch).sum()
                    pbar.set_description(
                        "Epoch: %d, loss: %0.4f" % (epoch, loss.item() / len(g_batch))
                    )
                    recon, kld = 0, 0
                else:
                    mu, logvar = model.encode(g_batch)
                    loss, recon, kld = model.loss(mu, logvar, g_batch)
                    if cfg.model.predictor:
                        y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
                        y_pred = model.predictor(mu)
                        pred = model.mseloss(y_pred, y_batch)
                        loss += pred
                        pbar.set_description(
                            "Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f, pred: %0.4f"
                            % (
                                epoch,
                                loss.item() / len(g_batch),
                                recon.item() / len(g_batch),
                                kld.item() / len(g_batch),
                                pred / len(g_batch),
                            )
                        )
                    else:
                        pbar.set_description(
                            "Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f"
                            % (
                                epoch,
                                loss.item() / len(g_batch),
                                recon.item() / len(g_batch),
                                kld.item() / len(g_batch),
                            )
                        )
                loss.backward()

                train_loss += float(loss)
                recon_loss += float(recon)
                kld_loss += float(kld)
                if cfg.model.predictor:
                    pred_loss += float(pred)
                optimizer.step()
                g_batch = []
                y_batch = []

        print(
            "====> Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss / len(train_data)
            )
        )

        if cfg.model.predictor:
            return train_loss, recon_loss, kld_loss, pred_loss
        return train_loss, recon_loss, kld_loss

    def extract_latent(data):
        model.eval()
        Z = []
        Y = []
        g_batch = []
        for i, (g, y) in enumerate(tqdm(data)):
            if cfg.model.name == "SVAE":
                g_ = g.to(device)
            elif cfg.model.name == "DVAE":
                # copy igraph
                # otherwise original igraphs will save the H states and consume more GPU memory
                g_ = g.copy()
            g_batch.append(g_)
            if len(g_batch) == cfg.optimization.infer_batch_size or i == len(data) - 1:
                g_batch = model._collate_fn(g_batch)
                mu, _ = model.encode(g_batch)
                mu = mu.cpu().detach().numpy()
                Z.append(mu)
                g_batch = []
            Y.append(y)
        return np.concatenate(Z, 0), np.array(Y)

    """Extract latent representations Z"""

    def save_latent_representations(epoch):
        Z_train, Y_train = extract_latent(train_data)
        # Z_test, Y_test = extract_latent(test_data)
        latent_pkl_name = os.path.join(
            cfg.data.res_dir, cfg.data.name + "_latent_epoch{}.pkl".format(epoch)
        )
        latent_mat_name = os.path.join(
            cfg.data.res_dir, cfg.data.name + "_latent_epoch{}.mat".format(epoch)
        )
        with open(latent_pkl_name, "wb") as f:
            pickle.dump((Z_train, Y_train), f)  # , Z_test, Y_test
        print("Saved latent representations to " + latent_pkl_name)
        scipy.io.savemat(
            latent_mat_name,
            mdict={
                "Z_train": Z_train,
                # 'Z_test': Z_test,
                "Y_train": Y_train,
                # 'Y_test': Y_test
            },
        )

    """Training begins here"""
    min_loss = math.inf  # >= python 3.5
    min_loss_epoch = None
    loss_name = os.path.join(cfg.data.res_dir, "train_loss.txt")
    loss_plot_name = os.path.join(cfg.data.res_dir, "train_loss_plot.pdf")
    test_results_name = os.path.join(cfg.data.res_dir, "test_results.txt")
    if os.path.exists(loss_name) and not cfg.data.keep_old:
        os.remove(loss_name)

    if cfg.data.only_test:
        epoch = cfg.model.continue_from

    start_epoch = cfg.model.continue_from if cfg.model.continue_from is not None else 0
    for epoch in range(start_epoch + 1, cfg.optimization.epochs + 1):
        if cfg.model.predictor:
            train_loss, recon_loss, kld_loss, pred_loss = train(epoch)
        else:
            train_loss, recon_loss, kld_loss = train(epoch)
            pred_loss = 0.0
        # with open(loss_name, 'a') as loss_file:
        #     loss_file.write("{:.2f} {:.2f} {:.2f} {:.2f}\n".format(
        #         train_loss/len(train_data),
        #         recon_loss/len(train_data),
        #         kld_loss/len(train_data),
        #         pred_loss/len(train_data),
        #         ))
        scheduler.step(train_loss)
        print("save to wandb...")
        wandb.log(
            {
                "train_loss": train_loss / len(train_data),
                "recon_loss": recon_loss / len(train_data),
                "kld_loss": kld_loss / len(train_data),
                "pred_loss": pred_loss / len(train_data),
            }
        )

        if epoch % cfg.data.save_interval == 0:
            print("save current model...")
            model_name = os.path.join(
                cfg.data.res_dir, f"model_checkpoint{epoch}_{cfg.optimization.batch_size}.pth"
            )
            optimizer_name = os.path.join(
                cfg.data.res_dir, f"optimizer_checkpoint{epoch}_{cfg.optimization.batch_size}.pth"
            )
            scheduler_name = os.path.join(
                cfg.data.res_dir, f"scheduler_checkpoint{epoch}_{cfg.optimization.batch_size}.pth"
            )
            torch.save(model.state_dict(), model_name)
            torch.save(optimizer.state_dict(), optimizer_name)
            torch.save(scheduler.state_dict(), scheduler_name)
            # print("visualize reconstruction examples...")
            # visualize_recon(epoch)
            print("extract latent representations...")
            save_latent_representations(epoch)
            print("sample from prior...")
            sampled = model.generate_sample(cfg.data.sample_number)
            for i, g in enumerate(sampled):
                types = g.vs["type"][1:-1]
                n_zeros = sum([t == 0 for t in types]) / (len(types) + 0.0001)
                wandb.log({"n_zeros": n_zeros})
                namei = "graph_{}_sample{}".format(epoch, i)
            #     plot_DAG(g, cfg.data.res_dir, namei, data_type=cfg.data.type)
            # print("plot train loss...")
            # losses = np.loadtxt(loss_name)
            # if losses.ndim == 1:
            #     continue
            # fig = plt.figure()
            # num_points = losses.shape[0]
            # plt.plot(range(1, num_points+1), losses[:, 0], label='Total')
            # plt.plot(range(1, num_points+1), losses[:, 1], label='Recon')
            # plt.plot(range(1, num_points+1), losses[:, 2], label='KLD')
            # plt.plot(range(1, num_points+1), losses[:, 3], label='Pred')
            # plt.xlabel('Epoch')
            # plt.ylabel('Train loss')
            # plt.legend()
            # plt.savefig(loss_plot_name)

    """Sampling"""
    graph_list = []
    i = feasible = 0
    while i < cfg.data.sample_number:
        feasible += 1
        g = model.generate_sample(1)[0]
        # Test that sample is valid
        if is_valid_DAG(g):
            i += 1
            # Convert to tensor
            nodes = torch.tensor(g.vs["type"][1:-1])
            edges = torch.tensor(g.get_adjacency().data)
            graph = [nodes, edges]
            # Append to test list
            graph_list.append(graph)

    print("Feasible samples: ", feasible)
    print("Percentage feasible: ", feasible / cfg.data.sample_number)

    """Metrics"""
    if cfg.data.name in ["er_dag", "ba_dag"]:
        datamodule = SyntheticDataModule(cfg)
        metrics_dictionary = {
            "er_dag": [
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
            "ba_dag": [
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
        }
        sampling_metrics = SyntheticSamplingMetrics(
            datamodule,
            acyclic=cfg.dataset.acyclic,
            metrics_list=metrics_dictionary[cfg.data.name],
            graph_type=cfg.dataset.graph_type,
        )
    elif cfg.data.name == "tpu_tile":
        datamodule = TPUGraphDataModule(cfg)
        sampling_metrics = TPUSamplingMetrics(datamodule)

    # defining dummy arguments and loading reference metrics
    ref_metrics_path = datamodule.train_dataloader().dataset.root

    train_metrics = load_pickle(os.path.join(ref_metrics_path, f"ref_metrics.pkl"))

    dummy_kwargs = {
        "name": "ref_metrics",
        "current_epoch": 0,
        "val_counter": 0,
        "local_rank": 0,
        "ref_metrics": {"val": train_metrics["val"], "test": train_metrics["test"]},
    }
    metrics_results = sampling_metrics.forward(graph_list, **dummy_kwargs, test=True)

    # Save metrics
    print("Saving metrics to", os.path.join(ref_metrics_path, f"ref_metrics_DVAE.pkl"))
    save_pickle(
        metrics_results, os.path.join(ref_metrics_path, f"ref_metrics_DVAE.pkl")
    )


if __name__ == "__main__":
    main()
