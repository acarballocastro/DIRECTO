import os
import pathlib
import torch
import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class TPUGraphDataset(InMemoryDataset):
    """
    TPU Graph Dataset for processing TPU Tile graphs into a PyTorch Geometric-compatible format.
    """

    def __init__(
        self, split, root, transform=None, pre_transform=None, pre_filter=None
    ):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def processed_file_names(self):
        return [f"{self.split}.pt"]

    def download(self):
        """
        Download raw data files. Taken from LayerDAG
        """

        train_url = "https://raw.githubusercontent.com/Graph-COM/LayerDAG/main/data_files/tpu_tile_processed/train.pth"
        val_url = "https://raw.githubusercontent.com/Graph-COM/LayerDAG/main/data_files/tpu_tile_processed/val.pth"
        test_url = "https://raw.githubusercontent.com/Graph-COM/LayerDAG/main/data_files/tpu_tile_processed/test.pth"

        train_data = torch.load(download_url(train_url, self.raw_dir))
        val_data = torch.load(download_url(val_url, self.raw_dir))
        test_data = torch.load(download_url(test_url, self.raw_dir))

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        """
        Processes raw TPU graph datasets into a PyTorch Geometric-compatible format.
        """
        file_idx = {"train": 0, "val": 1, "test": 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for src, dst, x_n, y in zip(
            raw_dataset["src_list"],
            raw_dataset["dst_list"],
            raw_dataset["x_n_list"],
            raw_dataset["y_list"],
        ):
            edge_index = torch.vstack((src, dst))
            # No edge attributes so setting them all to 1
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            x = F.one_hot(x_n, num_classes=47).float()
            y = torch.zeros([1, 0]).float()
            # TODO: we remove the label information because it is continuous and not supported
            # y = torch.tensor(y, dtype=torch.float).reshape((1, 1))
            num_nodes = x.size(0)
            data = torch_geometric.data.Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class TPUGraphDataModule(AbstractDataModule):
    """
    Data module for TPU Tile datasets compatible with PyTorch Geometric.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {
            "train": TPUGraphDataset(split="train", root=root_path),
            "val": TPUGraphDataset(split="val", root=root_path),
            "test": TPUGraphDataset(split="test", root=root_path),
        }

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class TPUDatasetInfos(AbstractDatasetInfos):
    """
    Metadata and information about the TPU Graph Dataset.
    """

    def __init__(self, datamodule, dataset_config):
        # self.datamodule = datamodule
        self.name = "tpu_graphs"
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
