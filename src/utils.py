import os
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch, dense_to_sparse
import torch
import omegaconf
import wandb


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs("graphs")
        os.makedirs("chains")
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs("graphs/" + args.general.name)
        os.makedirs("chains/" + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = X * norm_values[0] + norm_biases[0]
    E = E * norm_values[1] + norm_biases[1]
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(
        edge_index, edge_attr
    )
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False, directed=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            if not directed:
                assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

    def split(self, node_mask):
        """Split a PlaceHolder representing a batch into a list of placeholders representing individual graphs."""
        graph_list = []
        batch_size = self.X.shape[0]
        for i in range(batch_size):
            n = torch.sum(node_mask[i], dim=0)
            x = self.X[i, :n]
            e = self.E[i, :n, :n]
            y = self.y[i] if self.y is not None else None
            graph_list.append(PlaceHolder(X=x, E=e, y=y))
        return graph_list


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    config_dict["general"]["local_dir"] = os.getcwd()
    if cfg.model.backbone == 'diffusion':
        project = f"dirgen_{cfg.dataset.name}"
    elif cfg.model.backbone == 'flow':
        project = f"dirflow_{cfg.dataset.name}"
    kwargs = {
        "name": cfg.general.name,
        "project": project, #f"dirgen_{cfg.dataset.name}",
        "config": config_dict,
        "settings": wandb.Settings(_disable_stats=True),
        "reinit": True,
        "mode": cfg.general.wandb,
    }
    wandb.init(**kwargs)
    wandb.save("*.txt")


def to_sparse(X, E, y, node_mask, charge=None):
    """
    This function will only used for development, thus is not efficient
    """
    bs, _, _, de = E.shape
    device = X.device

    node_list = []
    charge_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_list = []
    n_nodes_cumsum = [0]

    for i in range(bs):
        mask_i = node_mask[i]
        n_node = int(mask_i.sum().item())

        Xi = X[i, :n_node]
        Ei = E[i, :n_node, :n_node]
        Ei_argmax = torch.argmax(Ei, dim=-1)

        node_list.append(Xi)

        if charge is not None and charge.numel() > 0:
            charge_list.append(charge[i][:n_node])

        batch_list.append(torch.full((n_node,), i, device=device, dtype=torch.long))

        edge_index_i, edge_attr_i = dense_to_sparse(Ei_argmax)
        edge_index_i += n_nodes_cumsum[-1]
        edge_index_list.append(edge_index_i)
        edge_attr_list.append(edge_attr_i)

        n_nodes_cumsum.append(n_nodes_cumsum[-1] + n_node)

    node = torch.cat(node_list, dim=0).float()
    edge_index = torch.cat(edge_index_list, dim=1).to(device)
    edge_attr = torch.cat(edge_attr_list)
    edge_attr = (
        torch.nn.functional.one_hot(edge_attr, num_classes=de).to(device).float()
    )
    batch = torch.cat(batch_list).to(device)

    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(
        edge_index, edge_attr
    )

    ptr = torch.unique(batch, sorted=True, return_counts=True)[1]
    ptr = torch.cat([torch.tensor([0], device=device), ptr.cumsum(0)]).long()

    if charge is not None and charge.numel() > 0:
        charge_list = torch.vstack(charge_list).float()
    else:
        charge_list = node.new_zeros((*node.shape[:-1], 0))

    return edge_index, batch
