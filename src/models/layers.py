import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_dense_batch


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """Map node features to global features"""
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        """X: bs, n, dx."""
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """Map edge features to global features."""
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        """E: bs, n, n, de
        Features relative to the diagonal of E could potentially be added.
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


# SPE network final modules


class MLPNodes(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, norm=None):
        super(MLPNodes, self).__init__()
        assert num_layers >= 2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))

        if norm == "bn":
            self.layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm == "ln":
            self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if norm == "bn":
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm == "ln":
                self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class MLPEdges(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, norm=None):
        super(MLPNodes, self).__init__()
        assert num_layers >= 2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))

        if norm == "bn":
            self.layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm == "ln":
            self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if norm == "bn":
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm == "ln":
                self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class GIN(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            conv = GINConv(
                mlp, node_dim=0
            )  # node_dim = 0 to deal with arbitrary shape pe
            self.convs.append(conv)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
        return x


class SPENodes(nn.Module):
    def __init__(self, pe_dim, q_dim, out_dim, hidden_dim, num_layers, norm=None):
        super(SPENodes, self).__init__()

        eigval_dim = 16
        self.eigval_encoder = MLPNodes(1, 32, eigval_dim, 3, norm=norm)
        self.readout = MLPNodes(
            2 * q_dim * eigval_dim, q_dim * eigval_dim, hidden_dim, 2, norm=norm
        )

        self.gin = GIN(hidden_dim, num_layers)

        self.pe_projection = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, eigenvals, eigenvecs, mask, edge_index, batch):
        # Applying phis and preparing output
        eigenvals = self.eigval_encoder(
            eigenvals.unsqueeze(-1)
        )  # (b, q, k, hidden_dim)
        x = self.weighted_gram_matrix_batched(
            eigenvecs, eigenvals
        )  # (b, n, n, 2 * q * hidden_dim)
        bs, n = x.shape[:2]
        x = self.readout(x)  # (b, n, n, hidden_dim)
        x = x[mask]  # Applying node mask: (be, n, hidden_dim)

        # GIN network (rho node)
        x = self.gin(x, edge_index)  # (be, n, hidden_dim)
        x = (x * mask[batch].unsqueeze(-1)).sum(1)  # (be, hidden_dim)
        x = self.pe_projection(x)  # (be, out_dim)
        x = self.act(x)  # (be, out_dim)

        out_dim = x.shape[-1]
        x_final = torch.zeros((bs, n, out_dim), device=x.device)
        x_final[mask] = x
        return x_final

    def weighted_gram_matrix_batched(self, x, weight):
        # input x: [B, N, pe_dim], weight: [B, q_dim, pe_actual_dim, channels], where pe_dim = q_dim * pe_actual_dim * 2
        # output x: [B, N, N, 2*q_dim * channels], doing inner product along pe_actual_dim axis
        gram = torch.einsum(
            "bnqd, bmqd, bqdc->bnmqc", x, torch.conj(x), weight.type(torch.complex64)
        )
        gram = gram.flatten(3)
        return torch.cat([torch.real(gram), torch.imag(gram)], dim=-1)


class SPEEdges(nn.Module):
    def __init__(
        self, pe_dim, q_dim, out_dim, hidden_dim, num_layers, norm=None, dropout=0.1
    ):
        super(SPEEdges, self).__init__()

        eigval_dim = 16
        self.eigval_encoder = MLPNodes(1, 32, eigval_dim, 3, norm=norm)
        self.readout = nn.Linear(2 * q_dim * eigval_dim, out_dim)

    def forward(self, eigenvals, eigenvecs, mask):
        # Applying phis and preparing output
        eigenvals = self.eigval_encoder(
            eigenvals.unsqueeze(-1)
        )  # (b, q, k, hidden_dim)
        x = self.weighted_gram_matrix_batched(
            eigenvecs, eigenvals
        )  # (b, n, n, 2 * q * hidden_dim)
        x = self.readout(x)  # (b, n, n, output_dim)
        return x

    def weighted_gram_matrix_batched(self, x, weight):
        # input x: [B, N, pe_dim], weight: [B, q_dim, pe_actual_dim, channels], where pe_dim = q_dim * pe_actual_dim * 2
        # output x: [B, N, N, 2*q_dim * channels], doing inner product along pe_actual_dim axis
        gram = torch.einsum(
            "bnqd, bmqd, bqdc->bnmqc", x, torch.conj(x), weight.type(torch.complex64)
        )
        gram = gram.flatten(3)
        return torch.cat([torch.real(gram), torch.imag(gram)], dim=-1)
