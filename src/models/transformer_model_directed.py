import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from src import utils
from src.diffusion import diffusion_utils
from src.models.layers import Xtoy, Etoy, masked_softmax, SPENodes, SPEEdges


def get_timing_signal_1d(
    length,
    channels,
    device,
    min_timescale=1.0,
    max_timescale=1.0e4,
    start_index=0,
    position=None,
):
    """
    Gets a bunch of sinusoids of different frequencies for positional encoding.

    Args:
        length (int): Length of the timing signal sequence.
        channels (int): Size of timing embeddings to create.
        min_timescale (float): Minimum timescale for sinusoidal signals.
        max_timescale (float): Maximum timescale for sinusoidal signals.
        start_index (int): Index of the first position.

    Returns:
        torch.Tensor: A tensor of timing signals of shape [1, length, channels].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if position is None:
        position = torch.arange(
            start_index, start_index + length, dtype=torch.float32, device=device
        )
    else:
        pass

    num_timescales = channels // 2
    log_timescale_increment = math.log(max_timescale / min_timescale) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        -torch.arange(num_timescales, dtype=torch.float32, device=device)
        * log_timescale_increment
    )
    scaled_time = torch.outer(position, inv_timescales)

    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    # Handle odd channels by padding
    if channels % 2 != 0:
        signal = torch.nn.functional.pad(signal, (0, 1))

    return signal.view(1, length, channels)


def get_rope_encoding(
    length, channels, device=None, base=10000, start_index=0, position=None
):
    """
    Computes Rotary Positional Encoding (RoPE).

    Args:
        length (int): Length of the sequence.
        channels (int): Size of positional embeddings (must be even).
        device (torch.device, optional): Device to place the tensor on.
        base (float, optional): Base for RoPE frequency computation.
        start_index (int, optional): Starting index of positions.
        position (torch.Tensor, optional): Precomputed positions tensor.

    Returns:
        torch.Tensor: RoPE angles of shape [1, length, channels].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if position is None:
        position = torch.arange(
            start_index, start_index + length, dtype=torch.float32, device=device
        )

    assert channels % 2 == 0, "RoPE requires an even number of channels."

    half_channels = channels // 2
    theta = 1.0 / (
        base
        ** (torch.arange(0, channels, 2, dtype=torch.float32, device=device) / channels)
    )
    idx_theta = torch.outer(position, theta)  # Shape: [length, half_channels/2]

    sin_theta = torch.sin(idx_theta)
    cos_theta = torch.cos(idx_theta)

    rope = torch.cat([sin_theta, cos_theta], dim=-1)  # Shape: [length, channels]

    return rope.unsqueeze(0)  # Shape: [1, length, channels]


class XEyTransformerLayer(nn.Module):
    """Transformer that updates node, edge and global features
    d_x: node features
    d_e: edge features
    dz : global features
    n_head: the number of heads in the multi_head_attention
    dim_feedforward: the dimension of the feedforward network model after self-attention
    dropout: dropout probablility. 0 to disable
    layer_norm_eps: eps value in layer normalizations.
    """

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        dim_ffX: int = 2048,
        dim_ffE: int = 128,
        dim_ffy: int = 2048,
        dropout: float = 0.1,
        dual: bool = True,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        kw = {"device": device, "dtype": dtype}
        super().__init__()

        if dual:
            self.self_attn = DualNodeEdgeBlock(dx, de, dy, n_head, **kw)
        else:
            self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """Pass the input through the encoder layer.
        X: (bs, n, d)
        E: (bs, n, n, d)
        y: (bs, dy)
        node_mask: (bs, n) Mask for the src keys per batch (optional)
        Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class DualNodeEdgeBlock(nn.Module):
    """Self-attention layer with dual attention mechanism (Source-to-Target & Target-to-Source)."""

    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention weight matrices for dual attention mechanism
        self.q_s = Linear(dx, dx)
        self.k_s = Linear(dx, dx)
        self.v_s = Linear(dx, dx)

        self.q_t = Linear(dx, dx)
        self.k_t = Linear(dx, dx)
        self.v_t = Linear(dx, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)  # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # FiLM E to X
        self.e_mul_s = Linear(de, dx)
        self.e_add_s = Linear(de, dx)
        self.e_mul_t = Linear(de, dx)
        self.e_add_t = Linear(de, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

        # Gating mechanism
        self.gate_x = Linear(2 * dx, dx)

        # Learned edge gating
        self.edge_gate_linear = Linear(de, n_head)
        self.edge_gate_linear_t = Linear(de, n_head)

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, dx        node features
        :param E: bs, n, n, de     edge features
        :param y: bs, dy           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)

        # CHANGE 2: Save original inputs for residual connections
        X_res = X
        E_res = E
        y_res = y

        # Compute Source (S) Queries, Keys, and Values
        Q_s = self.q_s(X) * x_mask
        K_s = self.k_s(X) * x_mask
        V_s = self.v_s(X) * x_mask

        # Compute Target (T) Queries, Keys, and Values
        Q_t = self.q_t(X) * x_mask
        K_t = self.k_t(X) * x_mask
        V_t = self.v_t(X) * x_mask

        # Reshape for multi-head attention
        Q_s = Q_s.reshape((bs, n, self.n_head, self.df))
        K_s = K_s.reshape((bs, n, self.n_head, self.df))
        V_s = V_s.reshape((bs, n, self.n_head, self.df))

        Q_t = Q_t.reshape((bs, n, self.n_head, self.df))
        K_t = K_t.reshape((bs, n, self.n_head, self.df))
        V_t = V_t.reshape((bs, n, self.n_head, self.df))

        Q_s = Q_s.unsqueeze(2)
        K_s = K_s.unsqueeze(1)
        Q_t = Q_t.unsqueeze(2)
        K_t = K_t.unsqueeze(1)

        # Compute source-to-target and target-to-source attention scores
        Y_st = Q_s * K_t / math.sqrt(self.df)
        Y_ts = Q_t * K_s / math.sqrt(self.df)

        # Leverage edge features to attn
        E_transpose = torch.permute(E, [0, 2, 1, 3])

        E1_s = self.e_mul_s(E) * e_mask1 * e_mask2
        E1_t = self.e_mul_t(E_transpose) * e_mask1 * e_mask2
        E2_s = self.e_add_s(E) * e_mask1 * e_mask2
        E2_t = self.e_add_t(E_transpose) * e_mask1 * e_mask2

        E1_s = E1_s.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
        E1_t = E1_t.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
        E2_s = E2_s.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
        E2_t = E2_t.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self-attention scores.
        Y_st = Y_st * (E1_s + 1) + E2_s
        Y_ts = Y_ts * (E1_t + 1) + E2_t

        # Compute edge gating factors from E and its transpose
        edge_gate = torch.sigmoid(self.edge_gate_linear(E))  # shape: (bs, n, n, n_head)
        edge_gate_t = torch.sigmoid(
            self.edge_gate_linear_t(E_transpose)
        )  # shape: (bs, n, n, n_head)
        # Use learned gating factors to modulate attention scores
        Y_st = Y_st * (edge_gate.unsqueeze(-1) * (E1_s + 1)) + E2_s
        Y_ts = Y_ts * (edge_gate_t.unsqueeze(-1) * (E1_t + 1)) + E2_t

        # Incorporate y to E
        newE = Y_st.flatten(start_dim=3)  # bs, n, n, dx
        # newE = (Y_st/2 + Y_ts/2).flatten(start_dim=3)         # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2  # bs, n, n, de
        # Add residual connection for edge features
        newE = (E_res + newE) * e_mask1 * e_mask2
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute softmax attention weights
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)

        # Aggregated attention
        # Step 1: Compute weighted values
        attn = torch.concatenate([Y_st, Y_ts], dim=2)  # (bs, n, 2n, de)
        attn_mask = torch.concatenate(
            [softmax_mask, softmax_mask], dim=2
        )  # (bs, n, 2n, de)
        attn = masked_softmax(attn, attn_mask, dim=2)  # (bs, n, 2n, de)
        weighted_V = torch.concatenate([V_t, V_s], dim=1)  # (bs, 2n, dx)
        # Step 2: Flatten and merge
        weighted_V = (
            (attn * weighted_V.unsqueeze(1)).sum(dim=2).flatten(start_dim=2)
        )  # (bs, n, dx)

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V
        # Apply gating mechanism to fuse original X with the FiLM-modulated output
        modulated_x = yx1 + (yx2 + 1) * weighted_V
        gate = torch.sigmoid(self.gate_x(torch.cat([X, weighted_V], dim=-1)))
        newX = gate * X_res + (1 - gate) * modulated_x

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X and E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)  # bs, dy
        # Add residual connection for global features
        new_y = y_res + new_y

        return newX, newE, new_y


class NodeEdgeBlock(nn.Module):
    """Self attention layer that also updates the representations on the edges."""

    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)  # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask  # (bs, n, dx)
        K = self.k(X) * x_mask  # (bs, n, dx)
        diffusion_utils.assert_correctly_masked(Q, x_mask)

        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df
        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)  # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)  # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)  # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2  # bs, n, n, de
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)  # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask  # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)  # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)  # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X and E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)  # bs, dy

        return newX, newE, new_y


class GraphTransformerDirected(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """

    def __init__(
        self,
        n_layers: int,
        directed: bool,
        pos_enc: str,
        input_dims: dict,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims: dict,
        spe_dims: dict,
        act_fn_in: nn.ReLU,
        act_fn_out: nn.ReLU,
        dual: bool,
        pos_enc_frac: float,
        pos_enc_mode: str,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.directed = directed
        self.pos_enc = pos_enc
        self.pos_enc_frac = pos_enc_frac
        self.pos_enc_mode = pos_enc_mode
        self.dual = dual

        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]

        # self.spe_q_dim = spe_dims["q_dim"] if pos_enc == "spe" else 0
        # self.spe_pe_dim = spe_dims["pe_dim"] if pos_enc == "spe" else 0
        # self.spe_out_dim = spe_dims["out_dim"] if pos_enc == "spe" else 0

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"] + self.spe_out_dim, hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            act_fn_in,
        )

        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"] + self.spe_out_dim, hidden_mlp_dims["E"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            act_fn_in,
        )

        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"], hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            act_fn_in,
        )

        self.tf_layers = nn.ModuleList(
            [
                XEyTransformerLayer(
                    dx=hidden_dims["dx"],  
                    de=hidden_dims["de"],  
                    dy=hidden_dims["dy"],
                    n_head=hidden_dims["n_head"],
                    dim_ffX=hidden_dims["dim_ffX"],
                    dim_ffE=hidden_dims["dim_ffE"],
                    dual=self.dual,
                )
                for i in range(n_layers)
            ]
        )

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            act_fn_out,  
            nn.Linear(hidden_mlp_dims["X"], output_dims["X"]),
        )

        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
            act_fn_out, 
            nn.Linear(hidden_mlp_dims["E"], output_dims["E"]),
        )

        self.mlp_out_y = nn.Sequential(
            nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["y"], output_dims["y"]),
        )

        # Experiments trying to use an SPE network for positional encoding
        # self.spe_nodes = SPENodes(
        #     pe_dim=spe_dims["pe_dim"],
        #     q_dim=spe_dims["q_dim"],
        #     out_dim=spe_dims["out_dim"],
        #     hidden_dim=spe_dims["hidden_dim"],
        #     num_layers=spe_dims["num_layers"],
        #     norm=None,
        # )
        # self.spe_edges = SPEEdges(
        #     pe_dim=spe_dims["pe_dim"],
        #     q_dim=spe_dims["q_dim"],
        #     out_dim=spe_dims["out_dim"],
        #     hidden_dim=spe_dims["hidden_dim"],
        #     num_layers=spe_dims["num_layers"],
        #     norm=None,
        # )

    def forward(self, X, E, y, node_mask, edge_index=None, batch=None):
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., : self.out_dim_X]
        E_to_out = E[..., : self.out_dim_E]
        y_to_out = y[..., : self.out_dim_y]

        # if self.pos_enc == "spe":
        #     # Prepare the eigenvalues and eigenvectors
        #     eigenvalues = y[..., self.out_dim_y :][:, 1:-1].reshape(
        #         bs, self.spe_q_dim, self.spe_pe_dim
        #     )  # (b, q, k)
        #     evec0 = X[..., self.out_dim_X :].reshape(
        #         bs, n, 2 * self.spe_q_dim - 1, self.spe_pe_dim
        #     )
        #     eigenvectors = torch.cat(
        #         (
        #             torch.complex(
        #                 evec0[:, :, 0], torch.zeros_like(evec0[:, :, 0])
        #             ).unsqueeze(2),
        #             torch.complex(evec0[:, :, 1::2], evec0[:, :, 2::2]),
        #         ),
        #         dim=2,
        #     )  # (b, n, q, k)

        #     # Pass through SPE network
        #     spe_nodes = self.spe_nodes(
        #         eigenvalues, eigenvectors, node_mask, edge_index, batch
        #     )
        #     spe_edges = self.spe_edges(eigenvalues, eigenvectors, node_mask)

        #     X = torch.cat((X, spe_nodes), dim=-1)
        #     E = torch.cat((E, spe_edges), dim=-1)

        new_E = self.mlp_in_E(E)
        if not self.directed:
            new_E = 1 / 2 * (new_E + torch.transpose(new_E, 1, 2))

        after_in = utils.PlaceHolder(
            X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)
        ).mask(node_mask, directed=self.directed)
        X, E, y = after_in.X, after_in.E, after_in.y

        # Experiments with other positional encodings
        # if self.pos_enc not in ["none", "spe"]:
        #     if self.pos_enc == "sinusoidal":
        #         pe = get_timing_signal_1d(
        #             X.shape[1], X.shape[2], device=X.device
        #         ).expand(bs, -1, -1)

        #     elif self.pos_enc == "rope":
        #         pe = get_rope_encoding(X.shape[1], X.shape[2], device=X.device).expand(
        #             bs, -1, -1
        #         )

        #     else:
        #         raise NotImplementedError(
        #             f"Positional encoding {self.pos_enc} not implemented"
        #         )
    
        #     X = X + pe

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)  

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        X = X + X_to_out
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        if not self.directed:
            E = 1 / 2 * (E + torch.transpose(E, 1, 2))

        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask, directed=self.directed)
