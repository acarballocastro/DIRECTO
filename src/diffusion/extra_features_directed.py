import torch
from src import utils
import networkx as nx


class DummyExtraFeatures:
    def __init__(self):
        """This class does not compute anything, just returns empty tensors."""

    def __call__(self, noisy_data):
        X = noisy_data["X_t"]
        E = noisy_data["E_t"]
        y = noisy_data["y_t"]
        empty_x = X.new_zeros((*X.shape[:-1], 0))
        empty_e = E.new_zeros((*E.shape[:-1], 0))
        empty_y = y.new_zeros((y.shape[0], 0))
        return utils.PlaceHolder(X=empty_x, E=empty_e, y=empty_y)


class ExtraDirectedFeatures:
    def __init__(self, extra_features_type, rrwp_steps, restart_prob, dataset_info):
        self.max_n_nodes = dataset_info.max_n_nodes
        self.features_type = extra_features_type
        self.rrwp_steps = rrwp_steps
        self.restart_prob = restart_prob
        self.RRWP = RRWPFeatures()
        self.RRWP_PPR = RRWPFeatures(mode="PPR", restart_prob=restart_prob)
        self.mag_eigenfeatures = MagneticEigenFeatures()
        self.scc_ordering = SCCOrdering()
        self.bfs_ordering = BFSDFSOrdering("bfs")
        self.dfs_ordering = BFSDFSOrdering("dfs")

    def __call__(self, noisy_data):
        n = noisy_data["node_mask"].sum(dim=1).unsqueeze(1) / self.max_n_nodes
        # X = noisy_data['X_t']
        E = noisy_data["E_t"]
        # y = noisy_data['y_t']
        empty_e = E.new_zeros((*E.shape[:-1], 0)).type_as(E)

        if self.features_type == "eigenvalues":
            (
                _,
                _,
                n_components,
                batched_eigenvalues,
                nonlcc_indicator,
                k_lowest_eigvec,
            ) = self.mag_eigenfeatures(q=0, noisy_data=noisy_data)
            return utils.PlaceHolder(
                X=torch.cat(
                    (nonlcc_indicator, k_lowest_eigvec),
                    dim=-1,
                ),
                E=empty_e,
                y=torch.hstack((n, n_components, batched_eigenvalues)),
            )

        elif self.features_type == "magnetic-eigenvalues":
            eigenvalues, eigenvectors, _ = self.mag_eigenfeatures(
                q=0.25, noisy_data=noisy_data
            )
            return utils.PlaceHolder(
                X=torch.cat(
                    (eigenvectors.real, eigenvectors.imag),
                    dim=-1,
                ),
                E=empty_e,
                y=torch.hstack((n, eigenvalues)),
            )

        elif self.features_type == "mult-magnetic-eigenvalues":
            # 5 different values of q: 0, 0.1, 0.2, 0.3, 0.4
            eigenvalues0, eigenvectors0, _, _, _, _ = self.mag_eigenfeatures(
                q=0, noisy_data=noisy_data
            )
            eigenvalues1, eigenvectors1, _ = self.mag_eigenfeatures(
                q=0.1, noisy_data=noisy_data
            )
            eigenvalues2, eigenvectors2, _ = self.mag_eigenfeatures(
                q=0.2, noisy_data=noisy_data
            )
            eigenvalues3, eigenvectors3, _ = self.mag_eigenfeatures(
                q=0.3, noisy_data=noisy_data
            )
            eigenvalues4, eigenvectors4, _ = self.mag_eigenfeatures(
                q=0.4, noisy_data=noisy_data
            )
            return utils.PlaceHolder(
                X=torch.cat(
                    (
                        eigenvectors0,
                        eigenvectors1.real,
                        eigenvectors1.imag,
                        eigenvectors2.real,
                        eigenvectors2.imag,
                        eigenvectors3.real,
                        eigenvectors3.imag,
                        eigenvectors4.real,
                        eigenvectors4.imag,
                    ),
                    dim=-1,
                ),
                E=empty_e,
                y=torch.hstack(
                    (
                        n,
                        eigenvalues0,
                        eigenvalues1,
                        eigenvalues2,
                        eigenvalues3,
                        eigenvalues4,
                    )
                ),
            )

        elif self.features_type == "mult10-magnetic-eigenvalues":
            # 5 different values of q: 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1
            eigenvalues0, eigenvectors0, _ = self.mag_eigenfeatures(
                q=0.01, noisy_data=noisy_data
            )
            eigenvalues1, eigenvectors1, _ = self.mag_eigenfeatures(
                q=0.02, noisy_data=noisy_data
            )
            eigenvalues2, eigenvectors2, _ = self.mag_eigenfeatures(
                q=0.03, noisy_data=noisy_data
            )
            eigenvalues3, eigenvectors3, _ = self.mag_eigenfeatures(
                q=0.04, noisy_data=noisy_data
            )
            eigenvalues4, eigenvectors4, _ = self.mag_eigenfeatures(
                q=0.05, noisy_data=noisy_data
            )
            eigenvalues5, eigenvectors5, _ = self.mag_eigenfeatures(
                q=0.06, noisy_data=noisy_data
            )
            eigenvalues6, eigenvectors6, _ = self.mag_eigenfeatures(
                q=0.07, noisy_data=noisy_data
            )
            eigenvalues7, eigenvectors7, _ = self.mag_eigenfeatures(
                q=0.08, noisy_data=noisy_data
            )
            eigenvalues8, eigenvectors8, _ = self.mag_eigenfeatures(
                q=0.09, noisy_data=noisy_data
            )
            eigenvalues9, eigenvectors9, _ = self.mag_eigenfeatures(
                q=0.1, noisy_data=noisy_data
            )
            return utils.PlaceHolder(
                X=torch.cat(
                    (
                        eigenvectors0.real,
                        eigenvectors0.imag,
                        eigenvectors1.real,
                        eigenvectors1.imag,
                        eigenvectors2.real,
                        eigenvectors2.imag,
                        eigenvectors3.real,
                        eigenvectors3.imag,
                        eigenvectors4.real,
                        eigenvectors4.imag,
                        eigenvectors5.real,
                        eigenvectors5.imag,
                        eigenvectors6.real,
                        eigenvectors6.imag,
                        eigenvectors7.real,
                        eigenvectors7.imag,
                        eigenvectors8.real,
                        eigenvectors8.imag,
                        eigenvectors9.real,
                        eigenvectors9.imag,
                    ),
                    dim=-1,
                ),
                E=empty_e,
                y=torch.hstack(
                    (
                        n,
                        eigenvalues0,
                        eigenvalues1,
                        eigenvalues2,
                        eigenvalues3,
                        eigenvalues4,
                        eigenvalues5,
                        eigenvalues6,
                        eigenvalues7,
                        eigenvalues8,
                        eigenvalues9,
                    )
                ),
            )

        elif self.features_type == "rrwp":
            rrwp_edge_attr = self.RRWP(noisy_data)
            diag_index = torch.arange(rrwp_edge_attr.shape[1])
            rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]

            return utils.PlaceHolder(
                X=rrwp_node_attr,
                E=rrwp_edge_attr,
                y=n,
            )

        elif self.features_type == "rrwp-ppr":
            rrwp_edge_attr = self.RRWP_PPR(noisy_data)
            diag_index = torch.arange(rrwp_edge_attr.shape[1])
            rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]

            return utils.PlaceHolder(
                X=rrwp_node_attr,
                E=rrwp_edge_attr,
                y=n,
            )

        elif self.features_type == "scc":
            sccs = self.scc_ordering(noisy_data)
            return utils.PlaceHolder(
                X=sccs,
                E=empty_e,
                y=n,
            )

        elif self.features_type == "bfs":
            bfs = self.bfs_ordering(noisy_data)
            return utils.PlaceHolder(
                X=bfs,
                E=empty_e,
                y=n,
            )

        elif self.features_type == "dfs":
            dfs = self.dfs_ordering(noisy_data)
            return utils.PlaceHolder(
                X=dfs,
                E=empty_e,
                y=n,
            )

        elif self.features_type == "scc-rrwp":
            sccs = self.scc_ordering(noisy_data)
            rrwp_edge_attr = self.RRWP_PPR(noisy_data)
            diag_index = torch.arange(rrwp_edge_attr.shape[1])
            rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]
            return utils.PlaceHolder(
                X=torch.cat(
                    (rrwp_node_attr, sccs),
                    dim=-1,
                ),
                E=rrwp_edge_attr,
                y=n,
            )

        elif self.features_type == "scc-mageigenv":
            sccs = self.scc_ordering(noisy_data)
            eigenvalues, eigenvectors, _ = self.mag_eigenfeatures(
                q=0.25, noisy_data=noisy_data
            )
            return utils.PlaceHolder(
                X=torch.cat(
                    (eigenvectors.real, eigenvectors.imag, sccs),
                    dim=-1,
                ),
                E=empty_e,
                y=torch.hstack((n, eigenvalues)),
            )

        elif self.features_type == "rrwp-mageigenv":
            eigenvalues, eigenvectors, _ = self.mag_eigenfeatures(
                q=0.25, noisy_data=noisy_data
            )
            rrwp_edge_attr = self.RRWP_PPR(noisy_data)
            diag_index = torch.arange(rrwp_edge_attr.shape[1])
            rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]

            return utils.PlaceHolder(
                X=torch.cat(
                    (eigenvectors.real, eigenvectors.imag, rrwp_node_attr),
                    dim=-1,
                ),
                E=rrwp_edge_attr,
                y=torch.hstack((n, eigenvalues)),
            )

        elif self.features_type == "all":
            eigenvalues, eigenvectors, _ = self.mag_eigenfeatures(
                q=0.25, noisy_data=noisy_data
            )
            (
                _,
                _,
                n_components,
                batched_eigenvalues,
                nonlcc_indicator,
                k_lowest_eigvec,
            ) = self.mag_eigenfeatures(
                q=0, noisy_data=noisy_data
            )  # q = 0 correspond to normal eigenvalues
            sccs = self.scc_ordering(noisy_data)
            rrwp_edge_attr = self.RRWP_PPR(noisy_data)
            diag_index = torch.arange(rrwp_edge_attr.shape[1])
            rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]

            return utils.PlaceHolder(
                X=torch.cat(
                    (
                        nonlcc_indicator,
                        k_lowest_eigvec,
                        eigenvectors.real,
                        eigenvectors.imag,
                        rrwp_node_attr,
                        sccs,
                    ),
                    dim=-1,
                ),
                E=rrwp_edge_attr,
                y=torch.hstack((n, n_components, batched_eigenvalues, eigenvalues)),
            )

        else:
            raise ValueError(f"Features type {self.features_type} not implemented")


def get_eigenvalues_features(eigenvalues, A, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 5e-5).sum(dim=-1)
    try:
        assert (n_connected_components > 0).all(), (n_connected_components, ev)
    except:
        import pdb

        pdb.set_trace()

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack(
            (eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues))
        )
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(
        0
    ) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask  # bs, n
    # Add random value to the mask to prevent 0 from becoming the mode
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)  # bs, n
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values  # values: bs -- indices: bs
    mask = ~(first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat(
            (vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2
        )  # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(
        0
    ) + n_connected.unsqueeze(
        2
    )  # bs, 1, k
    indices = indices.expand(-1, n, -1)  # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)  # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


class MagneticEigenFeatures:
    def __init__(
        self,
        q_absolute=True,
        use_symmetric_norm=False,
        k=10,
        k_excl=0,
        sign_rotate=True,
        norm_comps_sep=True,
        l2_norm=False,
    ):
        self.q_absolute = q_absolute
        self.use_symmetric_norm = use_symmetric_norm
        self.k = k
        self.k_excl = k_excl
        self.sign_rotate = sign_rotate
        self.norm_comps_sep = norm_comps_sep
        self.l2_norm = l2_norm

    def compute_magnetic_laplacian(self, A, q, mask):
        """
        adjacency : batched adjacency matrix (bs, n, n)
        normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
        Return:
            L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
        """
        (
            bs,
            max_n,
            _,
        ) = A.shape
        n = mask.sum(dim=1)  # different number of nodes per graph
        A_sym = torch.logical_or(A, A.transpose(1, 2)).to(A.dtype)
        D_sym = A_sym.sum(dim=1)

        if not self.q_absolute:
            m_imag = (A != A.transpose(1, 2)).sum(dim=[1, 2]) // 2
            m_imag = torch.where(m_imag > n, n, m_imag)
            q = q / torch.where(m_imag > 0, m_imag, 1)
        else:
            q = torch.full((bs,), q, dtype=A.dtype, device=A_sym.device)

        theta = 1j * 2 * torch.pi * q.reshape((bs, 1, 1)) * (A - A.transpose(1, 2))

        if self.use_symmetric_norm:
            inv_deg = torch.zeros(
                (bs, max_n, max_n), dtype=A.dtype, device=A_sym.device
            )
            fill = 1.0 / torch.sqrt(torch.where(D_sym > 0, D_sym, torch.inf))
            inv_deg = torch.diagonal_scatter(inv_deg, fill, dim1=1, dim2=2)
            eye = (
                torch.eye(max_n, dtype=A.dtype, device=A_sym.device)
                * mask.unsqueeze(1)
                * mask.unsqueeze(2)
            )
            deg = inv_deg @ A_sym @ inv_deg
            laplacian = eye - deg * torch.exp(theta)
        else:
            deg = torch.zeros((bs, max_n, max_n), dtype=A.dtype, device=A_sym.device)
            deg = torch.diagonal_scatter(deg, D_sym, dim1=1, dim2=2)
            laplacian = deg - A_sym * torch.exp(theta)

        for b in range(bs):
            assert torch.all(laplacian[b][n[b] :, :] == 0) and torch.all(
                laplacian[b][:, n[b] :] == 0
            ), "Laplacian should be zero outside the mask"
        return laplacian

    def __call__(self, q, noisy_data):
        E = noisy_data["E_t"]
        mask = noisy_data["node_mask"]  # For graphs with different number of nodes
        A = E[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        (
            bs,
            n,
            _,
        ) = A.shape

        laplacian = self.compute_magnetic_laplacian(A, q, mask)

        # Upper bound for the diagonal of the laplacian so that we push eigenvalues to the front
        mask_diag = (
            2
            * laplacian.shape[-1]
            * torch.eye(A.shape[-1]).type_as(laplacian).unsqueeze(0)
        )
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))

        if q == 0:
            laplacian = laplacian * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag
            laplacian_r = laplacian.real
            assert torch.allclose(laplacian_r, laplacian_r.transpose(1, 2), atol=1e-6)
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_r)

            eigenvalues = eigenvalues.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(
                eigenvalues=eigenvalues, A=A
            )
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(
                vectors=eigenvectors,
                node_mask=mask,
                n_connected=n_connected_comp,
            )
            eigenvalues = eigenvalues[:, self.k_excl : self.k_excl + self.k]
            eigenvectors = eigenvectors[:, :, self.k_excl : self.k_excl + self.k]
            if self.k > n:
                padded_eval = torch.zeros(
                    (bs, self.k), dtype=eigenvalues.dtype, device=eigenvalues.device
                )
                padded_eval[:, :n] = eigenvalues
                eigenvalues = padded_eval

                padded_evec = torch.zeros(
                    (bs, n, self.k),
                    dtype=eigenvectors.dtype,
                    device=eigenvectors.device,
                )
                padded_evec[:, :n, :n] = eigenvectors
                eigenvectors = padded_evec
            return (
                eigenvalues,
                eigenvectors,
                n_connected_comp,
                batch_eigenvalues,
                nonlcc_indicator,
                k_lowest_eigenvector,
            )

        else:
            device = A.device
            # laplacian = laplacian * mask.unsqueeze(1) * mask.unsqueeze(2)
            # Laplacian regularization
            epsilon = 1e-5
            I = torch.eye(laplacian.size(-1), device=laplacian.device).unsqueeze(0)
            laplacian = laplacian + epsilon * I
            laplacian = laplacian * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

            # Move Laplacian to CPU for eigenvalue computation
            # Necessary because torch.linalg.eigh sometimes fails on GPU
            # See issue: https://github.com/pytorch/pytorch/issues/105359
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian.cpu())
            eigenvalues = eigenvalues.to(device)
            eigenvectors = eigenvectors.to(device)

            eigenvalues = eigenvalues[:, self.k_excl : self.k_excl + self.k]
            eigenvectors = eigenvectors[:, :, self.k_excl : self.k_excl + self.k]

            if self.sign_rotate:
                argmax_i = torch.abs(eigenvectors.real).argmax(dim=1, keepdim=True)
                sign = torch.gather(eigenvectors.real, 1, argmax_i)
                eigenvectors = sign * eigenvectors

                argmax_imag_0 = eigenvectors[:, :, 0].imag.argmax(dim=1)
                rotation = torch.angle(eigenvectors[torch.arange(bs), argmax_imag_0])
                eigenvectors = eigenvectors * torch.exp(-1j * rotation[:, None, :])

            if self.norm_comps_sep:
                eps = 1e-12 / torch.sqrt(torch.tensor(n, dtype=A.dtype))
                if self.l2_norm:
                    scale_real = torch.linalg.norm(
                        eigenvectors.real, dim=1, keepdim=True
                    )
                    real = eigenvectors.real / scale_real
                else:
                    scale_real = (
                        torch.abs(eigenvectors.real).max(dim=1, keepdim=True).values
                    )
                    real = eigenvectors.real / scale_real
                scale_mask = (
                    torch.abs(eigenvectors.real).sum(dim=1, keepdim=True) / n
                ) > eps
                scale_mask = scale_mask.expand_as(eigenvectors)
                eigenvectors[scale_mask] = (
                    real[scale_mask] + 1j * eigenvectors.imag[scale_mask]
                )

                if self.l2_norm:
                    scale_imag = torch.linalg.norm(
                        eigenvectors.imag, dim=1, keepdim=True
                    )
                    imag = eigenvectors.imag / scale_imag
                else:
                    scale_imag = (
                        torch.abs(eigenvectors.imag).max(dim=1, keepdim=True).values
                    )
                    imag = eigenvectors.imag / scale_imag
                scale_mask = (
                    torch.abs(eigenvectors.imag).sum(dim=1, keepdim=True) / n
                ) > eps
                scale_mask = scale_mask.expand_as(eigenvectors)
                eigenvectors[scale_mask] = (
                    eigenvectors.real[scale_mask] + 1j * imag[scale_mask]
                )

            elif not self.l2_norm:
                scale = torch.abs(eigenvectors).max(dim=1, keepdim=True).values
                eigenvectors = eigenvectors / scale

            # Handling rare case where all graphs in a batch have less than k nodes (and k eigenvalues)
            # Padding with zeros to make sure all graphs have the same number of eigenvalues / eigenvectors
            if self.k > n:
                padded_eval = torch.zeros(
                    (bs, self.k), dtype=eigenvalues.dtype, device=eigenvalues.device
                )
                padded_eval[:, :n] = eigenvalues
                eigenvalues = padded_eval

                padded_evec = torch.zeros(
                    (bs, n, self.k),
                    dtype=eigenvectors.dtype,
                    device=eigenvectors.device,
                )
                padded_evec[:, :n, :n] = eigenvectors
                eigenvectors = padded_evec
            return eigenvalues.real, eigenvectors, laplacian


class RRWPFeatures:
    def __init__(self, k=10, mode="original", restart_prob=0.1):
        self.k = k
        self.mode = mode
        self.restart_prob = restart_prob

    def __call__(self, noisy_data):
        k = self.k
        mask = noisy_data["node_mask"]
        A = noisy_data["E_t"][..., 1:].sum(dim=-1).float()  # bs, n, n
        (
            bs,
            n,
            _,
        ) = A.shape

        batch_indices, senders, receivers = torch.nonzero(A, as_tuple=True)

        # Outgoing edges
        out_degrees = A.sum(dim=2)
        inv_degrees = torch.where(out_degrees > 0, 1.0 / out_degrees, torch.tensor(0.0))
        T = A.clone()
        T[batch_indices, senders, receivers] = inv_degrees[batch_indices, senders]
        # Create mask for disconnected nodes (taking batched node mask into account)
        disconnected_nodes = (out_degrees == 0).float() * mask
        T[:, torch.arange(n), torch.arange(n)] += disconnected_nodes  # Add self-loops

        # Incoming edges (reversing the role of senders and receivers)
        in_degrees = A.sum(dim=1)
        inv_degrees = torch.where(in_degrees > 0, 1.0 / in_degrees, torch.tensor(0.0))
        R = A.clone()
        R[batch_indices, senders, receivers] = inv_degrees[batch_indices, receivers]
        # Create mask for disconnected nodes (taking batched node mask into account)
        disconnected_nodes = (in_degrees == 0).float() * mask
        R[:, torch.arange(n), torch.arange(n)] += disconnected_nodes  # Add self-loops

        if self.mode == "PPR":
            assert (
                self.restart_prob > 0 and self.restart_prob < 1
            ), "Restart probability should be between 0 and 1 for PPR"
            I = torch.eye(n, device=A.device).unsqueeze(0).repeat(bs, 1, 1)
            # PPR matrix
            T = (
                self.restart_prob
                * torch.linalg.inv(I + (self.restart_prob - 1) * T).float()
            )
            R = (
                self.restart_prob
                * torch.linalg.inv(I + (self.restart_prob - 1) * R).float()
            )

        id = torch.eye(n, device=A.device).unsqueeze(0).repeat(bs, 1, 1)
        # rrwp_list = [id]
        out_rrwp_list = [id]
        in_rrwp_list = [id]

        for i in range(k - 1):
            # rrwp_list.append(rrwp_list[-1] @ E)
            out_rrwp_list.append(out_rrwp_list[-1] @ T)
            in_rrwp_list.append(in_rrwp_list[-1] @ R)

        return torch.stack(out_rrwp_list + in_rrwp_list, -1)  # bs, n, n, k*2


class SCCOrdering:
    def __init__(self):
        pass

    def __call__(self, noisy_data):
        """
        Manual implementation of the algorithm following:
        https://networkx.org/documentation/stable/_modules/networkx/algorithms/components/strongly_connected.html#strongly_connected_components

        Complexity: O(V + E) linear in number of edges and nodes

        A : (bs, n, n) : adjacency matrix
        """
        A = noisy_data["E_t"][..., 1:].sum(dim=-1).float()  # bs, n, n
        mask = noisy_data["node_mask"]
        (
            bs,
            n,
            _,
        ) = A.shape
        results = []
        node_scc_mapping = []

        for b in range(bs):
            preorder = torch.full((n,), -1, dtype=torch.int32)
            lowlink = torch.full((n,), -1, dtype=torch.int32)
            scc_found = torch.zeros(n, dtype=torch.bool)
            stack = []
            batch_result = []
            node_scc = torch.full((n,), -1, dtype=torch.int32)
            i = 0  # Preorder counter

            # Number of nodes is available in mask
            n_nodes = sum(mask[b])
            for source in range(n_nodes):
                if scc_found[source]:
                    continue
                queue = [source]
                temp_stack = []
                while queue:
                    v = queue[-1]
                    if preorder[v] == -1:
                        preorder[v] = lowlink[v] = i
                        i += 1
                        stack.append(v)
                        temp_stack.append(v)

                    neighbors = (A[b, v] > 0).nonzero(as_tuple=True)[0]
                    unvisited_neighbors = [
                        w.item() for w in neighbors if preorder[w] == -1
                    ]

                    if unvisited_neighbors:
                        queue.append(unvisited_neighbors[0])
                    else:
                        queue.pop()
                        for w in neighbors:
                            if not scc_found[w]:
                                lowlink[v] = min(lowlink[v], lowlink[w])

                        if lowlink[v] == preorder[v]:
                            scc = []
                            while temp_stack:
                                k = temp_stack.pop()
                                scc.append(k)
                                scc_found[k] = True
                                node_scc[k] = len(batch_result)
                                if k == v:
                                    break
                            batch_result.append(scc)

            results.append(batch_result)
            node_scc_mapping.append(node_scc.tolist())

        return torch.tensor(node_scc_mapping, device=A.device).unsqueeze(-1)


class BFSDFSOrdering:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, noisy_data):
        A = noisy_data["E_t"][..., 1:].sum(dim=-1).float()
        mask = noisy_data["node_mask"]
        (
            bs,
            n,
            _,
        ) = A.shape
        ordering = []

        # Create graph for networkx
        for b in range(bs):
            masked_matrix = A[b][:, mask[b]]
            masked_matrix = masked_matrix[mask[b], :]
            G = nx.from_numpy_matrix(
                masked_matrix.cpu().numpy(), create_using=nx.DiGraph
            )

            visited = set()
            node_order = []

            for node in G.nodes():
                if self.mode == "bfs":
                    if node not in visited:
                        bfs_tree = nx.bfs_tree(
                            G, node
                        )  # Returns nodes reachable from source
                        # Might be that a node is visited multiple times from different sources
                        # To fix this we append the nodes uniquely once in the order they are visited
                        component_nodes = [
                            n for n in bfs_tree.nodes() if n not in visited
                        ]
                        node_order.extend(
                            component_nodes
                        )  # Append while preserving order
                        visited.update(component_nodes)  # Mark nodes as visited
                elif self.mode == "dfs":
                    if node not in visited:
                        dfs_tree = nx.dfs_tree(G, node)
                        component_nodes = [
                            n for n in dfs_tree.nodes() if n not in visited
                        ]
                        node_order.extend(component_nodes)
                        visited.update(component_nodes)

            num_false = (mask[b] == False).sum().item()
            node_order.extend(
                [-1] * num_false
            )  # Append -1 for non-existing nodes to keep dimensionality
            ordering.append(node_order)

        return torch.tensor(ordering, device=A.device).unsqueeze(-1)
