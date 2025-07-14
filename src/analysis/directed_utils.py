import graph_tool.all as gt
import torch.nn as nn
import numpy as np
import networkx as nx
import concurrent.futures
from functools import partial
from tqdm import tqdm

import pygsp as pg
from scipy.linalg import eigvalsh
from scipy.stats import chi2, ks_2samp, powerlaw
from src.analysis.dist_helper import (
    compute_mmd,
    gaussian_emd,
    gaussian_tv,
)
from src.metrics.abstract_metrics import compute_ratios
from torch_geometric.utils import to_networkx

import wandb
import time

############################ Distributional measures ############################

# Degree distribution -----------------------------------------------------------


def degree_worker(G, is_out=True):
    if is_out:
        histogram = [value for _, value in G.in_degree]
    else:
        histogram = [value for _, value in G.out_degree]
    return np.array(histogram)


def degree_stats(
    graph_ref_list, graph_pred_list, is_parallel=True, is_out=True, compute_emd=False
):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    # prev = datetime.now()
    if is_parallel:
        degree_worker_partial = partial(degree_worker, is_out=is_out)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker_partial, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(
                degree_worker_partial, graph_pred_list_remove_empty
            ):
                sample_pred.append(deg_hist)
    else:
        attribute = "out_degree" if is_out else "in_degree"
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(
                [value for _, value in getattr(graph_ref_list[i], attribute)]
            )
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(
                [
                    value
                    for _, value in getattr(graph_pred_list_remove_empty[i], attribute)
                ]
            )
            sample_pred.append(degree_temp)

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    # elapsed = datetime.now() - prev
    # if PRINT_TIME:
    #     print('Time computing degree mmd: ', elapsed)
    return mmd_dist


# Cluster coefficient -----------------------------------------------------------


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
    )
    return hist


def clustering_stats(
    graph_ref_list, graph_pred_list, bins=100, is_parallel=True, compute_emd=False
):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]
    # prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)
            
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd, sigma=1.0 / 10)
        mmd_dist = compute_mmd(
            sample_ref,
            sample_pred,
            kernel=gaussian_emd,
            sigma=1.0 / 10,
            distance_scaling=bins,
        )
    else:
        mmd_dist = compute_mmd(
            sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10
        )

    # elapsed = datetime.now() - prev
    # if PRINT_TIME:
    #     print('Time computing clustering mmd: ', elapsed)
    return mmd_dist


# Spectre -----------------------------------------------------------------------


def spectral_worker(G, n_eigvals=-1):
    # eigs = nx.laplacian_spectrum(G)
    try:
        eigs = eigvalsh(
            np.asarray(nx.directed_laplacian_matrix(G, walk_type="pagerank"))
        )
    except:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1 : n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def spectral_stats(
    graph_ref_list, graph_pred_list, is_parallel=True, n_eigvals=-1, compute_emd=False
):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    # prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                spectral_worker, graph_ref_list, [n_eigvals for i in graph_ref_list]
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                spectral_worker,
                graph_pred_list_remove_empty,
                [n_eigvals for i in graph_ref_list],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i], n_eigvals)
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i], n_eigvals)
            sample_pred.append(spectral_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    # elapsed = datetime.now() - prev
    # if PRINT_TIME:
    #     print('Time computing degree mmd: ', elapsed)
    return mmd_dist


# Wavelet -----------------------------------------------------------------------


def eigh_worker(G):
    L = np.asarray(nx.directed_laplacian_matrix(G, walk_type="pagerank"))
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        eigvals = np.zeros(L[0, :].shape)
        eigvecs = np.zeros(L.shape)
    return (eigvals, eigvecs)


def compute_list_eigh(graph_list, is_parallel=False):
    eigval_list = []
    eigvec_list = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for e_U in executor.map(eigh_worker, graph_list):
                eigval_list.append(e_U[0])
                eigvec_list.append(e_U[1])
    else:
        for i in range(len(graph_list)):
            e_U = eigh_worker(graph_list[i])
            eigval_list.append(e_U[0])
            eigvec_list.append(e_U[1])
    return eigval_list, eigvec_list


def get_spectral_filter_worker(eigvec, eigval, filters, bound=1.4):
    ges = filters.evaluate(eigval)
    linop = []
    for ge in ges:
        linop.append(eigvec @ np.diag(ge) @ eigvec.T)
    linop = np.array(linop)
    norm_filt = np.sum(linop**2, axis=2)
    hist_range = [0, bound]
    hist = np.array(
        [np.histogram(x, range=hist_range, bins=100)[0] for x in norm_filt]
    )  # NOTE: change number of bins
    return hist.flatten()


def spectral_filter_stats(
    eigvec_ref_list,
    eigval_ref_list,
    eigvec_pred_list,
    eigval_pred_list,
    is_parallel=False,
    compute_emd=False,
):
    """Compute the distance between the eigvector sets.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    # prev = datetime.now()

    class DMG(object):
        """Dummy Normalized Graph"""

        lmax = 2

    n_filters = 12
    filters = pg.filters.Abspline(DMG, n_filters)
    bound = np.max(filters.evaluate(np.arange(0, 2, 0.01)))
    sample_ref = []
    sample_pred = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_filter_worker,
                eigvec_ref_list,
                eigval_ref_list,
                [filters for i in range(len(eigval_ref_list))],
                [bound for i in range(len(eigval_ref_list))],
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_filter_worker,
                eigvec_pred_list,
                eigval_pred_list,
                [filters for i in range(len(eigval_pred_list))],
                [bound for i in range(len(eigval_pred_list))],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eigval_ref_list)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    eigvec_ref_list[i], eigval_ref_list[i], filters, bound
                )
                sample_ref.append(spectral_temp)
            except:
                pass
        for i in range(len(eigval_pred_list)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    eigvec_pred_list[i], eigval_pred_list[i], filters, bound
                )
                sample_pred.append(spectral_temp)
            except:
                pass

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    # elapsed = datetime.now() - prev
    # if PRINT_TIME:
    #     print("Time computing spectral filter stats: ", elapsed)
    return mmd_dist


############################ Validity measures ############################


def is_erdos_renyi(G, acyclic=False, expected_p=0.6, strict=True):
    """
    Check how closely a given graph matches an Erdős–Rényi (ER) model.

    Parameters:
    - G: NetworkX graph (assumed undirected)
    - expected_p: Expected probability of an edge in an ER(n, p) model
    - strict: If True, return a boolean decision based on significance threshold (p > 0.9).
              If False, return the computed p-value.

    Returns:
    - True if the graph follows an ER model (if strict=True)
    - p-value of the Wald test (if strict=False)
    """
    num_nodes = len(G)
    num_edges = G.number_of_edges()

    # Compute estimated edge probability
    if num_nodes <= 1:
        return False
    total_possible_edges = (
        num_nodes * (num_nodes - 1) / 2 if acyclic else num_nodes * (num_nodes - 1)
    )
    est_p = num_edges / total_possible_edges

    # Compute Wald statistic
    W = (est_p - expected_p) ** 2 / (est_p * (1 - est_p) + 1e-6)

    # Compute p-value from Chi-squared distribution
    p_value = 1 - chi2.cdf(abs(W), df=1)

    return p_value > 0.9 if strict else p_value


def eval_erdos_renyi(G_list, acyclic=False):
    count = 0
    for gg in tqdm(G_list):
        if is_erdos_renyi(gg, acyclic):
            count += 1
    return count / float(len(G_list))


def is_barabasi_albert(G, strict=True):
    """
    Check if a given graph follows a Barabási–Albert (BA) model.

    Parameters:
    - G: NetworkX graph (assumed undirected)
    - strict: If True, return a boolean decision (p > 0.9).
              If False, return the p-value from Kolmogorov-Smirnov 2 samples test.

    Returns:
    - True if the graph follows a BA model (if strict=True)
    - p-value of the Kolmogorov-Smirnov 2 samples test (if strict=False)
    """
    degrees = np.array([d for _, d in G.degree()])

    # Estimate power-law exponent
    degrees = degrees[degrees > 1]  # ignoring degree=1 nodes
    if len(degrees) < 2:
        return False

    # Verify m < n in the graph
    if len(G) <= 6:
        return False

    # Generate a synthetic BA graph with the same number of nodes and m
    G_synthetic = nx.barabasi_albert_graph(n=len(G), m=6)
    synthetic_degrees = np.array([d for _, d in G_synthetic.degree()])

    # Perform Kolmogorov-Smirnov test
    p_value = 1 - ks_2samp(degrees, synthetic_degrees)[1]

    return p_value > 0.9 if strict else p_value
    # xmin = min(degrees)
    # alpha = 1 + len(degrees) / np.sum(np.log(degrees / xmin))
    # synthetic_degrees = np.round(powerlaw.rvs(alpha - 1, size=len(degrees)) * max(degrees)).astype(int)
    # p_value = ks_2samp(degrees, synthetic_degrees)[1]

    # # A typical BA network should have 2 < alpha < 3
    # return 2 < alpha < 3 if strict else p_value


def eval_barabasi_albert(G_list):
    count = 0
    for G in tqdm(G_list):
        if is_barabasi_albert(G):
            count += 1
    return count / float(len(G_list))


def is_sbm_graph(G, p_intra=0.3, p_inter=0.01, strict=True, refinement_steps=1000):
    """
    Check if how closely given graph matches a SBM with given probabilites by computing mean probability of Wald test statistic for each recovered parameter
    """

    adj = nx.adjacency_matrix(G).toarray()
    idx = adj.nonzero()
    g = gt.Graph(directed=True)
    g.add_edge_list(np.transpose(idx))
    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        if strict:
            return False
        else:
            return 0.0

    # Refine using merge-split MCMC
    for i in range(refinement_steps):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    # b = state.get_blocks()
    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = e.todense()[:n_blocks, :n_blocks]
    # if strict:
    #     if (node_counts > 40).sum() > 0 or (node_counts < 20).sum() > 0 or n_blocks > 5 or n_blocks < 2:
    #         return False

    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = (edge_counts) / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    if strict:
        return p > 0.9  # p value < 10 %
    else:
        return p


def eval_acc_sbm_graph(
    G_list,
    p_intra=0.3,
    p_inter=0.01,
    strict=True,
    refinement_steps=1000,
    is_parallel=True,
):
    count = 0.0
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prob in executor.map(
                is_sbm_graph,
                [gg for gg in G_list],
                [p_intra for i in range(len(G_list))],
                [p_inter for i in range(len(G_list))],
                [strict for i in range(len(G_list))],
                [refinement_steps for i in range(len(G_list))],
            ):
                count += prob
    else:
        for gg in tqdm(G_list):
            count += is_sbm_graph(
                gg,
                p_intra=p_intra,
                p_inter=p_inter,
                strict=strict,
                refinement_steps=refinement_steps,
            )
    return count / float(len(G_list))


def is_planar_graph(G):
    adj = nx.adjacency_matrix(G).toarray()
    adj = np.maximum(adj, adj.T)
    G = nx.from_numpy_array(adj)
    return nx.is_connected(G) and nx.check_planarity(G)[0]


def eval_acc_planar_graph(G_list):
    count = 0
    for gg in tqdm(G_list):
        adj = nx.adjacency_matrix(gg).toarray()
        adj = np.maximum(adj, adj.T)
        G = nx.from_numpy_array(adj)
        if is_planar_graph(G):
            count += 1
    return count / float(len(G_list))


def eval_acc_directed_acyclic_graph(G_list):
    count = 0
    for gg in tqdm(G_list):
        if nx.is_directed_acyclic_graph(gg):
            count += 1
    return count / float(len(G_list))


def is_dag_and_erdos_renyi(G):
    return nx.is_directed_acyclic_graph(G) and is_erdos_renyi(
        G, acyclic=True, expected_p=0.6, strict=True
    )


def eval_acc_dag_and_erdos_renyi(G_list):
    count = 0
    for gg in tqdm(G_list):
        if is_dag_and_erdos_renyi(gg):
            count += 1
    return count / float(len(G_list))


def is_dag_and_barabasi_albert(G):
    return nx.is_directed_acyclic_graph(G) and is_barabasi_albert(G)


def eval_acc_dag_and_barabasi_albert(G_list):
    count = 0
    for gg in tqdm(G_list):
        if is_dag_and_barabasi_albert(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_scene_graph(G_list, val_scene_graph_fn):
    count = 0
    for gg in tqdm(G_list):
        if val_scene_graph_fn(gg):
            count += 1
    return count / float(len(G_list))


def eval_connected_graph(G_list):
    count = 0
    for gg in tqdm(G_list):
        adj = nx.adjacency_matrix(gg).toarray()
        adj = np.maximum(adj, adj.T)
        G = nx.from_numpy_array(adj)
        if nx.is_connected(G):
            count += 1
    return count / float(len(G_list))


def eval_fraction_isomorphic(fake_graphs, train_graphs):
    count = 0
    for fake_g in tqdm(fake_graphs):
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                if nx.is_isomorphic(fake_g, train_g):
                    count += 1
                    break
    return count / float(len(fake_graphs))


def time_eval_fraction_isomorphic(fake_graphs, train_graphs):
    count = 0
    count_non_validated = 0
    for fake_g in tqdm(fake_graphs):
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                timeout, isomorphic = is_isomorphic_with_timeout(fake_g, train_g)
                if isomorphic:
                    count += 1
                    break
                elif timeout:
                    count_non_validated += 1
    return count / float(len(fake_graphs)), count_non_validated / float(
        len(fake_graphs)
    )


# def eval_fraction_unique(fake_graphs, precise=False):
#     count_non_unique = 0
#     fake_evaluated = []
#     for fake_g in tqdm(fake_graphs):
#         unique = True
#         if not fake_g.number_of_nodes() == 0:
#             for fake_old in fake_evaluated:
#                 if precise:
#                     if nx.faster_could_be_isomorphic(fake_g, fake_old):
#                         if nx.is_isomorphic(fake_g, fake_old):
#                             count_non_unique += 1
#                             unique = False
#                             break
#                 else:
#                     if nx.faster_could_be_isomorphic(fake_g, fake_old):
#                         if nx.could_be_isomorphic(fake_g, fake_old):
#                             count_non_unique += 1
#                             unique = False
#                             break
#             if unique:
#                 fake_evaluated.append(fake_g)

#     frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
#         len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs

#     return frac_unique


def eval_fraction_unique_non_isomorphic_valid(
    fake_graphs, train_graphs, validity_func=(lambda x: True)
):
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    fake_evaluated = []
    start = time.time()
    for i, fake_g in enumerate(tqdm(fake_graphs)):
        if i % 100 == 0:
            print(f"Processing graph {i}")
            print(f"Time elapsed: {time.time() - start}")
        unique = True
        for fake_old in fake_evaluated:
            if nx.faster_could_be_isomorphic(fake_g, fake_old):
                if nx.is_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if nx.is_isomorphic(fake_g, train_g):
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
            if non_isomorphic:
                if validity_func(fake_g):
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs
    frac_unique_non_isomorphic = (
        float(len(fake_graphs)) - count_non_unique - count_isomorphic
    ) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set
    frac_unique_non_isomorphic_valid = count_valid / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set and are valid
    return frac_unique, frac_unique_non_isomorphic, frac_unique_non_isomorphic_valid


import threading


def is_isomorphic_worker(fake_g, train_g, result_container):
    """Worker function to check isomorphism and store the result."""
    is_attributed = "label" in train_g.nodes[0]
    node_match_fn = lambda x, y: x["label"] == y["label"] if is_attributed else None
    result_container.append(
        # nx.is_isomorphic(
        #     fake_g, train_g
        # )
        nx.is_isomorphic(
            fake_g,
            train_g,
            node_match=node_match_fn,
        )
    )  # Store result in a shared list


def is_isomorphic_with_timeout(fake_g, train_g, timeout=5):
    """Check if two graphs are isomorphic with a timeout (fixed thread-based)."""
    result_container = []  # Shared list to store the result
    thread = threading.Thread(
        target=is_isomorphic_worker,
        args=(fake_g, train_g, result_container),
        daemon=True,
    )
    thread.start()

    thread.join(timeout)  # Wait for the thread to finish within timeout

    if thread.is_alive():
        print("is_isomorphic took too long!")
        return True, False  # Timeout occurred

    return False, (
        result_container[0] if result_container else False
    )  # Return actual result if available


def time_eval_fraction_unique_non_isomorphic_valid(
    fake_graphs, train_graphs, validity_func=(lambda x: True)
):
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    count_non_unique_non_validated = 0
    count_isomorphic_non_validated = 0
    fake_evaluated = []
    start = time.time()
    for i, fake_g in enumerate(tqdm(fake_graphs)):
        if i % 100 == 0:
            print(f"Processing graph {i}")
            print(f"Time elapsed: {time.time() - start}")
        unique = True
        for fake_old in fake_evaluated:
            if nx.faster_could_be_isomorphic(fake_g, fake_old):
                timeout, isomorphic = is_isomorphic_with_timeout(fake_g, fake_old)
                if isomorphic:
                    count_non_unique += 1
                    unique = False
                    break
                elif timeout:
                    count_non_unique_non_validated += 1
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    timeout, isomorphic = is_isomorphic_with_timeout(fake_g, train_g)
                    if isomorphic:
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
                    elif timeout:
                        count_isomorphic_non_validated += 1
            if non_isomorphic:
                if validity_func(fake_g):
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs
    frac_unique_non_isomorphic = (
        float(len(fake_graphs)) - count_non_unique - count_isomorphic
    ) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set
    frac_unique_non_isomorphic_valid = count_valid / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set and are valid
    frac_non_unique_non_validated = count_non_unique_non_validated / float(
        len(fake_graphs)
    )  # Fraction of graphs non-validated due to timeout
    frac_isomorphic_non_validated = count_non_unique_non_validated / float(
        len(fake_graphs)
    )  # Fraction of graphs non-validated due to timeout
    return (
        frac_unique,
        frac_unique_non_isomorphic,
        frac_unique_non_isomorphic_valid,
        frac_non_unique_non_validated,
        frac_isomorphic_non_validated,
    )


############################ Metrics classes ############################
class DirectedSamplingMetrics(nn.Module):
    def __init__(self, datamodule, acyclic, metrics_list, graph_type, compute_emd):
        super().__init__()

        # self.train_graphs = self.loader_to_nx(datamodule.train_dataloader())
        # self.val_graphs = self.loader_to_nx(datamodule.val_dataloader())
        # self.test_graphs = self.loader_to_nx(datamodule.test_dataloader())
        self.train_digraphs = self.loader_to_nx(
            datamodule.train_dataloader(), directed=True
        )
        self.val_digraphs = self.loader_to_nx(
            datamodule.val_dataloader(), directed=True
        )
        self.test_digraphs = self.loader_to_nx(
            datamodule.test_dataloader(), directed=True
        )
        self.num_graphs_test = len(self.test_digraphs)
        self.num_graphs_val = len(self.val_digraphs)
        self.acyclic = acyclic
        self.compute_emd = compute_emd
        self.metrics_list = metrics_list
        self.graph_type = graph_type

        # Store for wavelet computaiton
        self.val_ref_eigvals, self.val_ref_eigvecs = compute_list_eigh(
            self.val_digraphs
        )
        self.test_ref_eigvals, self.test_ref_eigvecs = compute_list_eigh(
            self.test_digraphs
        )

    def loader_to_nx(self, loader, directed=False):
        networkx_graphs = []
        for i, batch in enumerate(loader):
            data_list = batch.to_data_list()
            for j, data in enumerate(data_list):
                if directed:
                    networkx_graphs.append(
                        to_networkx(
                            data,
                            node_attrs=None,
                            edge_attrs=None,
                            to_undirected=False,
                            remove_self_loops=True,
                        )
                    )
                else:
                    networkx_graphs.append(
                        to_networkx(
                            data,
                            node_attrs=None,
                            edge_attrs=None,
                            to_undirected=True,
                            remove_self_loops=True,
                        )
                    )
        return networkx_graphs

    def is_scene_graph(self, G):
        pass

    def forward(
        self,
        generated_graphs: list,
        ref_metrics,
        name,
        current_epoch,
        val_counter,
        local_rank,
        test=False,
    ):
        # reference_graphs = self.test_graphs if test else self.val_graphs
        reference_digraphs = self.test_digraphs if test else self.val_digraphs
        if local_rank == 0:
            print(
                f"Computing sampling metrics between {len(generated_graphs)} generated graphs and {len(reference_digraphs)}"
            )
        networkx_digraphs = []
        # networkx_graphs = []
        adjacency_matrices = []
        if local_rank == 0:
            print("Building networkx graphs...")

        for graph in generated_graphs:
            node_types, edge_types = graph
            A = edge_types.bool().cpu().numpy()
            adjacency_matrices.append(A)

            nx_digraph = nx.from_numpy_array(
                A, create_using=nx.DiGraph
            )  # we need to specify it is directed
            # nx_graph = nx.from_numpy_array(A, create_using=nx.Graph)

            # need to add labels if it's a scene graph
            if self.graph_type in ["visual_genome", "tpu_tile"]:
                for i, node in enumerate(nx_digraph.nodes()):
                    nx_digraph.nodes[i]["label"] = node_types[i].item()

            networkx_digraphs.append(nx_digraph)
            # networkx_graphs.append(nx_graph)

        np.savez("generated_adjs.npz", *adjacency_matrices)

        to_log = {}
        metrics_prefix = "test" if test else "sampling"

        if "out_degree" in self.metrics_list:
            if local_rank == 0:
                print("Computing out-degree stats...")
            out_degree = degree_stats(
                reference_digraphs,
                networkx_digraphs,
                is_parallel=True,
                is_out=True,
                compute_emd=self.compute_emd,
            )
            to_log[f"{metrics_prefix}/out_degree"] = out_degree
            # if wandb.run:
            #     wandb.run.summary['out_degree'] = out_degree

        if "in_degree" in self.metrics_list:
            if local_rank == 0:
                print("Computing in-degree stats...")
            in_degree = degree_stats(
                reference_digraphs,
                networkx_digraphs,
                is_parallel=True,
                is_out=False,
                compute_emd=self.compute_emd,
            )
            to_log[f"{metrics_prefix}/in_degree"] = in_degree
            # if wandb.run:
            #     wandb.run.summary['in_degree'] = in_degree

        if "clustering" in self.metrics_list:
            if local_rank == 0:
                print("Computing clustering stats...")
            clustering = clustering_stats(
                reference_digraphs,
                networkx_digraphs,
                bins=100,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
            to_log[f"{metrics_prefix}/clustering"] = clustering
            # if wandb.run:
            #     wandb.run.summary['clustering'] = clustering

        if "spectre" in self.metrics_list:
            if local_rank == 0:
                print("Computing spectre stats...")
            spectre = spectral_stats(
                reference_digraphs,
                networkx_digraphs,
                is_parallel=True,
                n_eigvals=-1,
                compute_emd=self.compute_emd,
            )

            to_log[f"{metrics_prefix}/spectre"] = spectre
            # if wandb.run:
            #   wandb.run.summary['spectre'] = spectre

        if "wavelet" in self.metrics_list:
            if local_rank == 0:
                print("Computing wavelet stats...")

            ref_eigvecs = self.test_ref_eigvecs if test else self.val_ref_eigvecs
            ref_eigvals = self.test_ref_eigvals if test else self.val_ref_eigvals

            pred_graph_eigvals, pred_graph_eigvecs = compute_list_eigh(
                networkx_digraphs
            )
            wavelet = spectral_filter_stats(
                eigvec_ref_list=ref_eigvecs,
                eigval_ref_list=ref_eigvals,
                eigvec_pred_list=pred_graph_eigvecs,
                eigval_pred_list=pred_graph_eigvals,
                is_parallel=False,
                compute_emd=self.compute_emd,
            )
            to_log[f"{metrics_prefix}/wavelet"] = wavelet
            # if wandb.run:
            #     wandb.run.summary["wavelet"] = wavelet

        if "connected" in self.metrics_list:
            if local_rank == 0:
                print("Computing connected accuracy...")
            con_acc = eval_connected_graph(networkx_digraphs)
            to_log[f"{metrics_prefix}/con_acc"] = con_acc
            # if wandb.run:
            #     wandb.run.summary['con_acc'] = con_acc

        if "er" in self.metrics_list:
            if local_rank == 0:
                print("Computing ER accuracy...")
            er_acc = eval_erdos_renyi(networkx_digraphs, acyclic=self.acyclic)
            to_log[f"{metrics_prefix}/er_acc"] = er_acc
            # if wandb.run:
            #     wandb.run.summary['er_acc'] = er_acc

        if "ba" in self.metrics_list:
            if local_rank == 0:
                print("Computing BA accuracy...")
            ba_acc = eval_barabasi_albert(networkx_digraphs)
            to_log[f"{metrics_prefix}/ba_acc"] = ba_acc
            # if wandb.run:
            #     wandb.run.summary['ba_acc'] = ba_acc

        if "planar" in self.metrics_list:
            if local_rank == 0:
                print("Computing planar accuracy...")
            planar_acc = eval_acc_planar_graph(networkx_digraphs)
            to_log[f"{metrics_prefix}/planar_acc"] = planar_acc
            # if wandb.run:
            #     wandb.run.summary['planar_acc'] = planar_acc

        if "sbm" in self.metrics_list:
            if local_rank == 0:
                print("Computing SBM accuracy...")
            sbm_acc = eval_acc_sbm_graph(networkx_digraphs)
            to_log[f"{metrics_prefix}/sbm_acc"] = sbm_acc
            # if wandb.run:
            #     wandb.run.summary['sbm_acc'] = sbm_acc

        if "dag" in self.metrics_list:
            if local_rank == 0:
                print("Computing DAG accuracy...")
            dag_acc = eval_acc_directed_acyclic_graph(networkx_digraphs)
            to_log[f"{metrics_prefix}/dag_acc"] = dag_acc
            # if wandb.run:
            #     wandb.run.summary['dag_acc'] = dag_acc
            if "er" in self.metrics_list and self.acyclic:
                dag_er_acc = eval_acc_dag_and_erdos_renyi(networkx_digraphs)
                to_log[f"{metrics_prefix}/dag_er_acc"] = dag_er_acc
            elif "ba" in self.metrics_list and self.acyclic:
                dag_ba_acc = eval_acc_dag_and_barabasi_albert(networkx_digraphs)
                to_log[f"{metrics_prefix}/dag_ba_acc"] = dag_ba_acc

        if "scene_graph" in self.metrics_list:
            if local_rank == 0:
                print("Computing scene graph accuracy...")
            scene_graph_acc = eval_acc_scene_graph(
                networkx_digraphs, val_scene_graph_fn=self.is_scene_graph
            )
            to_log[f"{metrics_prefix}/scene_graph_acc"] = scene_graph_acc
            # if wandb.run:
            #     wandb.run.summary['scene_graph_acc'] = scene_graph_acc

        if "valid" in self.metrics_list:
            validity_dictionary = {
                "er": is_dag_and_erdos_renyi if self.acyclic else is_erdos_renyi,
                "ba": is_dag_and_barabasi_albert,
                "planar": is_planar_graph,
                "sbm": is_sbm_graph,
                "er_dag": nx.is_directed_acyclic_graph,
                "tpu_tile": nx.is_directed_acyclic_graph,
                "visual_genome": self.is_scene_graph,
            }
            validity_metric = validity_dictionary[self.graph_type]

            if local_rank == 0:
                print("Computing all fractions...")
            (
                frac_unique,
                frac_unique_non_isomorphic,
                frac_unique_non_isomorphic_valid,
                frac_non_unique_non_validated,
                frac_isomorphic_non_validated,
            ) = time_eval_fraction_unique_non_isomorphic_valid(
                networkx_digraphs, self.train_digraphs, validity_func=validity_metric
            )
            frac_isomorphic, frac_isomorphic_non_validated2 = (
                time_eval_fraction_isomorphic(networkx_digraphs, self.train_digraphs)
            )
            frac_non_isomorphic = 1.0 - frac_isomorphic
            to_log.update(
                {
                    f"{metrics_prefix}/frac_unique": frac_unique,
                    f"{metrics_prefix}/frac_unique_non_iso": frac_unique_non_isomorphic,
                    f"{metrics_prefix}/frac_unic_non_iso_valid": frac_unique_non_isomorphic_valid,
                    f"{metrics_prefix}/frac_non_iso": frac_non_isomorphic,
                    f"{metrics_prefix}/frac_non_unique_non_validated": frac_non_unique_non_validated,
                    f"{metrics_prefix}/frac_isomorphic_non_validated": frac_isomorphic_non_validated,
                    f"{metrics_prefix}/frac_isomorphic_non_validated2": frac_isomorphic_non_validated2,
                }
            )

        ratios = compute_ratios(
            gen_metrics=to_log,
            ref_metrics=ref_metrics["test"] if test else ref_metrics["val"],
            metrics_keys=[
                f"{metrics_prefix}/out_degree",
                f"{metrics_prefix}/in_degree",
                f"{metrics_prefix}/clustering",
                f"{metrics_prefix}/spectre",
                f"{metrics_prefix}/wavelet",
            ],
        )
        to_log.update(ratios)

        if local_rank == 0:
            print("Sampling statistics", to_log)
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log

    def reset(self):
        pass


# Override loader so that the node labels are kept
def node_attributed_loader_to_nx(loader, directed=False):
    networkx_graphs = []
    for i, batch in enumerate(loader):
        data_list = batch.to_data_list()
        for j, data in enumerate(data_list):
            if directed:
                labels = data.x.argmax(dim=1)
                new_nx_graph = to_networkx(
                    data,
                    node_attrs=None,
                    edge_attrs=None,
                    to_undirected=False,
                    remove_self_loops=True,
                )
                # add label to the nodes
                for i, node in enumerate(new_nx_graph.nodes()):
                    new_nx_graph.nodes[i]["label"] = labels[i].item()
                networkx_graphs.append(new_nx_graph)
            else:
                raise ValueError("Graphs must be directed, please set directed=True")

    return networkx_graphs


class TPUSamplingMetrics(DirectedSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            acyclic=True,
            metrics_list=[
                "in_degree",
                "out_degree",
                "clustering",
                "spectre",
                "wavelet",
                "connected",
                "dag",
                "valid",
                "unique",
            ],
            graph_type="tpu_tile",
            compute_emd=False, 
        )

    # Override loader so that the node labels are kept
    def loader_to_nx(self, loader, directed=False):
        return node_attributed_loader_to_nx(loader, directed=directed)


class VisualGenomeSamplingMetrics(DirectedSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            acyclic=False,
            metrics_list=[
                "in_degree",
                "out_degree",
                "clustering",
                "spectre",
                "wavelet",
                "valid",
                "unique",
                "scene_graph",
            ],
            graph_type="visual_genome",
            compute_emd=False, 
        )
        self.num_objects = datamodule.cfg.dataset.num_objects
        self.num_relationships = datamodule.cfg.dataset.num_relationships
        self.num_attributes = datamodule.cfg.dataset.num_attributes

    # Override loader so that the node labels are kept
    def loader_to_nx(self, loader, directed=False):
        return node_attributed_loader_to_nx(loader, directed=directed)

    def is_object_node(self, node_label):
        return 0 <= node_label < self.num_objects

    def is_relationship_node(self, node_label):
        return (
            self.num_objects <= node_label < self.num_objects + self.num_relationships
        )

    def is_attribute_node(self, node_label):
        return (
            self.num_objects + self.num_relationships
            <= node_label
            < self.num_objects + self.num_relationships + self.num_attributes
        )

    def is_scene_graph(self, G):
        for node_idx, node in enumerate(G.nodes()):

            # Object node
            if self.is_object_node(G.nodes[node]["label"]):
                out_edges = list(G.out_edges(node))
                in_edges = list(G.in_edges(node))
                for _, target in out_edges:
                    if self.is_object_node(G.nodes[target]["label"]):
                        return False
                for source, _ in in_edges:
                    if not self.is_relationship_node(G.nodes[source]["label"]):
                        return False
                # print("Passed object node check")

            # Relationship node
            elif self.is_relationship_node(G.nodes[node]["label"]):
                out_edges = list(G.out_edges(node))
                in_edges = list(G.in_edges(node))
                if len(out_edges) != 1:
                    return False
                if len(in_edges) != 1:
                    return False
                source, _ = in_edges[0]
                _, target = out_edges[0]
                if not self.is_object_node(
                    G.nodes[target]["label"]
                ) or not self.is_object_node(G.nodes[source]["label"]):
                    return False
                # print("Passed relationship node check")

            # Attribute node
            elif self.is_attribute_node(G.nodes[node]["label"]):
                out_edges = list(G.out_edges(node))
                in_edges = list(G.in_edges(node))
                if len(out_edges) > 0:
                    return False
                if len(in_edges) != 1:
                    return False
                source, _ = in_edges[0]
                if not self.is_object_node(G.nodes[source]["label"]):
                    return False
                # print("Passed attribute node check")

            # Error
            else:
                raise ValueError(
                    f"Node {node} has an invalid label: {G.nodes[node]['label']}"
                )

        return True


class SyntheticSamplingMetrics(DirectedSamplingMetrics):
    def __init__(self, datamodule, acyclic, metrics_list, graph_type):
        super().__init__(
            datamodule=datamodule,
            acyclic=acyclic,
            metrics_list=metrics_list,
            graph_type=graph_type,
            compute_emd=False, 
        )
