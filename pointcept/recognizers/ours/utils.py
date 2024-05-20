import torch
import networkx as nx
from networkx.algorithms import minimum_spanning_tree
import numpy as np


def distance_similarity(node, node_nn, coord):
    device = coord.device
    valid_mask = node_nn != -1
    self_mask = node_nn == node[:, None]
    dist_sim = torch.norm(coord[node_nn] - coord[node, None], dim=-1)
    dist_min = torch.where(
        valid_mask & ~self_mask,
        dist_sim,
        torch.tensor(0).to(device),
    ).min(
        -1
    )[0][:, None]
    dist_max = torch.where(
        valid_mask & ~self_mask,
        dist_sim,
        torch.tensor(0).to(device),
    ).max(
        -1
    )[0][:, None]
    dist_sim = torch.where(
        valid_mask & ~self_mask,
        1 - (dist_sim - dist_min) / (dist_max - dist_min + 1e-3),
        torch.tensor([-10]).to(device),
    )
    return dist_sim


def confidence_similarity(node, node_nn, score):
    device = score.device
    valid_mask = node_nn != -1
    self_mask = node_nn == node[:, None]
    conf_sim = torch.where(
        valid_mask & ~self_mask,
        torch.exp(-1 * torch.abs(score[node_nn] - score[node, None])),
        torch.tensor(-10).to(device),
    )
    return conf_sim


def MST(node, edge):
    if torch.is_tensor(node):
        node = node.tolist()
    if torch.is_tensor(edge):
        edge = edge.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(node)
    graph.add_weighted_edges_from(edge)
    mst = minimum_spanning_tree(graph)
    return mst


def z_score_mask(x, mean=None, std=None, area="left", score=3.0):
    if not torch.is_tensor(x):
        x = torch.tensor(x).float()
    mean = torch.mean(x) if mean is None else mean
    std = torch.std(x) if std is None else std
    if area == "left":
        data_score = (mean - x) / std
    elif area == "right":
        data_score = (x - mean) / std
    elif area == "both":
        data_score = torch.abs((x - mean) / std)
    else:
        raise ValueError("area must be left, right or both")
    return data_score > score


def z_score_mask_torch(x:torch.Tensor, mean=None, std=None, area="left", score=3.0):
    mean = torch.mean(x) if mean is None else mean
    std = torch.std(x) if std is None else std
    if area == "left":
        data_score = (mean - x) / std
    elif area == "right":
        data_score = (x - mean) / std
    elif area == "both":
        data_score = torch.abs((x - mean) / std)
    else:
        raise ValueError("area must be left, right or both")
    return data_score > score

def z_score_mask_np(x:np.ndarray, mean=None, std=None, area="left", score=3.0):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    if area == "left":
        data_score = (mean - x) / std
    elif area == "right":
        data_score = (x - mean) / std
    elif area == "both":
        data_score = np.abs((x - mean) / std)
    else:
        raise ValueError("area must be left, right or both")
    return data_score > score

def z_score_filter(x, mean=None, std=None, area="left", score=3.0):
    if not torch.is_tensor(x):
        x = torch.tensor(x).float()
    mean = torch.mean(x) if mean is None else mean
    std = torch.std(x) if std is None else std
    if area == "left":
        lower_bound = mean - score * std
        return x < lower_bound
    elif area == "right":
        upper_bound = mean + score * std
        return x > upper_bound
    elif area == "both":
        lower_bound = mean - score * std
        upper_bound = mean + score * std
        return (x < lower_bound) | (x > upper_bound)
    else:
        raise ValueError("area must be left, right or both")
    
def z_score_filter_np(x: np.ndarray, mean=None, std=None, area="left", score=3.0):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    if area == "left":
        lower_bound = mean - score * std
        return x < lower_bound
    elif area == "right":
        upper_bound = mean + score * std
        return x > upper_bound
    elif area == "both":
        lower_bound = mean - score * std
        upper_bound = mean + score * std
        return (x < lower_bound) | (x > upper_bound)
    else:
        raise ValueError("area must be left, right or both")