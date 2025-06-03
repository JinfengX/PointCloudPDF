from functools import partial
from joblib import Parallel, delayed

import torch
import numpy as np
import torch_points_kernels as tp
from torch import nn
from sklearn.mixture import GaussianMixture

# from torch_sparse import
from .utils import (
    MST,
    z_score_filter_np,
    z_score_mask_np,
    distance_similarity,
    confidence_similarity,
)
from ..builder import RECOGNIZER
from pointcept.models.utils.misc import offset2batch
from pointcept.utils.visualization import save_point_cloud
from pointcept.models.builder import MODELS, build_model
from pointcept.models.losses.builder import build_criteria
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components


@RECOGNIZER.register_module("PointPdf-v1m1")
class PointPdfV1(nn.Module):
    def __init__(
        self,
        recognizer,
        criteria,
        loss_weight,
        step_loss_weight: bool,
        num_classes,
        start_epoch,
        kp_ball_radius,
        kp_max_neighbor,
        condition_from,
        beta,
        seed_from,
        seed_range,
        num_seed,
        slide_window=False,
        adaptive_radius=False,
        softmax_score=True,
    ):
        super().__init__()
        self.need_input = True
        self.init_disable_update = True
        self.start_epoch = start_epoch
        self.runtime_update = True
        self.num_classes = num_classes
        self.alpha = loss_weight
        self.step_loss_weight = step_loss_weight
        self.kp_ball_radius = kp_ball_radius
        self.kp_max_neighbor = kp_max_neighbor
        self.recognizer = build_model(recognizer)
        self.criteria = build_criteria(criteria)

        self.condition_from = condition_from
        self.beta = beta
        self.seed_from = seed_from
        self.seed_range = seed_range
        self.num_seed = num_seed
        self.slide_window = slide_window
        self.adaptive_radius = adaptive_radius
        self.softmax_score = softmax_score

        self.parallel_processing = Parallel(n_jobs=4)

    def forward(self, input_dict):
        seg_logits = self.model_hooks["backbone"]["forward_output"]
        self.trigger_operation()
        score = self.recognizer(self.model_hooks)
        # train
        if self.training:
            if self.epoch < self.start_epoch:
                return dict(score=score)
            pseudo_mask = self.get_pseudo_mask(
                input_dict["coord"],
                seg_logits,
                input_dict["offset"],
            )

            # visualize open-set label
            # batch = offset2batch(input_dict["offset"])
            # color = torch.ones(len(input_dict["coord"]), 3) * 0.7
            # color[
            #     torch.isin(input_dict["segment_oracle"], torch.tensor([4, 7, 14, 16]).cuda()),
            #     0,
            # ] = 1  # [4, 7, 14, 16] for scannet, [5, 9] for s3dis
            # for i, o in enumerate(input_dict["offset"]):
            #     save_point_cloud(
            #         input_dict["coord"][batch == i],
            #         color[batch == i],
            #         f".tmp/visualizations/{i}_unknown.ply",
            #     )

            segment_pseudo = input_dict["segment"].clone()
            segment_pseudo[pseudo_mask] = self.num_classes
            loss = (
                self.criteria(torch.cat([seg_logits, score], -1), segment_pseudo)
                * self.alpha
            )
            if self.softmax_score:
                score = torch.cat([seg_logits, score], -1).softmax(-1)[:, -1]
            return dict(score=score, loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            if self.softmax_score:
                score = torch.cat([seg_logits, score], -1).softmax(-1)[:, -1]
            return dict(score=score)
        # test
        else:
            return dict(seg_logits=seg_logits)

    def get_pseudo_mask(self, coord, seg_logits, offset, neighbors=None):
        with torch.no_grad():
            batch = offset2batch(offset)
            if neighbors is None and not self.adaptive_radius:
                neighbors = tp.ball_query(
                    self.kp_ball_radius,
                    self.kp_max_neighbor,
                    coord,
                    coord,
                    mode="partial_dense",
                    batch_x=batch,
                    batch_y=batch,
                )[0]
            bt_coords = []
            bt_seg_logits = []
            bt_neighbors = []
            for i, o in enumerate(offset):
                bt_mask = batch == i
                bt_coord = coord[bt_mask]
                if self.adaptive_radius:
                    search_radius = (
                        (bt_coord.max(0)[0] - bt_coord.min(0)[0] + 1e-6) / 16
                    ).min()
                    bt_nn = tp.ball_query(
                        search_radius,
                        self.kp_max_neighbor,
                        bt_coord,
                        bt_coord,
                        mode="partial_dense",
                        batch_x=torch.zeros(size=(bt_coord.shape[0],)).cuda().long(),
                        batch_y=torch.zeros(size=(bt_coord.shape[0],)).cuda().long(),
                    )[0]
                    bt_neighbors.append(bt_nn)
                else:
                    if i == 0:
                        bt_neighbors.append(neighbors[batch == i])
                    else:
                        bt_nn = neighbors[batch == i]
                        bt_nn[bt_nn != -1] -= offset[i - 1]
                        bt_neighbors.append(bt_nn)
                bt_coords.append(bt_coord)
                bt_seg_logits.append(seg_logits[bt_mask])

            pseudo_mask = self.parallel_processing(
                delayed(self.pseudo_labeling)(
                    bt_coords[bt],
                    bt_seg_logits[bt],
                    bt_neighbors[bt],
                    self.condition_from,
                    self.beta,
                    self.seed_from,
                    self.seed_range,
                    self.num_seed,
                    self.slide_window,
                )
                for bt in range(batch.max() + 1)
            )

            # visualize pseudo label
            # for i, mask in enumerate(pseudo_mask):
            #     color = torch.ones(len(bt_coords[i]), 3) * 0.7
            #     color[mask, 0] = 1
            #     save_point_cloud(
            #         bt_coords[i], color, f".tmp/visualizations/{i}_pseudo_label.ply"
            #     )

            pseudo_mask = torch.cat(pseudo_mask)
        return pseudo_mask

    @staticmethod
    def pseudo_labeling(
        bt_coord,
        bt_output,
        bt_neighbors,
        condition_from,
        beta,
        seed_from,
        seed_range,
        num_seed,
        slide_window,
    ):
        def get_condition(score, beta=beta):
            mean = torch.mean(score)
            std = torch.std(score)
            stop_condition = mean - beta * std
            return stop_condition

        def get_seed(score, seed_range=seed_range, num_seed=num_seed):
            dice = torch.randint(0, int(seed_range * len(score)), [num_seed])
            sort_score_idx = torch.sort(score, dim=-1)[1]
            seed_idx = sort_score_idx[dice]
            return seed_idx

        with torch.no_grad():
            bt_out_msp = torch.softmax(bt_output, dim=-1).max(dim=-1)[0]
            bt_out_ml = bt_output.max(dim=-1)[0]
            bt_out_ml = (bt_out_ml - bt_out_ml.min()) / (
                bt_out_ml.max() - bt_out_ml.min() + 1e-6
            )
            if condition_from == "msp":
                stop_condition = get_condition(bt_out_msp)
                bt_out_score = bt_out_msp
            elif condition_from == "ml":
                stop_condition = get_condition(bt_out_ml)
                bt_out_score = bt_out_ml

            if seed_from == "msp":
                graph_idx = get_seed(bt_out_msp)
            elif seed_from == "ml":
                graph_idx = get_seed(bt_out_ml)

            # heuristic search unknown area
            while True:
                graph_coord = bt_coord[graph_idx]
                graph_score = bt_out_score[graph_idx]
                if (
                    graph_score.mean(0) > stop_condition
                    and len(graph_idx) > 0.01 * len(bt_coord)
                    and len(graph_idx) > 50
                ):
                    break
                graph_nn_idx = bt_neighbors[graph_idx]
                graph_nn_idx = torch.unique(graph_nn_idx)
                graph_nn_idx = graph_nn_idx[
                    (graph_nn_idx != -1).logical_and(
                        ~torch.isin(graph_nn_idx, graph_idx)
                    )
                ]
                # # resample if not enough neighbors
                # if graph_nn_idx.shape[0] < num_seed:
                #     cnt = 0
                #     while True:
                #         if graph_nn_idx.shape[0] > num_seed:
                #             break
                #         dice = torch.randint(
                #             0,
                #             min(
                #                 int((seed_range + 0.01 * cnt) * len(bt_coord)),
                #                 len(bt_coord),
                #             ),
                #             [num_seed + cnt * int(0.01 * len(bt_coord))],
                #         )
                #         graph_nn_idx = bt_sort_ml_idx[dice]
                #         graph_nn_idx = graph_nn_idx[
                #             (graph_nn_idx != -1).logical_and(
                #                 ~torch.isin(graph_nn_idx, graph_idx)
                #             )
                #         ]
                #         cnt += 1
                # Mean shift
                dist_sim = torch.norm(
                    bt_coord[graph_nn_idx] - graph_coord.mean(0), dim=-1
                )
                dist_sim = 1 - (dist_sim - dist_sim.min()) / (
                    dist_sim.max() - dist_sim.min() + 1e-3
                )
                # sliding window
                if slide_window:
                    cut_off_s = torch.kthvalue(
                        graph_score, int(len(graph_score) * 0.1)
                    ).values
                    cut_off_e = torch.kthvalue(
                        graph_score, int(len(graph_score) * 0.6)
                    ).values
                else:
                    cut_off_s = graph_score.min()
                    cut_off_e = graph_score.max()
                # Mean shift
                conf_sim = torch.exp(
                    -torch.abs(
                        bt_out_score[graph_nn_idx]
                        - graph_score[
                            (graph_score >= cut_off_s) & (graph_score <= cut_off_e)
                        ].mean(0)
                    )
                )
                similarity = 0.4 * dist_sim + 0.6 * conf_sim
                select_sim_idx = torch.topk(
                    similarity.view(-1), k=int(similarity.numel() * 0.4)
                )[1]
                selected_nn = graph_nn_idx.view(-1)[select_sim_idx]
                new_graph_idx = torch.cat([graph_idx, selected_nn])
                new_graph_idx = torch.unique(new_graph_idx)
                new_graph_idx = new_graph_idx[new_graph_idx != -1]
                if new_graph_idx.shape[0] == graph_idx.shape[0]:
                    break
                graph_idx = new_graph_idx

            # accept_node_idx = graph_idx

            # graph boundary detection
            node = graph_idx
            node_nn = bt_neighbors[node]
            dist_sim = distance_similarity(node, node_nn, bt_coord)
            conf_sim = confidence_similarity(node, node_nn, bt_out_msp)
            similarity = 0.4 * dist_sim + 0.6 * conf_sim

            # use sparse matrix for graph construction
            num_cord = len(bt_coord)
            valid_mask = node_nn != -1
            in_graph_mask = torch.isin(node_nn, node)
            self_mask = node_nn == node[:, None]
            mask_flatten = (valid_mask & in_graph_mask & (~self_mask)).flatten()
            adj_matrix = csr_matrix(
                (
                    similarity.view(-1)[mask_flatten].cpu(),
                    (
                        node.repeat_interleave(node_nn.shape[1])[mask_flatten].cpu(),
                        node_nn.view(-1)[mask_flatten].cpu(),
                    ),
                ),
                shape=(num_cord, num_cord),
            )
            mst_adj_matrix = minimum_spanning_tree(adj_matrix)

            # filter outlier edge
            mst_edge_weight = mst_adj_matrix.data
            gmm = GaussianMixture(n_components=2).fit(
                mst_edge_weight.reshape(-1, 1)
            )  # gaussian mixture model
            gmm_means = gmm.means_.flatten()
            gmm_cov = gmm.covariances_.flatten()
            gmm_idx = np.argmax(gmm_means)

            mst_edge_outlier_filter = ~z_score_filter_np(
                mst_edge_weight, gmm_means[gmm_idx], gmm_cov[gmm_idx], "left", 2.0
            )
            mst_adj_matrix.data[mst_edge_outlier_filter] = 0
            mst_adj_matrix.eliminate_zeros()

            # find outlier sub graph
            num_subgraph, node_graph_label = connected_components(
                mst_adj_matrix, directed=False
            )
            # unique_subgraph_label, subgraph_size = np.unique(
            #     node_graph_label, return_counts=True
            # )
            # single_node_subgraph_idx = np.where(subgraph_size == 1)[0]
            # single_node_subgraph_label = unique_subgraph_label[single_node_subgraph_idx]
            # single_node_subgraph_mask = np.isin(
            #     node_graph_label, single_node_subgraph_label
            # )
            # node_graph_label_filtered = node_graph_label[~single_node_subgraph_mask]
            ingraph_node_label = node_graph_label[
                torch.unique(torch.cat([node, node_nn.view(-1)]))[1:].cpu()
            ]
            unique_ingraph_node_label, filtered_subgraph_size = np.unique(
                ingraph_node_label, return_counts=True
            )
            filtered_subgraph_outlier_mask = z_score_mask_np(
                filtered_subgraph_size, area="right", score=2.0
            )
            accept_subgraph_label = unique_ingraph_node_label[
                filtered_subgraph_outlier_mask
            ]
            accept_node_idx = np.where(
                np.isin(node_graph_label, accept_subgraph_label)
            )[0]
            assert np.isin(accept_node_idx, node.cpu()).all()

            pseudo_mask = torch.zeros(len(bt_coord), dtype=torch.long)
            pseudo_mask[accept_node_idx] = 1
            pseudo_mask = pseudo_mask.bool()

        return pseudo_mask

    def trigger_operation(self):
        if self.init_disable_update:
            for name, param in self.recognizer.named_parameters():
                param.requires_grad = False
            self.init_disable_update = False

        if self.epoch >= self.start_epoch and self.runtime_update:
            torch.cuda.empty_cache()
            for name, param in self.recognizer.named_parameters():
                param.requires_grad = True
            self.runtime_update = False

        if self.epoch > self.start_epoch + 1 and self.step_loss_weight:
            self.alpha = self.alpha * 0.1
            self.step_loss_weight = False
