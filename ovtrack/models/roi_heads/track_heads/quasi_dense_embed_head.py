import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models import HEADS, build_loss

from ovtrack.core import cal_similarity
import torch.nn.functional as F

@HEADS.register_module(force=True)
class QuasiDenseEmbedHead(nn.Module):
    def __init__(
        self,
        num_convs=4,
        num_fcs=1,
        roi_feat_size=7,
        in_channels=256,
        conv_out_channels=256,
        fc_out_channels=1024,
        embed_channels=256,
        conv_cfg=None,
        norm_cfg=None,
        softmax_temp=-1,
        loss_track=dict(type="MultiPosCrossEntropyLoss", loss_weight=0.25),
        loss_track_aux=dict(
            type="L2Loss", sample_ratio=3, margin=0.3, loss_weight=1.0, hard_mining=True
        ),
        # loss_cyc=dict(
        #         type='CycleLoss',
        #         margin=0.5,
        #         loss_type=['pairwise', 'triplewise']
        # )
        loss_cyc=None, # self train loss
        cluster_num=-1

    ):
        super(QuasiDenseEmbedHead, self).__init__()
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.embed_channels = embed_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.relu = nn.ReLU(inplace=True)
        self.convs, self.fcs, last_layer_dim = self._add_conv_fc_branch(
            self.num_convs, self.num_fcs, self.in_channels
        )
        self.fc_embed = nn.Linear(last_layer_dim, embed_channels)

        self.softmax_temp = softmax_temp
        self.loss_track = build_loss(loss_track)
        if loss_cyc is not None:
            self.loss_cyc = build_loss(loss_cyc)

        if loss_track_aux is not None:
            self.loss_track_aux = build_loss(loss_track_aux)
        else:
            self.loss_track_aux = None

        self.cluster_num = cluster_num

    def _add_conv_fc_branch(self, num_convs, num_fcs, in_channels):
        last_layer_dim = in_channels
        # add branch specific conv layers
        convs = nn.ModuleList()
        if num_convs > 0:
            for i in range(num_convs):
                conv_in_channels = last_layer_dim if i == 0 else self.conv_out_channels
                convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                    )
                )
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        fcs = nn.ModuleList()
        if num_fcs > 0:
            last_layer_dim *= self.roi_feat_size * self.roi_feat_size
            for i in range(num_fcs):
                fc_in_channels = last_layer_dim if i == 0 else self.fc_out_channels
                fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return convs, fcs, last_layer_dim

    def init_weights(self):
        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fc_embed.weight, 0, 0.01)
        nn.init.constant_(self.fc_embed.bias, 0)

    def forward(self, x):
        if self.num_convs > 0:
            for i, conv in enumerate(self.convs):
                x = conv(x)
        x = x.view(x.size(0), -1)
        if self.num_fcs > 0:
            for i, fc in enumerate(self.fcs):
                x = self.relu(fc(x))
        x = self.fc_embed(x)
        return x

    def get_track_targets(
        self, gt_match_indices, key_sampling_results, ref_sampling_results
    ):
        track_targets = []
        track_weights = []
        for _gt_match_indices, key_res, ref_res in zip(
            gt_match_indices, key_sampling_results, ref_sampling_results
        ):
            targets = _gt_match_indices.new_zeros(
                (key_res.pos_bboxes.size(0), ref_res.bboxes.size(0)), dtype=torch.int
            )
            _match_indices = _gt_match_indices[key_res.pos_assigned_gt_inds]
            pos2pos = (
                _match_indices.view(-1, 1) == ref_res.pos_assigned_gt_inds.view(1, -1)
            ).int()
            targets[:, : pos2pos.size(1)] = pos2pos
            weights = (targets.sum(dim=1) > 0).float()
            track_targets.append(targets)
            track_weights.append(weights)
        return track_targets, track_weights

    def match(self, key_embeds, ref_embeds, key_sampling_results, ref_sampling_results):
        num_key_rois = [res.pos_bboxes.size(0) for res in key_sampling_results]
        key_embeds = torch.split(key_embeds, num_key_rois)
        num_ref_rois = [res.bboxes.size(0) for res in ref_sampling_results]
        ref_embeds = torch.split(ref_embeds, num_ref_rois)

        dists, cos_dists = [], []
        for key_embed, ref_embed in zip(key_embeds, ref_embeds):
            dist = cal_similarity(
                key_embed,
                ref_embed,
                method="dot_product",
                temperature=self.softmax_temp,
            )
            dists.append(dist)
            if self.loss_track_aux is not None:
                cos_dist = cal_similarity(key_embed, ref_embed, method="cosine")
                cos_dists.append(cos_dist)
            else:
                cos_dists.append(None)
        return dists, cos_dists

    def loss(self, dists, cos_dists, targets, weights):
        losses = dict()

        loss_track = 0.0
        loss_track_aux = 0.0
        for _dists, _cos_dists, _targets, _weights in zip(
            dists, cos_dists, targets, weights
        ):
            loss_track += self.loss_track(
                _dists, _targets, _weights, avg_factor=_weights.sum()
            )
            if self.loss_track_aux is not None:
                loss_track_aux += self.loss_track_aux(_cos_dists, _targets)
        losses["loss_track"] = loss_track / len(dists)

        if self.loss_track_aux is not None:
            losses["loss_track_aux"] = loss_track_aux / len(dists)

        return losses

    def self_loss(self, features_list):
        losses = dict()
        cyc_loss = self.loss_cyc(features_list)
        losses['cyc_loss'] = cyc_loss
        return losses
    @staticmethod
    def filter_top_k_objects(xyxy_list, features_list, top_k):
        new_xyxy_list = []
        new_features_list = []

        for xyxy_tensor, features_tensor in zip(xyxy_list, features_list):
            # Check that tensor dimensions match
            assert xyxy_tensor.shape[0] == features_tensor.shape[0], "The first dimension of xyxy_tensor and features_tensor must match"

            # Sort by confidence score
            sorted_indices = torch.argsort(xyxy_tensor[:, 4], descending=True)

            # Take the top_k objects
            top_k_indices = sorted_indices[:top_k]

            # Keep top_k objects
            new_xyxy_tensor = xyxy_tensor[top_k_indices, :4]
            new_features_tensor = features_tensor[top_k_indices]

            # Append to new lists
            new_xyxy_list.append(new_xyxy_tensor)
            new_features_list.append(new_features_tensor)

        return new_xyxy_list, new_features_list

    def compute_iou(self,boxes1, boxes2):
        """
        Compute IoU between two sets of boxes.
        boxes1, boxes2: [N, 4], format (x1, y1, x2, y2)
        Return IoU matrix: [N, N]
        """
        # Intersection area
        x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

        inter_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        # Area of each box
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # IoU
        iou = inter_area / (area1[:, None] + area2 - inter_area)
        return iou

    def create_assignment_matrix(self, iou_matrix, iou_thresh):
        """
        Create assignment matrix based on IoU matrix and threshold
        """
        return (iou_matrix > iou_thresh).float()

    def calculate_similarity(self, features_tensor):
        """
        Compute cosine similarity matrix of features
        """
        similarity_matrix = F.cosine_similarity(features_tensor.unsqueeze(1), features_tensor.unsqueeze(0), dim=-1)
        # map the range to [0, 1]
        similarity_matrix = (similarity_matrix + 1) / 2
        return similarity_matrix

    def calculate_assignment_matrix_loss(self, assignment_matrix, similarity_matrix):
        """
        Compute loss by supervising similarity matrix with assignment matrix
        """
        loss = F.binary_cross_entropy(similarity_matrix, assignment_matrix)
        return loss

    def self_spatial_loss(self, features_list, xyxy_list, top_k=10, iou_thres=0.9, learning_rate=0.01):
        new_xyxy_list, new_features_list = self.filter_top_k_objects(xyxy_list, features_list, top_k)
        xyxy_tensor = torch.cat(new_xyxy_list, dim=0)
        features_tensor = torch.cat(new_features_list, dim=0)
        # Calculate IoU matrix
        iou_matrix = self.compute_iou(xyxy_tensor, xyxy_tensor)
        # Assignment matrix
        assignment_matrix = self.create_assignment_matrix(iou_matrix, iou_thres)
        # Calculate object similarity matrix
        similarity_matrix = self.calculate_similarity(features_tensor)
        # Calculate loss
        loss = self.calculate_assignment_matrix_loss(assignment_matrix, similarity_matrix)

        return learning_rate*loss



    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]

@HEADS.register_module(force=True)
class QuasiDenseEmbedHeadSharedConvSize(nn.Module):
    def __init__(
            self,
            num_convs=4,
            num_fcs=1,
            roi_feat_size=7,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            embed_channels=256,
            conv_cfg=None,
            norm_cfg=None,
            softmax_temp=-1,
            in_chhanel=256,
            loss_track=dict(type="MultiPosCrossEntropyLoss", loss_weight=0.25),
            loss_track_aux=dict(
                type="L2Loss", sample_ratio=3, margin=0.3, loss_weight=1.0, hard_mining=True
            ),
            # loss_cyc=dict(
            #         type='CycleLoss',
            #         margin=0.5,
            #         loss_type=['pairwise', 'triplewise']
            # )
            loss_cyc=None # self train loss
    ):
        super(QuasiDenseEmbedHeadSharedConvSize, self).__init__()
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.embed_channels = embed_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.relu = nn.ReLU(inplace=True)
        self.convs, self.fcs, last_layer_dim = self._add_conv_fc_branch(
            self.num_convs, self.num_fcs, self.in_channels
        )


        self.fc_embed = nn.Linear(last_layer_dim, embed_channels)

        self.softmax_temp = softmax_temp
        self.loss_track = build_loss(loss_track)
        if loss_cyc is not None:
            self.loss_cyc = build_loss(loss_cyc)

        if loss_track_aux is not None:
            self.loss_track_aux = build_loss(loss_track_aux)
        else:
            self.loss_track_aux = None

    def _add_conv_fc_branch(self, num_convs, num_fcs, in_channels):
        last_layer_dim = in_channels
        # add branch specific conv layers
        convs = nn.ModuleList()
        if num_convs > 0:
            for i in range(num_convs):
                conv_in_channels = last_layer_dim if i == 0 else self.conv_out_channels
                convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                    )
                )
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        fcs = nn.ModuleList()
        if num_fcs > 0:
            last_layer_dim *= self.roi_feat_size * self.roi_feat_size
            for i in range(num_fcs):
                fc_in_channels = last_layer_dim if i == 0 else self.fc_out_channels
                fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return convs, fcs, last_layer_dim

    def init_weights(self):
        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fc_embed.weight, 0, 0.01)
        nn.init.constant_(self.fc_embed.bias, 0)

    def forward(self, x):
        if self.num_convs > 0:
            for i, conv in enumerate(self.convs):
                x = conv(x)
        x = x.view(x.size(0), -1)
        if self.num_fcs > 0:
            for i, fc in enumerate(self.fcs):
                x = self.relu(fc(x))
        # x = self.fc_embed(x)
        return x

    def get_track_targets(
            self, gt_match_indices, key_sampling_results, ref_sampling_results
    ):
        track_targets = []
        track_weights = []
        for _gt_match_indices, key_res, ref_res in zip(
                gt_match_indices, key_sampling_results, ref_sampling_results
        ):
            targets = _gt_match_indices.new_zeros(
                (key_res.pos_bboxes.size(0), ref_res.bboxes.size(0)), dtype=torch.int
            )
            _match_indices = _gt_match_indices[key_res.pos_assigned_gt_inds]
            pos2pos = (
                    _match_indices.view(-1, 1) == ref_res.pos_assigned_gt_inds.view(1, -1)
            ).int()
            targets[:, : pos2pos.size(1)] = pos2pos
            weights = (targets.sum(dim=1) > 0).float()
            track_targets.append(targets)
            track_weights.append(weights)
        return track_targets, track_weights

    def match(self, key_embeds, ref_embeds, key_sampling_results, ref_sampling_results):
        num_key_rois = [res.pos_bboxes.size(0) for res in key_sampling_results]
        key_embeds = torch.split(key_embeds, num_key_rois)
        num_ref_rois = [res.bboxes.size(0) for res in ref_sampling_results]
        ref_embeds = torch.split(ref_embeds, num_ref_rois)

        dists, cos_dists = [], []
        for key_embed, ref_embed in zip(key_embeds, ref_embeds):
            dist = cal_similarity(
                key_embed,
                ref_embed,
                method="dot_product",
                temperature=self.softmax_temp,
            )
            dists.append(dist)
            if self.loss_track_aux is not None:
                cos_dist = cal_similarity(key_embed, ref_embed, method="cosine")
                cos_dists.append(cos_dist)
            else:
                cos_dists.append(None)
        return dists, cos_dists

    def loss(self, dists, cos_dists, targets, weights):
        losses = dict()

        loss_track = 0.0
        loss_track_aux = 0.0
        for _dists, _cos_dists, _targets, _weights in zip(
                dists, cos_dists, targets, weights
        ):
            loss_track += self.loss_track(
                _dists, _targets, _weights, avg_factor=_weights.sum()
            )
            if self.loss_track_aux is not None:
                loss_track_aux += self.loss_track_aux(_cos_dists, _targets)
        losses["loss_track"] = loss_track / len(dists)

        if self.loss_track_aux is not None:
            losses["loss_track_aux"] = loss_track_aux / len(dists)

        return losses

    def self_loss(self, features_list):
        losses = dict()
        cyc_loss = self.loss_cyc(features_list)
        losses['cyc_loss'] = cyc_loss
        return losses


    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]
