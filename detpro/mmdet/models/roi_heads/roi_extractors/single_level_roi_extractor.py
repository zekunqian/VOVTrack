import torch
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor


@ROI_EXTRACTORS.register_module()
class SingleRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        det_map_to_lvls (list): using [0, 1,...] layer of features to make detetection
        cls_map_to_lvls (list): using [2, 3, ...] layer of features to make classification
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 det_map_to_lvls = None,
                 cls_map_to_lvls = None,
                 ):
        super(SingleRoIExtractor, self).__init__(roi_layer, out_channels,
                                                 featmap_strides)
        self.finest_scale = finest_scale
        self.det_map_to_lvls = det_map_to_lvls
        self.cls_map_to_lvls = cls_map_to_lvls

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None, det_cls_layer_tag=-1):
        """
        :param feats:
        :param rois:
        :param roi_scale_factor:
        :param det_cls_layer_tag:  0 means use detection layers, 1 means use classification layers, -1 mean no use
        :return:
        """
        """Forward function."""
        assert ((self.det_map_to_lvls is not None and det_cls_layer_tag == 0) or
                (self.cls_map_to_lvls is not None and det_cls_layer_tag == 1) or det_cls_layer_tag == -1), \
            f"Error detection and classification layer choosing!"

        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        if torch.onnx.is_in_onnx_export():
            # Work around to export mask-rcnn to onnx
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(
                -1, self.out_channels * out_size[0] * out_size[1])
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
        else:
            roi_feats = feats[0].new_zeros(
                rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)

        # divide the lvls to different group
        if det_cls_layer_tag == 0:
            group_num = len(self.det_map_to_lvls)
            group_size = len(self.det_map_to_lvls)
            index_list = list(range(num_levels))
            groups = [index_list[i * group_size: (i + 1) * group_size] for i in range(group_num)]

            remainder = len(index_list) % group_num
            if remainder:
                groups[-1] += index_list[-remainder:]
            index_map = {}
            for i, group in enumerate(groups):
                for g_index in group:
                    index_map[g_index] = self.det_map_to_lvls[i]
            # mapping to target lvls
            layer_mask = [target_lvls == ori_index for ori_index in index_map.keys()]
            for i, (ori_index, dst_index) in enumerate(index_map.items()):
                target_lvls[layer_mask[i]] = dst_index
        elif det_cls_layer_tag == 1:
            group_num = len(self.cls_map_to_lvls)
            group_size = len(self.cls_map_to_lvls)
            index_list = list(range(num_levels))
            groups = [index_list[i * group_size: (i + 1) * group_size] for i in range(group_num)]

            remainder = len(index_list) % group_num
            if remainder:
                groups[-1] += index_list[-remainder:]
            index_map = {}
            for i, group in enumerate(groups):
                for g_index in group:
                    index_map[g_index] = self.cls_map_to_lvls[i]
            # mapping to target lvls
            layer_mask = [target_lvls == ori_index for ori_index in index_map.keys()]
            for i, (ori_index, dst_index) in enumerate(index_map.items()):
                target_lvls[layer_mask[i]] = dst_index


        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            # TODO: make it nicer when exporting to onnx
            if torch.onnx.is_in_onnx_export():
                # To keep all roi_align nodes exported to onnx
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
                continue
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats
