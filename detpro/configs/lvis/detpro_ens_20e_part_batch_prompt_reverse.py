_base_ = 'mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_pretrain_ens.py'
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.000025)
optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.000025)
# optimizer = dict(type='SGD', lr=0.2e-6, momentum=0.9, weight_decay=2.5e-5)
lr_config = dict(step=[16])
total_epochs = 1
evaluation = dict(interval=1, metric=['bbox', 'segm'])
# only segmentation part
# evaluation = dict(interval=1, metric=['bbox'])
checkpoint_config = dict(interval=1, create_symlink=False)
data = dict(
    # samples_per_gpu=3,
    # workers_per_gpu=4,
    samples_per_gpu=3,
    workers_per_gpu=4,
    test=dict(
        # samples_per_gpu=3,
        samples_per_gpu=3,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadProposals', num_max_proposals=None),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='ImageToTensor', keys=['img',
                                                    'img_no_normalize']),
                    # dict(type='ToTensor', keys=['proposals']),
                    # dict(
                    #     type='ToDataContainer',
                    #     fields=[dict(key='proposals', stack=False)]),
                    dict(
                        type='Collect',
                        keys=['img', 'img_no_normalize', 'proposals'])
                ])
        ])
)

model = dict(
    freeze_backbone=True,
    freeze_fpn=True,
    freeze_neck=True,
    roi_head=dict(
        # mask_head=None,
        train_kd_loss=True,
        prompt_prob_train=True,
        prompt_prob_test=False,
        softmax_mode=False,
        prompt_group_list=[ "complete", "incomplete",
                            # 'clear', 'blur',
                            "unoccluded", "occluded",
                            "unobscured", "obscured",
                            "recognizable", "unrecognizable"],
        # prompt_group_list=[ "complete", "incomplete",
        #                     # "blur", "clear",
        #                     "occluded", "unoccluded",
        #                     "obscured", "unobscured",
        #                     "recognizable", "unrecognizable"],
        prompt_group_mean_way=True,
        prompt_prob_level_calc=True,
        level_low=0.4,
        level_high=0.7,
        debug=False,
        use_part_layer_to_extract_features=False,
        bbox_roi_extractor=dict(
            # only used to test
            det_map_to_lvls=(0, 1),
            cls_map_to_lvls=(2, 3),
        )
    )
)
load_from = '/home/clark/workspace2/detpro/workdirs/vild_ens_20e_fg_bg_5_10_end/epoch_18.pth'
# resume_from = '/home/clark/workspace2/detpro/workdirs/vild_use_complete_to_train_without_complete_test_low_lr/epoch_1.pth'