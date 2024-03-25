ann_file_train = '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/chaos_train_1fps_final_version.csv'
ann_file_val = '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/chaos_test_1fps_final_version.csv'
anno_root = '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format'
auto_scale_lr = dict(base_batch_size=4, enable=True)
data_root = '/opt/data/share/106012/Chaotic_World/dataset/frames_320x180'
dataset_type = 'AVAKineticsDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=2, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
exclude_file_train = '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/chaos_train_exclude_final_version.csv'
exclude_file_val = '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/chaos_test_exclude_final_version.csv'
label_file = '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/list_action.pbtxt'
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    _scope_='mmdet',
    backbone=dict(
        depth=12,
        drop_path_rate=0.2,
        embed_dims=768,
        img_size=224,
        mlp_ratio=4,
        norm_cfg=dict(eps=1e-06, type='LN'),
        num_frames=16,
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        return_feat_map=True,
        type='mmaction.VisionTransformer_CLIP',
        use_mean_pooling=False),
    data_preprocessor=dict(
        _scope_='mmaction',
        format_shape='NCTHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    init_cfg=dict(
        checkpoint=
        'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth',
        type='Pretrained'),
    roi_head=dict(
        bbox_head=dict(
            background_class=True,
            clip_model='/opt/data/private/Chaotic_World/clip/ViT-B-16.pt',
            dropout_ratio=0.5,
            extention=
            '/opt/data/private/Chaotic_World/AR_ava_format/label_extend.json',
            in_channels=768,
            label=
            '/opt/data/private/Chaotic_World/AR_ava_format/list_action.pbtxt',
            multilabel=True,
            num_classes=51,
            type='CLIPVHeadAVA'),
        bbox_roi_extractor=dict(
            output_size=8,
            roi_layer_type='RoIAlign',
            type='SingleRoIExtractor3D',
            with_temporal_pool=True),
        type='AVARoICLIPVHead'),
    test_cfg=dict(rcnn=None),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                min_pos_iou=0.9,
                neg_iou_thr=0.9,
                pos_iou_thr=0.9,
                type='MaxIoUAssignerAVA'),
            pos_weight=1.0,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=32,
                pos_fraction=1,
                type='RandomSampler'))),
    type='FastRCNN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    constructor='LearningRateDecayOptimizerConstructor',
    optimizer=dict(lr=0.000125, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        decay_rate=0.75, decay_type='layer_wise', num_layers=12))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.1,
        type='LinearLR'),
    dict(
        T_max=15,
        begin=5,
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        eta_min=0,
        type='CosineAnnealingLR'),
]
proposal_file_train = '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/bbox_train_1fps.pkl'
proposal_file_val = '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/bbox_test_1fps.pkl'
randomness = dict(deterministic=True, diff_rank_seed=False, seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/chaos_test_1fps_final_version.csv',
        data_prefix=dict(
            img='/opt/data/share/106012/Chaotic_World/dataset/frames_320x180'),
        exclude_file=
        '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/chaos_test_exclude_final_version.csv',
        filename_tmpl='_{:06}.png',
        fps=1,
        label_file=
        '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/list_action.pbtxt',
        num_classes=51,
        num_max_proposals=1000,
        pipeline=[
            dict(
                clip_len=16,
                frame_interval=4,
                test_mode=True,
                type='SampleAVAFrames'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(collapse=True, input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        proposal_file=
        '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/bbox_test_1fps.pkl',
        test_mode=True,
        timestamp_end=999999,
        timestamp_start=0,
        type='AVAKineticsDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/chaos_test_1fps_final_version.csv',
    exclude_file=
    '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/chaos_test_exclude_final_version.csv',
    label_file=
    '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/list_action.pbtxt',
    num_classes=51,
    type='AVAMetric')
train_cfg = dict(
    max_epochs=20, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/chaos_train_1fps_final_version.csv',
        data_prefix=dict(
            img='/opt/data/share/106012/Chaotic_World/dataset/frames_320x180'),
        exclude_file=
        '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/chaos_train_exclude_final_version.csv',
        filename_tmpl='_{:06}.png',
        fps=1,
        label_file=
        '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/list_action.pbtxt',
        num_classes=51,
        num_max_proposals=1000,
        pipeline=[
            dict(clip_len=16, frame_interval=4, type='SampleAVAFrames'),
            dict(type='RawFrameDecode'),
            dict(scale_range=(
                256,
                320,
            ), type='RandomRescale'),
            dict(size=256, type='RandomCrop'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(collapse=True, input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        proposal_file=
        '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/bbox_train_1fps.pkl',
        timestamp_end=999999,
        timestamp_start=0,
        type='AVAKineticsDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(clip_len=16, frame_interval=4, type='SampleAVAFrames'),
    dict(type='RawFrameDecode'),
    dict(scale_range=(
        256,
        320,
    ), type='RandomRescale'),
    dict(size=256, type='RandomCrop'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(collapse=True, input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
url = 'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth'
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/chaos_test_1fps_final_version.csv',
        data_prefix=dict(
            img='/opt/data/share/106012/Chaotic_World/dataset/frames_320x180'),
        exclude_file=
        '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/chaos_test_exclude_final_version.csv',
        filename_tmpl='_{:06}.png',
        fps=1,
        label_file=
        '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/list_action.pbtxt',
        num_classes=51,
        num_max_proposals=1000,
        pipeline=[
            dict(
                clip_len=16,
                frame_interval=4,
                test_mode=True,
                type='SampleAVAFrames'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(collapse=True, input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        proposal_file=
        '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/bbox_test_1fps.pkl',
        test_mode=True,
        timestamp_end=999999,
        timestamp_start=0,
        type='AVAKineticsDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/chaos_test_1fps_final_version.csv',
    exclude_file=
    '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/chaos_test_exclude_final_version.csv',
    label_file=
    '/opt/data/share/106012/Chaotic_World/annotations/AR_ava_format/list_action.pbtxt',
    num_classes=51,
    type='AVAMetric')
val_pipeline = [
    dict(
        clip_len=16, frame_interval=4, test_mode=True, type='SampleAVAFrames'),
    dict(type='RawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(collapse=True, input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/videomae_clip_visual_decoder'
