_base_ = [
    'ours_slowfast_r50.py'
]

model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(temporal_pool_mode='max'),
        bbox_head=dict(focal_alpha=3.0, focal_gamma=1.0)))
