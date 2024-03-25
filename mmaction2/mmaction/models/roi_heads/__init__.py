# Copyright (c) OpenMMLab. All rights reserved.
try:
    from mmdet.registry import MODELS as MMDET_MODELS

    from .bbox_heads import BBoxHeadAVA, CLIPHeadAVA, CLIPVHeadAVA
    from .roi_extractors import SingleRoIExtractor3D
    from .roi_head import AVARoIHead
    from .roi_head_clipv import AVARoICLIPVHead
    from .shared_heads import ACRNHead, FBOHead, LFBInferHead, ACARHead

    for module in [
            AVARoIHead, BBoxHeadAVA, SingleRoIExtractor3D, ACRNHead, FBOHead,
            LFBInferHead, ACARHead, CLIPHeadAVA, AVARoICLIPVHead, CLIPVHeadAVA
    ]:

        MMDET_MODELS.register_module()(module)

    __all__ = [
        'AVARoIHead', 'BBoxHeadAVA', 'SingleRoIExtractor3D', 'ACRNHead',
        'FBOHead', 'LFBInferHead', 'ACARHead', 'CLIPHeadAVA', 'AVARoICLIPVHead', 'CLIPVHeadAVA'
    ]

except (ImportError, ModuleNotFoundError):
    pass
