# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
#from .single_stage import SingleStageDetector, SingleStageDetector_TRT, ONNX_EXPORTER, SingleStageDetector_Depth_TRT, SingleStageDetector_RGBD_TRT
from .single_stage import *
import importlib.util

@DETECTORS.register_module()
class YOLOF(SingleStageDetector):
    r"""Implementation of `You Only Look One-level Feature
    <https://arxiv.org/abs/2103.09460>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOF, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)

if importlib.util.find_spec("tensorrt") is not None:
    @DETECTORS.register_module()
    class YOLOF_TRT(SingleStageDetector_TRT):
        r"""Implementation of `You Only Look One-level Feature
        <https://arxiv.org/abs/2103.09460>`_"""

        def __init__(self,
                     backbone,
                     neck,
                     bbox_head,
                     train_cfg=None,
                     test_cfg=None,
                     pretrained=None):
            super(YOLOF_TRT, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)

    @DETECTORS.register_module()
    class YOLOF_Depth_TRT(SingleStageDetector_Depth_TRT):
        r"""Implementation of `You Only Look One-level Feature
        <https://arxiv.org/abs/2103.09460>`_"""

        def __init__(self,
                     backbone,
                     neck,
                     bbox_head,
                     train_cfg=None,
                     test_cfg=None,
                     pretrained=None):
            super(YOLOF_Depth_TRT, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)

    @DETECTORS.register_module()
    class YOLOF_RGBD_TRT(SingleStageDetector_RGBD_TRT):
        r"""Implementation of `You Only Look One-level Feature
        <https://arxiv.org/abs/2103.09460>`_"""

        def __init__(self,
                     backbone,
                     neck,
                     bbox_head,
                     train_cfg=None,
                     test_cfg=None,
                     pretrained=None):
            super(YOLOF_RGBD_TRT, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)



@DETECTORS.register_module()
class YOLOF_ONNX(ONNX_EXPORTER):
    r"""Implementation of `You Only Look One-level Feature
    <https://arxiv.org/abs/2103.09460>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOF_ONNX, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)
