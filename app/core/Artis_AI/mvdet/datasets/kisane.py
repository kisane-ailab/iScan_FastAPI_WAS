# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
import mmcv
from mmcv.utils import print_log
from mmdet.core import eval_map, eval_recalls
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class KisaneDataset(CocoDataset):
    CLASSES = ('object',)
    PALETTE = [(106, 0, 228)]