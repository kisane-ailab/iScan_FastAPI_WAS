# MLSYS
TRT_PATH = ""
TRT_PATH_DEPTH = ""

# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import gc
import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

import numpy as np
#import tensorrt as trt
#import pycuda.driver as cuda
#import pycuda.autoinit
import importlib.util
import threading

import logging
logging.basicConfig(level=logging.ERROR)

host_inputs = []
cuda_inputs = []
host_outputs = []
cuda_outputs = []
bindings = []

host_inputs_depth = []
cuda_inputs_depth = []
host_outputs_depth = []
cuda_outputs_depth = []
bindings_depth = []

@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels

if importlib.util.find_spec("tensorrt") is not None:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    @DETECTORS.register_module()
    class SingleStageDetector_TRT(SingleStageDetector):
        """Base class for single-stage detectors using TensorRT for inference.

        Single-stage detectors directly and densely predict bounding boxes on the
        output features of the backbone+neck.

        This class utilizes TensorRT for efficient inference.
        """
        def __init__(self,
                     backbone,
                     neck=None,
                     bbox_head=None,
                     train_cfg=None,
                     test_cfg=None,
                     pretrained=None,
                     init_cfg=None):
            super(SingleStageDetector_TRT, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
            cuda.init()
            self.cuda_context = cuda.Device(0).make_context()
            self.load_and_prepare(TRT_PATH)

        def load_and_prepare(self, engine_path):
            self.engine = self._load_engine(engine_path)
            self.context = self._prepare_buffers()

        def _load_engine(self, engine_file_path):
                """
                Load a TensorRT engine from a file.

                Args:
                    engine_file_path (str): Path to the TensorRT engine file.

                Returns:
                    trt.ICudaEngine: Loaded TensorRT engine.
                """
                TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
                runtime = trt.Runtime(TRT_LOGGER)
                with open(engine_file_path, "rb") as f:
                    engine_data = f.read()
                engine = runtime.deserialize_cuda_engine(engine_data)
                return engine

        def _prepare_buffers(self):
            """
            Prepare the buffers for input and output data and create an execution context.

            Returns:
                trt.IExecutionContext: Execution context for the TensorRT engine.
            """
            for binding in self.engine:
                size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                host_mem = cuda.pagelocked_empty(size, dtype)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings.append(int(cuda_mem))
                if self.engine.binding_is_input(binding):
                    host_inputs.append(host_mem)
                    cuda_inputs.append(cuda_mem)
                else:
                    host_outputs.append(host_mem)
                    cuda_outputs.append(cuda_mem)

            return self.engine.create_execution_context()

        def execute_infer(self, input_data):
            """
            Execute inference on the input data.

            Args:
                input_data (numpy.ndarray): Input data for inference.

            Returns:
                numpy.ndarray: Output data after inference.
            NOTE:
                When you export a new ONNX file, make sure to check the model's output shape before running inference.
                You may need to update the following code accordingly:

                                                                        ↓↓↓↓↓
                reshaped_output = np.reshape(host_outputs[0], (1, 512, 23, 41))
                                                                        ↑↑↑↑↑
            """
            self.cuda_context.push()
            try:
                np.copyto(host_inputs[0], input_data.ravel())
                cuda.memcpy_htod(cuda_inputs[0], host_inputs[0])
                self.context.execute_v2(bindings)
                cuda.memcpy_dtoh(host_outputs[0], cuda_outputs[0])
                ###reshaped_output = np.reshape(host_outputs[0], (1, 512, 23, 41))
                ###reshaped_output = np.reshape(host_outputs[0], (1, 512, 26, 38))
                reshaped_output = np.reshape(host_outputs[0], (1, 512, 30, 40))
                reshaped_output = np.expand_dims(reshaped_output, axis=0)
                return reshaped_output
            finally:
                self.cuda_context.pop()

        def infer(self, input_data):
            """
            This method was modified to use a separate thread to avoid errors
            that occur when not specifying a thread for inference.

            Args:
                input_data (numpy.ndarray): Input data for inference.

            Returns:
                numpy.ndarray: Inference result.

            """
            result = [None]
            def target():
                result[0] = self.execute_infer(input_data)
            infer_thread = threading.Thread(target=target)
            infer_thread.start()
            infer_thread.join()
            return result[0]

        def extract_feat(self, img):
            """
            Directly extract features from the backbone+neck using the TensorRT engine for inference.

            Args:
                img (torch.Tensor): Input image.

            Returns:
                torch.Tensor: Extracted features.

            Note:
                This function performs inference using the TensorRT engine to extract features from the input image.
            """
            rt_img = img.cpu()
            rt_img = np.array(rt_img)
            rt_img = rt_img.astype(np.float16)
            x = self.infer(rt_img)
            x = torch.from_numpy(x).to(device = "cuda", dtype= torch.float16)
            return x

        def __del__(self):
            self.cuda_context.pop()

    @DETECTORS.register_module()
    class SingleStageDetector_Depth_TRT(SingleStageDetector):
        """Base class for single-stage detectors using TensorRT for inference.

        Single-stage detectors directly and densely predict bounding boxes on the
        output features of the backbone+neck.

        This class utilizes TensorRT for efficient inference.
        """

        def __init__(self,
                     backbone,
                     neck=None,
                     bbox_head=None,
                     train_cfg=None,
                     test_cfg=None,
                     pretrained=None,
                     init_cfg=None):
            super(SingleStageDetector_Depth_TRT, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                          test_cfg, pretrained)
            cuda.init()
            self.cuda_context = cuda.Device(0).make_context()
            ###self.load_and_prepare(TRT_PATH)
            self.load_and_prepare(TRT_PATH_DEPTH)

        def load_and_prepare(self, engine_path):
            self.engine = self._load_engine(engine_path)
            self.context = self._prepare_buffers()

        def _load_engine(self, engine_file_path):
            """
            Load a TensorRT engine from a file.

            Args:
                engine_file_path (str): Path to the TensorRT engine file.

            Returns:
                trt.ICudaEngine: Loaded TensorRT engine.
            """
            TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(TRT_LOGGER)
            with open(engine_file_path, "rb") as f:
                engine_data = f.read()
            engine = runtime.deserialize_cuda_engine(engine_data)
            return engine

        def _prepare_buffers(self):
            """
            Prepare the buffers for input and output data and create an execution context.

            Returns:
                trt.IExecutionContext: Execution context for the TensorRT engine.
            """
            for binding_depth in self.engine:
                size = trt.volume(self.engine.get_binding_shape(binding_depth)) * self.engine.max_batch_size
                dtype = trt.nptype(self.engine.get_binding_dtype(binding_depth))
                host_mem = cuda.pagelocked_empty(size, dtype)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)
                ###bindings.append(int(cuda_mem))
                bindings_depth.append(int(cuda_mem))
                if self.engine.binding_is_input(binding_depth):
                    ###host_inputs.append(host_mem)
                    ###cuda_inputs.append(cuda_mem)
                    host_inputs_depth.append(host_mem)
                    cuda_inputs_depth.append(cuda_mem)
                else:
                    ###host_outputs.append(host_mem)
                    ###cuda_outputs.append(cuda_mem)
                    host_outputs_depth.append(host_mem)
                    cuda_outputs_depth.append(cuda_mem)

            return self.engine.create_execution_context()

        def execute_infer(self, input_data):
            """
            Execute inference on the input data.

            Args:
                input_data (numpy.ndarray): Input data for inference.

            Returns:
                numpy.ndarray: Output data after inference.
            NOTE:
                When you export a new ONNX file, make sure to check the model's output shape before running inference.
                You may need to update the following code accordingly:

                                                                        ↓↓↓↓↓
                reshaped_output = np.reshape(host_outputs[0], (1, 512, 23, 41))
                                                                        ↑↑↑↑↑
            """
            self.cuda_context.push()
            try:
                ###np.copyto(host_inputs[0], input_data.ravel())
                np.copyto(host_inputs_depth[0], input_data.ravel())
                ###cuda.memcpy_htod(cuda_inputs[0], host_inputs[0])
                cuda.memcpy_htod(cuda_inputs_depth[0], host_inputs_depth[0])
                ###self.context.execute_v2(bindings)
                self.context.execute_v2(bindings_depth)
                ###cuda.memcpy_dtoh(host_outputs[0], cuda_outputs[0])
                cuda.memcpy_dtoh(host_outputs_depth[0], cuda_outputs_depth[0])
                ###reshaped_output = np.reshape(host_outputs[0], (1, 512, 23, 41))
                ###reshaped_output = np.reshape(host_outputs_depth[0], (1, 512, 13, 19))
                reshaped_output = np.reshape(host_outputs_depth[0], (1, 512, 15, 20))
                reshaped_output = np.expand_dims(reshaped_output, axis=0)
                return reshaped_output
            finally:
                self.cuda_context.pop()

        def infer(self, input_data):
            """
            This method was modified to use a separate thread to avoid errors
            that occur when not specifying a thread for inference.

            Args:
                input_data (numpy.ndarray): Input data for inference.

            Returns:
                numpy.ndarray: Inference result.

            """
            result = [None]

            def target():
                result[0] = self.execute_infer(input_data)

            ###infer_thread = threading.Thread(target=target)
            ###infer_thread.start()
            ###infer_thread.join()
            depth_infer_thread = threading.Thread(target=target)
            depth_infer_thread.start()
            depth_infer_thread.join()
            return result[0]

        def extract_feat(self, img):
            """
            Directly extract features from the backbone+neck using the TensorRT engine for inference.

            Args:
                img (torch.Tensor): Input image.

            Returns:
                torch.Tensor: Extracted features.

            Note:
                This function performs inference using the TensorRT engine to extract features from the input image.
            """
            rt_img = img.cpu()
            rt_img = np.array(rt_img)
            rt_img = rt_img.astype(np.float16)
            x = self.infer(rt_img)
            x = torch.from_numpy(x).to(device="cuda", dtype=torch.float16)
            return x

        def __del__(self):
            self.cuda_context.pop()

    @DETECTORS.register_module()
    class SingleStageDetector_RGBD_TRT(SingleStageDetector):
        """Base class for single-stage detectors using TensorRT for inference.

        Single-stage detectors directly and densely predict bounding boxes on the
        output features of the backbone+neck.

        This class utilizes TensorRT for efficient inference.
        """

        def __init__(self,
                     backbone,
                     neck=None,
                     bbox_head=None,
                     train_cfg=None,
                     test_cfg=None,
                     pretrained=None,
                     init_cfg=None):
            super(SingleStageDetector_RGBD_TRT, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                          test_cfg, pretrained)
            cuda.init()
            self.cuda_context = cuda.Device(0).make_context()
            ###self.load_and_prepare(TRT_PATH)
            self.load_and_prepare(TRT_PATH_DEPTH)

        def load_and_prepare(self, engine_path):
            self.engine = self._load_engine(engine_path)
            self.context = self._prepare_buffers()

        def _load_engine(self, engine_file_path):
            """
            Load a TensorRT engine from a file.

            Args:
                engine_file_path (str): Path to the TensorRT engine file.

            Returns:
                trt.ICudaEngine: Loaded TensorRT engine.
            """
            TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(TRT_LOGGER)
            with open(engine_file_path, "rb") as f:
                engine_data = f.read()
            engine = runtime.deserialize_cuda_engine(engine_data)
            return engine

        def _prepare_buffers(self):
            """
            Prepare the buffers for input and output data and create an execution context.

            Returns:
                trt.IExecutionContext: Execution context for the TensorRT engine.
            """
            for binding_depth in self.engine:
                size = trt.volume(self.engine.get_binding_shape(binding_depth)) * self.engine.max_batch_size
                dtype = trt.nptype(self.engine.get_binding_dtype(binding_depth))
                host_mem = cuda.pagelocked_empty(size, dtype)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)
                ###bindings.append(int(cuda_mem))
                bindings_depth.append(int(cuda_mem))
                if self.engine.binding_is_input(binding_depth):
                    ###host_inputs.append(host_mem)
                    ###cuda_inputs.append(cuda_mem)
                    host_inputs_depth.append(host_mem)
                    cuda_inputs_depth.append(cuda_mem)
                else:
                    ###host_outputs.append(host_mem)
                    ###cuda_outputs.append(cuda_mem)
                    host_outputs_depth.append(host_mem)
                    cuda_outputs_depth.append(cuda_mem)

            return self.engine.create_execution_context()

        def execute_infer(self, input_data):
            """
            Execute inference on the input data.

            Args:
                input_data (numpy.ndarray): Input data for inference.

            Returns:
                numpy.ndarray: Output data after inference.
            NOTE:
                When you export a new ONNX file, make sure to check the model's output shape before running inference.
                You may need to update the following code accordingly:

                                                                        ↓↓↓↓↓
                reshaped_output = np.reshape(host_outputs[0], (1, 512, 23, 41))
                                                                        ↑↑↑↑↑
            """
            self.cuda_context.push()
            try:
                ###np.copyto(host_inputs[0], input_data.ravel())
                np.copyto(host_inputs_depth[0], input_data.ravel())
                ###cuda.memcpy_htod(cuda_inputs[0], host_inputs[0])
                cuda.memcpy_htod(cuda_inputs_depth[0], host_inputs_depth[0])
                ###self.context.execute_v2(bindings)
                self.context.execute_v2(bindings_depth)
                ###cuda.memcpy_dtoh(host_outputs[0], cuda_outputs[0])
                cuda.memcpy_dtoh(host_outputs_depth[0], cuda_outputs_depth[0])
                ###reshaped_output = np.reshape(host_outputs[0], (1, 512, 23, 41))
                ###reshaped_output = np.reshape(host_outputs_depth[0], (1, 512, 13, 19))
                reshaped_output = np.reshape(host_outputs_depth[0], (1, 512, 30, 40))
                reshaped_output = np.expand_dims(reshaped_output, axis=0)
                return reshaped_output
            finally:
                self.cuda_context.pop()

        def infer(self, input_data):
            """
            This method was modified to use a separate thread to avoid errors
            that occur when not specifying a thread for inference.

            Args:
                input_data (numpy.ndarray): Input data for inference.

            Returns:
                numpy.ndarray: Inference result.

            """
            result = [None]

            def target():
                result[0] = self.execute_infer(input_data)

            ###infer_thread = threading.Thread(target=target)
            ###infer_thread.start()
            ###infer_thread.join()
            depth_infer_thread = threading.Thread(target=target)
            depth_infer_thread.start()
            depth_infer_thread.join()
            return result[0]

        def extract_feat(self, img):
            """
            Directly extract features from the backbone+neck using the TensorRT engine for inference.

            Args:
                img (torch.Tensor): Input image.

            Returns:
                torch.Tensor: Extracted features.

            Note:
                This function performs inference using the TensorRT engine to extract features from the input image.
            """
            rt_img = img.cpu()
            rt_img = np.array(rt_img)
            rt_img = rt_img.astype(np.float16)
            x = self.infer(rt_img)
            x = torch.from_numpy(x).to(device="cuda", dtype=torch.float16)
            return x

        def __del__(self):
            self.cuda_context.pop()


@DETECTORS.register_module()
class ONNX_EXPORTER(SingleStageDetector):
    """
    This class is used to convert a model's backbone and neck to an ONNX file.
    It is not intended for actual inference or training.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(ONNX_EXPORTER, self).__init__(backbone, neck, bbox_head, train_cfg,
                                            test_cfg, pretrained)

    def convert_onnx(self, img):
        """
        Convert the combined backbone and neck model to ONNX format.

        This method exports the backbone and neck of the model to an ONNX file named 'backbone_neck.onnx'.
        It sets the model to evaluation mode and uses a dummy input tensor for the export.

        Note:
            When you export a new ONNX file, make sure to check the model's input shape.
                                            ↓↓↓↓↓
            dummy_input = torch.randn(1, 3, 720, 1296, dtype = torch.float16).cuda()
                                            ↑↑↑↑↑
        """
        # TODO: make debug code for checking tensor.shape
        combined_model = torch.nn.Sequential(self.backbone, self.neck)
        combined_model.eval().cuda()
        dummy_input = torch.randn(size=img.shape, dtype=torch.float16).cuda()

        torch.onnx.export(combined_model,
                          dummy_input,
                          "backbone_neck.onnx",
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          )

        output_shape = combined_model(dummy_input)[0].shape
        print("*" * 150)
        print(
            f"In single_stage.py, change reshaped_output to {output_shape} in the execute_infer function of the SingleStageDetector_TRT class.")
        print("*" * 150)

    def simple_test(self, img, img_metas, rescale=False):
        self.convert_onnx(img)
        print("onnx_file_converting")
        exit(0)