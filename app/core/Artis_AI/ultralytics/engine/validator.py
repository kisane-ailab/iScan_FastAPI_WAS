# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Check a model's accuracy on a test or val split of a dataset.

Usage:
    $ yolo mode=val model=yolov8n.pt data=coco8.yaml imgsz=640

Usage - formats:
    $ yolo mode=val model=yolov8n.pt                 # PyTorch
                          yolov8n.torchscript        # TorchScript
                          yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolov8n_openvino_model     # OpenVINO
                          yolov8n.engine             # TensorRT
                          yolov8n.mlpackage          # CoreML (macOS-only)
                          yolov8n_saved_model        # TensorFlow SavedModel
                          yolov8n.pb                 # TensorFlow GraphDef
                          yolov8n.tflite             # TensorFlow Lite
                          yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolov8n_paddle_model       # PaddlePaddle
                          yolov8n.mnn                # MNN
                          yolov8n_ncnn_model         # NCNN
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.utils import ops

import os
import cv2
from collections import Counter


# MSP
def draw_debug_image(img_org_path, img_save_path, batch, preds):
    img = cv2.imread(img_org_path)
    img_h, img_w, img_c = img.shape
    img_save = np.empty((img_h*2, img_w, img_c), dtype=img.dtype)
    im_size = 640
    print(f"=====> img = {img.shape},  img_save = {img_save.shape}")
    
    img_gt = img.copy()
    img_pred = img.copy()

    gt_classes = batch['cls']
    gt_bboxes = batch['bboxes']
    for gt_cls, gt_bbox in zip(gt_classes, gt_bboxes):
        print(f"=====> gt_bbox = {gt_bbox}")
        cx_f, cy_f, w_f, h_f = gt_bbox
        cx_i, cy_i = int(cx_f * img_w), int(cy_f * img_h)
        w_i, h_i = int(w_f * img_w), int(h_f * img_h)
        print(f"====== ===== =====> [gt] (cx, cy, w, h) = ({cx_i}, {cy_i}, {w_i}, {h_i})")
        tlx, brx = int(cx_i - (w_i//2)), int(cx_i + (w_i//2))
        tly, bry = int(cy_i - (h_i//2)), int(cy_i + (h_i//2))
        tlx, tly = max(0, tlx), max(0, tly)
        brx, bry = min(img_w, brx), min(img_h, bry)
        print(f"====== ===== =====> [gt] (tlx, tly, brx, bry) = ({tlx}, {tly}, {brx}, {bry})")
        img_gt = cv2.rectangle(img_gt, (tlx, tly), (brx, bry), (0, 0, 255), 3)


    for pred in preds:
        cx_f, cy_f, w_f, h_f, cls, conf = pred
        cx_i, cy_i = int(cx_f * img_w), int(cy_f * img_h)
        w_i, h_i = int(w_f * img_w), int(h_f * img_h)
        tlx, brx = int(cx_i - w_i//2), int(cx_i + w_i//2)
        tly, bry = int(cy_i - h_i//2), int(cy_i + h_i//2)
        tlx, tly = max(0, tlx), max(0, tly)
        brx, bry = min(img_w, brx), min(img_h, bry)
        img_pred = cv2.rectangle(img_pred, (tlx, tly), (brx, bry), (255, 0, 0), 3)

    img_save[:img_h, :, :] = img_gt.copy()
    img_save[img_h:, :, :] = img_pred.copy()
    cv2.imwrite(img_save_path, img_save)

    img_tmp_path = os.path.join("debug_folder", "test.jpg")
    cv2.imwrite(img_tmp_path, img_gt)
    #import sys
    #sys.exit(0)

    print(f"=====> gt_classes = {gt_classes}")
    print(f"=====> gt_bboxes = {gt_bboxes}")



class BaseValidator:
    """
    BaseValidator.

    A base class for creating validators.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names.
        seen: Records the number of images seen so far during validation.
        stats: Placeholder for statistics during validation.
        confusion_matrix: Placeholder for a confusion matrix.
        nc: Number of classes.
        iouv: (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (dict): Dictionary to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective
                      batch processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (dict): Dictionary to store various callback functions.
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Executes validation process, running inference on dataloader and computing performance metrics."""
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml"):
                LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        TP = []
        FP = []
        FN = []
        TP_cls = []
        FP_cls = []
        FN_cls = []
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)
                
            #tp, fp, fn, tp_cls, fp_cls, fn_cls = self.compute_tp_fp_fn(preds, batch, iou_thres=0.5)
            tp, fp, fn, tp_cls, fp_cls, fn_cls, _preds = self.compute_tp_fp_fn(preds, batch, iou_thres=0.5)       # MSP
            print(f"==========> [MSP] batch_i = {batch_i},  batch = {batch.keys()},  img = {batch['im_file']} {batch['img'].shape},  preds = {_preds},  tp = {tp},  fp = {fp},  fn = {fn},  tp_cls = {tp_cls},  fp_cls = {fp_cls},  fn_cls = {fn_cls}\n")

            # MSP : save debug image
            if (fp[0] > 0) or (fn[0] > 0):
                debug_folder = os.path.join("debug_folder", "wrong")
                os.makedirs(debug_folder, exist_ok=True)
            else:
                debug_folder = os.path.join("debug_folder", "correct")
                os.makedirs(debug_folder, exist_ok=True)
            _img_org_path = batch['im_file'][0]
            img_path = os.path.basename(_img_org_path)
            _img_save_path = os.path.join(debug_folder, img_path)
            draw_debug_image(_img_org_path, _img_save_path, batch, _preds)

            # Collect samples for visualization (both correct and error samples)
            if not hasattr(self, 'error_samples'):
                self.error_samples = []
            if not hasattr(self, 'correct_samples'):
                self.correct_samples = []
            if not hasattr(self, 'vanilla_newyork_roll_samples'):
                self.vanilla_newyork_roll_samples = []
            
            # Check for errors and correct predictions in this batch and collect samples
            for i, (pred, tp_count, fp_count, fn_count) in enumerate(zip(preds, tp, fp, fn)):
                sample_data = {
                    'batch_idx': batch_i,
                    'img_idx': i,
                    'batch': batch,
                    'preds': pred,
                    'tp': tp_count,
                    'fp': fp_count,
                    'fn': fn_count
                }
                
                # Collect error samples
                if (fp_count > 0 or fn_count > 0) and len(self.error_samples) < 10:  # Has errors
                    self.error_samples.append(sample_data)
                
                # Collect correct samples (no errors and has detections)
                elif (fp_count == 0 and fn_count == 0 and tp_count > 0) and len(self.correct_samples) < 10:
                    self.correct_samples.append(sample_data)
                
                # Collect vanilla_newyork_roll samples (check if vanilla_newyork_roll is in GT or predictions)
                if len(self.vanilla_newyork_roll_samples) < 10:
                    # Find vanilla_newyork_roll class ID
                    vanilla_newyork_roll_id = None
                    if hasattr(self, 'names'):
                        print(f"Debug: Looking for vanilla new york roll in {len(self.names)} classes")
                        for cls_id, name in self.names.items():
                            if 'vanilla' in name.lower() and 'new york roll' in name.lower():
                                vanilla_newyork_roll_id = cls_id
                                print(f"Found vanilla new york roll class: {cls_id} -> {name}")
                                break
                        
                        # If not found, try broader search
                        if vanilla_newyork_roll_id is None:
                            for cls_id, name in self.names.items():
                                if 'vanilla' in name.lower() or 'new york' in name.lower():
                                    print(f"Debug: Similar class found: {cls_id} -> {name}")
                        
                        if vanilla_newyork_roll_id is None:
                            print("Debug: No vanilla new york roll class found, collecting any sample for debugging")
                    
                    # Check if this image contains orange_pastry
                    contains_vanilla_newyork_roll = False
                    
                    # Check GT boxes for this image
                    batch_idx_mask = batch['batch_idx'] == i
                    gt_cls_in_img = batch['cls'][batch_idx_mask].cpu().numpy()
                    if vanilla_newyork_roll_id is not None and vanilla_newyork_roll_id in gt_cls_in_img:
                        contains_vanilla_newyork_roll = True
                    
                    # Check predictions for this image
                    if pred.numel() > 0:
                        pred_cls_in_img = pred[:, 5].cpu().numpy()
                        if vanilla_newyork_roll_id is not None and vanilla_newyork_roll_id in pred_cls_in_img:
                            contains_vanilla_newyork_roll = True
                    
                    # Collect sample if contains vanilla new york roll OR for debugging if class not found
                    if contains_vanilla_newyork_roll or (vanilla_newyork_roll_id is None and len(self.vanilla_newyork_roll_samples) < 3):
                        self.vanilla_newyork_roll_samples.append(sample_data)
                        if contains_vanilla_newyork_roll:
                            print(f"Debug: Collected sample with vanilla new york roll")
                        else:
                            print(f"Debug: Collected debug sample (no vanilla new york roll class found)")
            

            TP.extend(tp)
            FP.extend(fp)
            FN.extend(fn)
            TP_cls.extend(tp_cls)
            FP_cls.extend(fp_cls)
            FN_cls.extend(fn_cls)
            if batch_i % 100 == 0:
                tp_rate = sum(TP) / (sum(TP) + sum(FP) + sum(FN))
            
            
            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        tp_rate = sum(TP) / (sum(TP) + sum(FP) + sum(FN))
        print("tp_rate", tp_rate)
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        '''
        # BASE / NOVEL mAP 출력
        json_path = os.path.join(self.save_dir, "per_class_metrics.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        map50_list = [d.get("metrics/mAP50(B)", 0.0) for d in data]
        map_list = [d.get("metrics/mAP50-95(B)", 0.0) for d in data]
        base_map50 = map50_list[:190]
        base_map = map_list[:190]
        novel_map50 = map50_list[190:]
        novel_map = map_list[190:]
        base_map50_mean = sum(base_map50) / len(base_map50) if base_map50 else 0.0
        base_map_mean = sum(base_map) / len(base_map) if base_map else 0.0
        novel_map50_mean = sum(novel_map50) / len(novel_map50) if novel_map50 else 0.0
        novel_map_mean = sum(novel_map) / len(novel_map) if novel_map else 0.0
        LOGGER.info(f":막대_차트: Base classes (0–189): mAP50={base_map50_mean:.3f}, mAP50-95={base_map_mean:.3f}")
        LOGGER.info(f":막대_차트: Novel classes (190–199): mAP50={novel_map50_mean:.3f}, mAP50-95={novel_map_mean:.3f}")
        '''
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            print("tp_rate", tp_rate, ", TP: ", sum(TP), ", FP: ", sum(FP), ", FN: ", sum(FN))
            print("tp_cls_rate", sum(TP_cls) / (sum(TP_cls) + sum(FP_cls) + sum(FN_cls)))
            
            # Calculate class-wise accuracy: TP / (TP + FP + FN)
            TP_counter = Counter(TP_cls)
            FP_counter = Counter(FP_cls)
            FN_counter = Counter(FN_cls)
            
            # Get all unique classes that appear in TP, FP, or FN
            all_classes = set(TP_cls + FP_cls + FN_cls)
            class_accuracies = {}
            
            print("\nClass-wise Accuracy (TP / (TP + FP + FN)):")
            print("=" * 80)
            for cls_id in sorted(all_classes):
                tp_count = TP_counter.get(cls_id, 0)
                fp_count = FP_counter.get(cls_id, 0)
                fn_count = FN_counter.get(cls_id, 0)
                total = tp_count + fp_count + fn_count
                accuracy = tp_count / total if total > 0 else 0.0
                class_accuracies[cls_id] = accuracy
                
                # Get class name if available
                class_name = self.names.get(cls_id, f"Class_{cls_id}") if hasattr(self, 'names') else f"Class_{cls_id}"
                print(f"Class {cls_id:3d} ({class_name[:35]:35s}): {accuracy:6.3f} ({tp_count:3d}/{total:3d}) [TP:{tp_count:3d}, FP:{fp_count:3d}, FN:{fn_count:3d}]")
            
            # Show top and bottom performing classes
            sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\nTop 10 Best Performing Classes:")
            print("-" * 60)
            for i, (cls_id, acc) in enumerate(sorted_classes[:10]):
                class_name = self.names.get(cls_id, f"Class_{cls_id}") if hasattr(self, 'names') else f"Class_{cls_id}"
                tp_count = TP_counter.get(cls_id, 0)
                fp_count = FP_counter.get(cls_id, 0)
                fn_count = FN_counter.get(cls_id, 0)
                total = tp_count + fp_count + fn_count
                print(f"{i+1:2d}. Class {cls_id:3d} ({class_name[:30]:30s}): {acc:.3f} ({tp_count}/{total})")
                
            print(f"\nTop 10 Worst Performing Classes:")
            print("-" * 60)
            for i, (cls_id, acc) in enumerate(sorted_classes[-10:]):
                class_name = self.names.get(cls_id, f"Class_{cls_id}") if hasattr(self, 'names') else f"Class_{cls_id}"
                tp_count = TP_counter.get(cls_id, 0)
                fp_count = FP_counter.get(cls_id, 0)
                fn_count = FN_counter.get(cls_id, 0)
                total = tp_count + fp_count + fn_count
                print(f"{i+1:2d}. Class {cls_id:3d} ({class_name[:30]:30s}): {acc:.3f} ({tp_count}/{total})")
            
            tp_counts = [TP_counter[i] for i in range(self.nc)]
            fp_counts = [FP_counter[i] for i in range(self.nc)]
            fn_counts = [FN_counter[i] for i in range(self.nc)]
            print("\nTop FP classes:")
            print("-" * 40)
            top_fp_indices = np.argsort(fp_counts)[::-1][:10]
            for idx in top_fp_indices:
                if fp_counts[idx] > 0:
                    class_name = self.names.get(idx, f"Class_{idx}") if hasattr(self, 'names') else f"Class_{idx}"
                    print(f"Class {idx:3d} ({class_name[:30]:30s}): {fp_counts[idx]} times")
            
            print("\nTop FN classes:")
            print("-" * 40)
            top_fn_indices = np.argsort(fn_counts)[::-1][:10]
            for idx in top_fn_indices:
                if fn_counts[idx] > 0:
                    class_name = self.names.get(idx, f"Class_{idx}") if hasattr(self, 'names') else f"Class_{idx}"
                    print(f"Class {idx:3d} ({class_name[:30]:30s}): {fn_counts[idx]} times")
            
            return stats

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
    
    def filter_predictions(self, pred, iou_thres=0.2, conf_thres = 0.5):
        """
        Performs IoU-based NMS grouped by class.

        Args:
            pred (Tensor): (N, 6) → [x1, y1, x2, y2, conf, cls]

        Returns:
            filtered_pred (Tensor): (M, 6)
        """
        from ..utils.metrics import box_iou

        if pred.numel() == 0:
            return pred

        # Ensure everything is on CPU for consistency
        pred = pred.cpu()
        boxes = pred[:, :4]
        confs = pred[:, 4]
        classes = pred[:, 5]

        keep_indices = []

        for cls in classes.unique():
            
            cls_mask = classes == cls
            b = boxes[cls_mask]
            c = confs[cls_mask]
            cls_indices = cls_mask.nonzero(as_tuple=True)[0]

            if b.numel() == 0:
                continue
            if c[0] < conf_thres:
                continue
            # Sort by confidence
            order = c.argsort(descending=True)
            b = b[order]
            idxs = cls_indices[order]

            suppressed = torch.zeros(len(b), dtype=torch.bool)

            for i in range(len(b)):
                if suppressed[i]:
                    continue
                keep_indices.append(idxs[i])
                if i + 1 < len(b):
                    ious = box_iou(b[i].unsqueeze(0), b[i+1:]).squeeze(0).to(b.device)
                    suppressed[i+1:] |= ious >= iou_thres

        keep_indices = torch.stack(keep_indices) if keep_indices else torch.empty(0, dtype=torch.long)
        return pred[keep_indices]
    
    def compute_tp_fp_fn(self, preds, batch, iou_thres=0.3):
        """
        Compute TP, FP, FN per image in a batch, using IoU threshold and class match.
        Args:
            preds (List[Tensor]): List of length B (batch), each Tensor is (N, 6) → [x1, y1, x2, y2, conf, cls]
            batch (dict): Batch dict from dataloader, includes keys:
                - 'bboxes': (M, 4) GT boxes in [cx, cy, w, h] (normalized)
                - 'cls': (M,) GT class indices
                - 'ori_shape': (H, W)
                - 'batch_idx': (M,) batch index for each GT
            iou_thres (float): IoU threshold to consider a prediction a true positive
        Returns:
            List[int], List[int], List[int]: TP, FP, FN per image
        """
        from ..utils.metrics import box_iou
        import torch
        # Prepare GT boxes
        gt_boxes = batch['bboxes'].cpu().clone()  # (M, 4) in [cx, cy, w, h], normalized
        gt_cls = batch['cls'].cpu()               # (M,)
        batch_idx = batch['batch_idx'].cpu().long()  # (M,)
        h, w = batch['ori_shape'][0]  
        gt_boxes = gt_boxes.clone()
        TP, FP, FN = [], [], []
        TP_cls, FP_cls, FN_cls = [], [], []

        # MSP
        preds_list = list()

        for img_idx, pred in enumerate(preds):         # len(preds): B / preds[0]: [N_pred, 6]
            # Apply confidence threshold
            pred = pred[pred[:,4] > 0.3].cpu()         # objectness threshold
            # Apply NMS to remove duplicate detections (like the standard mAP calculation)
            if pred.numel() > 0:
                pred = self.filter_predictions(pred, iou_thres=0.0, conf_thres=0.3)
            
            pred_boxes = pred[:, :4]
            pred_cls = pred[:, 5]
            pred_confs = pred[:, 4]
            if pred.numel() == 0:
                TP.append(0)
                FP.append(0)
                FN.append((batch_idx == img_idx).sum().item())
                continue
            mask = batch_idx == img_idx
            boxes = gt_boxes[mask]                     # boxes: [N_gt,4]
            if len(boxes):
                if boxes[:, :4].max() <= 1.1:  # if normalized with tolerance 0.1
                    # Convert from normalized xywh to pixel xyxy (like standard mAP calculation)
                    boxes = ops.xywh2xyxy(boxes)
                    boxes[:, [0, 2]] *= w  # scale x coordinates
                    boxes[:, [1, 3]] *= h  # scale y coordinates
                else:
                    # Already in pixel coordinates, just convert to xyxy format
                    is_obb = boxes.shape[-1] == 5  # xywhr
                    boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
            gt_b = boxes
            gt_c = gt_cls[mask]                        # len(gt_c): N_gt
            if gt_b.numel() == 0:
                TP.append(0)
                FP.append(len(pred))
                FN.append(0)
                continue
            # Compute IoU
            # breakpoint()
            ious = box_iou(pred_boxes*2, gt_b)         # (N_pred, N_gt)
            matched_pred = set()
            matched_gt = set()
            full_gt = set([i for i in range(len(gt_b))])
            full_pred = set([i for i in range(len(pred))])
            # breakpoint()
            for pi in range(ious.shape[0]):
                for gi in range(ious.shape[1]):
                    if (pred_cls[pi] == gt_c[gi]):
                        if gi not in matched_gt and pi not in matched_pred:
                            matched_pred.add(pi)
                            matched_gt.add(gi)
            TP.append(len(matched_pred))
            FP.append(len(pred) - len(matched_pred))
            FN.append(len(gt_b) - len(matched_gt))
            unmatched_gt = full_gt - matched_gt
            unmatched_pred = full_pred - matched_pred
            tp_cls = gt_c[[i for i in matched_gt]]
            tp_cls = [int(i) for i in tp_cls]
            fp_cls = pred_cls[[i for i in unmatched_pred]]
            fp_cls = [int(i) for i in fp_cls]
            fn_cls = gt_c[[i for i in unmatched_gt]]
            fn_cls = [int(i) for i in fn_cls]
            TP_cls.extend(tp_cls)
            FP_cls.extend(fp_cls)
            FN_cls.extend(fn_cls)
        
        #return TP, FP, FN, TP_cls, FP_cls, FN_cls
        return TP, FP, FN, TP_cls, FP_cls, FN_cls, pred

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Runs all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def build_dataset(self, img_path):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        return batch

    def postprocess(self, preds):
        """Preprocesses the predictions."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        pass

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes and returns all metrics."""
        pass

    def get_stats(self):
        """Returns statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Checks statistics."""
        pass

    def print_results(self):
        """Prints the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Returns the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plots YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass
