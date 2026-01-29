# Ultralytics YOLO 🚀, AGPL-3.0 license

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
import yaml

import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
    LoadVisualPrompt
)
from .base import BaseDataset
from .utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"

def create_class_index_map(old_class_names, new_class_names):
    """
    이전 클래스 인덱스를 새 클래스 인덱스로 매핑하는 딕셔너리를 생성합니다.

    Args:
        old_class_names (list): 이전 데이터셋 정의의 클래스 이름 리스트.
        new_class_names (list): 새 데이터셋 정의의 클래스 이름 리스트.

    Returns:
        (dict): 이전 인덱스를 새 인덱스로 매핑하는 딕셔너리.
    """
    new_name_to_idx = {name: i for i, name in enumerate(new_class_names)}

    old_idx_to_new_idx_map = {}
    for i, old_name in enumerate(old_class_names):
        if old_name in new_name_to_idx:
            old_idx_to_new_idx_map[i] = new_name_to_idx[old_name]
        # else: 클래스가 제거된 경우이므로 매핑에 추가하지 않습니다.

    return old_idx_to_new_idx_map


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", load_vp=False, **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        self.load_vp = load_vp
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        if 'kisan' in self.img_path:
            if 'train' in self.img_path or 'val' in self.img_path:
                json_ann_path = os.path.join(self.img_path, "/home/dataset/da_detection/01_data/train_val/annotations_cil_10.json")
                # class_info_path = os.path.join(self.img_path, "/home/dataset/da_detection/02_model/kisan/Yolof_Class_Info.json")
                # 클래스 이름 ➝ ID 매핑
                with open(json_ann_path, "r") as f:
                    class_info = json.load(f)
                class_name_to_id = {cat['name']: cat['id'] for cat in class_info['categories']}
                width, height = 1280, 960  # 고정 크기
                # 이미지 정보
                labels = []
                for im_file in self.im_files:
                    label_file = im_file.replace('.jpg', '.txt')
                    objects = []
                    with open(label_file, "r") as f:
                        for line in f:
                            parts = line.strip().split(',')
                            if len(parts) != 5:
                                continue
                            # if "Color" in im_file:
                            #     breakpoint()
                            keywords = ["sample", "bread", "beverage", "wrap"]
                            if any(k in im_file.split('/')[-3] for k in keywords):
                                x1, y1, x2, y2, class_name = parts
                                try:
                                    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
                                except ValueError:
                                    continue
                                if int(class_name) > 900:
                                    class_id = class_name_to_id.get(class_name, -1)
                                    if class_id == -1:
                                        continue
                                # (x1, y1, x2, y2) → (x_center, y_center, w, h)
                                w = x2 - x1
                                h = y2 - y1
                                x_center = x1 + w / 2
                                y_center = y1 + h / 2
                                # 정규화
                                x_center /= width
                                y_center /= height
                                w /= width
                                h /= height
                                if "object" in self.img_path:
                                    class_id = 1
                                objects.append((class_id, x_center, y_center, w, h))
                            else:
                                x, y, w, h, class_name = parts
                                x, y, w, h = map(float, (x, y, w, h))
                                if 'original' in im_file:
                                    class_name = im_file.split('/')[7]
                                if int(class_name) > 900:
                                    class_id = class_name_to_id.get(class_name, -1)
                                    if class_id == -1:
                                        continue
                                else:
                                    class_id = int(class_name)
                                x_center = (x + w / 2) / width
                                y_center = (y + h / 2) / height
                                w_norm = w / width
                                h_norm = h / height
                                if "object" in self.img_path:
                                    class_id = 1
                                objects.append((class_id, x_center, y_center, w_norm, h_norm))
                            
                    if objects:
                        cls = np.array([[obj[0]] for obj in objects], dtype=np.float32)
                        bboxes = np.array([[obj[1], obj[2], obj[3], obj[4]] for obj in objects], dtype=np.float32)
                    else:
                        cls = np.zeros((0, 1), dtype=np.float32)
                        bboxes = np.zeros((0, 4), dtype=np.float32)
                    labels.append({
                        "im_file": im_file,
                        "shape": (height, width),  # H, W
                        "cls": cls,
                        "bboxes": bboxes,
                        "segments": [], # 세그먼트가 없다면 빈 리스트
                        'bbox_format': 'xywh',
                        'normalized': True
                    })
                return labels
            else:
                ###category_path = os.path.join(self.img_path, "/home/dataset/da_detection/01_data/train_val/annotations_cil_10.json")
                #category_path = "/code/outsourcing_koreaUniv/datasets/kisan/annotations/annotations_cil_10.json"
                #category_path = "C:/Dev/outsourcing_koreaUniv/datasets/kisan/annotations/annotations_cil_10.json"
                category_path = "/home/nvidia/Kisan/outsourcing_koreaUniv/datasets/kisan/annotations/annotations_cil_10.json"
                # breakpoint()
                # 클래스 이름 ➝ ID 매핑
                with open(category_path, "r") as f:
                    category = json.load(f)
                class_name_to_id = {cat['name']: cat['id'] for cat in category['categories']}
                #json_ann_path = os.path.join(self.img_path, "/home/dataset/da_detection/01_data/train_val/annotations_test_cil_200.json")
                #with open(json_ann_path, "r") as f:
                #    class_info = json.load(f)
                img_lst = []
                width, height = 1280, 960  # 고정 이미지 크기
                #for img in class_info['images']:
                #    img_lst.append(img['file_name'])
                # 이미지 정보
                labels = []
                for im_file in self.im_files:
                    # if im_file in img_lst:
                    label_file = im_file.replace('.jpg', '.txt')
                    objects = []
                    with open(label_file, "r") as f:
                        for line in f:
                            parts = line.strip().split(',')
                            if len(parts) != 5:
                                continue
                            x1, y1, x2, y2, class_name = parts
                            try:
                                x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
                            except ValueError:
                                continue
                            class_id = class_name_to_id.get(class_name, -1)
                            if class_id == -1:
                                continue
                            # (x1, y1, x2, y2) → (x_center, y_center, w, h)
                            w = x2 - x1
                            h = y2 - y1
                            x_center = x1 + w / 2
                            y_center = y1 + h / 2
                            # 정규화
                            x_center /= width
                            y_center /= height
                            w /= width
                            h /= height
                            if "object" in self.img_path:
                                class_id = 1
                            objects.append((class_id, x_center, y_center, w, h))
                    if objects:
                        cls = np.array([[obj[0]] for obj in objects], dtype=np.float32)
                        bboxes = np.array([[obj[1], obj[2], obj[3], obj[4]] for obj in objects], dtype=np.float32)
                    else:
                        cls = np.zeros((0, 1), dtype=np.float32)
                        bboxes = np.zeros((0, 4), dtype=np.float32)
                    labels.append({
                        "im_file": im_file,
                        "shape": (height, width),  # (H, W)
                        "cls": cls,
                        "bboxes": bboxes,
                        "segments": [],  # 세그먼트 없음
                        'bbox_format': 'xywh',
                        'normalized': True
                    })
                return labels
            
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        if self.load_vp:
            if not self.augment:
                assert(self.batch_size == 1)
                nc = len(self.data["names"])
            else:
                nc = 80
            transforms.append(LoadVisualPrompt(nc=nc, augment=self.augment))
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k == "texts":
                value = torch.stack(value, 0)
            if k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes a dataset object for object detection tasks with optional specifications."""
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label):
        """Add texts information for multi-modal model training."""
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        if not self.single_cls:
            labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        
        return labels

    def build_transforms(self, hyp=None):
        """Enhances data transformations with optional text augmentation for multi-modal training."""
        transforms = super().build_transforms(hyp)
        if self.augment and not self.single_cls:
            # NOTE: hard-coded the args for now.
            index = -2 if self.load_vp else -1
            transforms.insert(index, RandomLoadText(text_model=hyp.text_model, max_samples=min(self.data["nc"], 80), padding=True))
        return transforms

from ultralytics.utils.ops import xyxy2xywhn

class GroundingDataset(YOLODataset):
    """Handles object detection tasks by loading annotations from a specified JSON file, supporting YOLO format."""

    def __init__(self, *args, task="detect", json_file, **kwargs):
        """Initializes a GroundingDataset for object detection, loading annotations from a specified JSON file."""
        assert task == "detect" or task == "segment", "`GroundingDataset` only support `detect` task for now!"
        self.json_file = json_file
        super().__init__(*args, task=task, data={}, **kwargs)

    def get_img_files(self, img_path):
        """The image files would be read in `get_labels` function, return empty list here."""
        return []

    def verify_labels(self, labels):
        instance_count = 0
        for label in labels:
            instance_count += label["bboxes"].shape[0]
        
        if "final_mixed_train_no_coco_segm" in self.json_file:
            assert(instance_count == 3662344)
        elif "final_mixed_train_no_coco" in self.json_file:
            assert(instance_count == 3681235)
        elif "final_flickr_separateGT_train_segm" in self.json_file:
            assert(instance_count == 638214)
        elif "final_flickr_separateGT_train" in self.json_file:
            assert(instance_count == 640704)
        elif "new" in self.json_file:
            pass
        else:
            assert(False)
    
    def get_labels(self):
        """Loads annotations from a JSON file, filters, and normalizes bounding boxes for each image."""
        cache_path = Path(self.json_file).with_suffix('.cache')
        labels = np.load(str(cache_path), allow_pickle=True)
        self.verify_labels(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}") 
        return labels
    
    def build_transforms(self, hyp=None):
        """Configures augmentations for training with optional text loading; `hyp` adjusts augmentation intensity."""
        transforms = super().build_transforms(hyp)
        if self.augment and not self.single_cls:
            # NOTE: hard-coded the args for now.
            index = -2 if self.load_vp else -1
            transforms.insert(index, RandomLoadText(text_model=hyp.text_model, max_samples=80, padding=True))
        return transforms


class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.
    """

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        return YOLODataset.collate_fn(batch)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """
    Semantic Segmentation Dataset.

    This class is responsible for handling datasets used for semantic segmentation tasks. It inherits functionalities
    from the BaseDataset class.

    Note:
        This class is currently a placeholder and needs to be populated with methods and attributes for supporting
        semantic segmentation tasks.
    """

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class ClassificationDataset:
    """
    Extends torchvision ImageFolder to support YOLO classification tasks, offering functionalities like image
    augmentation, caching, and verification. It's designed to efficiently handle large datasets for training deep
    learning models, with optional image transformations and caching mechanisms to speed up training.

    This class allows for augmentations using both torchvision and Albumentations libraries, and supports caching images
    in RAM or on disk to reduce IO overhead during training. Additionally, it implements a robust verification process
    to ensure data integrity and consistency.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
    """

    def __init__(self, root, args, augment=False, prefix=""):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings. It includes attributes like `imgsz` (image size), `fraction` (fraction
                of data to use), `scale`, `fliplr`, `flipud`, `cache` (disk or RAM caching for faster training),
                `auto_augment`, `hsv_h`, `hsv_s`, `hsv_v`, and `crop_fraction`.
            augment (bool, optional): Whether to apply augmentations to the dataset. Default is False.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification and
                debugging. Default is an empty string.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        if TORCHVISION_0_18:  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "WARNING ⚠️ Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz, crop_fraction=args.crop_fraction)
        )

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self):
        """Verify all images in dataset."""
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples
