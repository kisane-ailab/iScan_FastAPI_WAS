import os, sys

from kisane import Kisane

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import cv2
import torch
import json
import time

from raft_core.raft_stereo import RAFTStereo
from raft_core.utils.utils import InputPadder

from xml.etree.ElementTree import ElementTree, Element, SubElement, tostring
import common_config as cc
from common_config import make_artis_ai_log, CurrentDateTime, CamInfo

import importlib.util

class Depth:
    def __init__(self, cal_file_path, depth_range, crop_args, config_file_path):
        if cal_file_path == None:
            self.is_depth_enable = False
            return
        else:
            self.is_depth_enable = True
            self.args = self.define_args()
            self.args.cal_config = cal_file_path
        
        self.timer_starter = torch.cuda.Event(enable_timing=True)
        self.timer_ender = torch.cuda.Event(enable_timing=True)

        self.load_model(cal_file_path, config_file_path)
        self.init_cam_params(depth_range, crop_args)
        print(f"{CurrentDateTime(0)} [Artis_AI][Load_models] Depth Model Parameters\n{self.args}")

    def load_model(self, cal_file_path, config_file_path):
        self.config_file_path = config_file_path
        if cal_file_path == None:
            self.is_depth_enable = False
            return
        else:
            self.is_depth_enable = True
            self.args = self.define_args()
            self.args.cal_config = cal_file_path

        self.args.restore_ckpt = cc.artis_ai_gen_depth_model_path
        print(f"{CurrentDateTime(0)} [Artis_AI] {self.args.restore_ckpt} exists : {os.path.exists(self.args.restore_ckpt)}")
        if not os.path.exists(self.args.restore_ckpt):
            return False

        if importlib.util.find_spec("corr_sampler") is None:
            self.args.corr_implementation = "reg"

        # Load Model
        self.device = "cuda"
        self.model = torch.nn.DataParallel(RAFTStereo(self.args), device_ids=[0])
        self.model.load_state_dict(torch.load(self.args.restore_ckpt))

        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

    def init_cam_params(self, depth_range, crop_args):
        init_mtx_l, init_K_l, init_dist_l, init_mtx_r, init_K_r, init_dist_r, init_R, init_t, origin_size = self.load_calibration_results(depth_range)
        
        image_size = self.args.image_size
        self.args.image_size = (image_size[1], image_size[0])
        self.resize_scale_x = self.args.image_size[0] / origin_size[0] # default 1920
        self.resize_scale_y = self.args.image_size[1] / origin_size[1] # default 1200

        # Image Undistortion
        self.undist_map_l_x, self.undist_map_l_y = cv2.initUndistortRectifyMap(init_mtx_l, init_dist_l, None, init_K_l, origin_size, cv2.CV_32F)
        self.undist_map_r_x, self.undist_map_r_y = cv2.initUndistortRectifyMap(init_mtx_r, init_dist_r, None, init_K_r, origin_size, cv2.CV_32F)

        # Resize Matrix
        mtx_l = self.scaleCameraMatrix(init_mtx_l, self.resize_scale_x, self.resize_scale_y)
        K_l = self.scaleCameraMatrix(init_K_l, self.resize_scale_x, self.resize_scale_y)
        mtx_r = self.scaleCameraMatrix(init_mtx_r, self.resize_scale_x, self.resize_scale_y)
        K_r = self.scaleCameraMatrix(init_K_r, self.resize_scale_x, self.resize_scale_y)

        # Image Rectification
        t = np.squeeze(init_t)
        rect_l, rect_r, proj_l, proj_r, Q, _, _ = cv2.stereoRectify(
            mtx_l, init_dist_l, mtx_r, init_dist_r, self.args.image_size, init_R, t, 
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)
        self.rect_map_l_x, self.rect_map_l_y = cv2.initUndistortRectifyMap(K_l, None, rect_l, proj_l, self.args.image_size, cv2.CV_32F)
        self.rect_map_r_x, self.rect_map_r_y = cv2.initUndistortRectifyMap(K_r, None, rect_r, proj_r, self.args.image_size, cv2.CV_32F)

        K, new_R, new_t, _, _, _, _ = cv2.decomposeProjectionMatrix(proj_r)

        self.fx=K[0,0]
        self.baseline = (new_t[0] / new_t[3])

        image_size_tuple = (self.args.image_size[0], self.args.image_size[1])  # (width, height)
        
        # Left와 Right의 rectified → original 변환 맵
        self.unrect_map_l_x, self.unrect_map_l_y = self._get_orig_to_rect_map(mtx_l, init_dist_l, rect_l, proj_l, image_size_tuple)
        self.unrect_map_r_x, self.unrect_map_r_y = self._get_orig_to_rect_map(mtx_r, init_dist_r, rect_r, proj_r, image_size_tuple)
        
        self.dist_map_l_x, self.dist_map_l_y = cv2.initInverseRectificationMap(mtx_l, init_dist_l, None, K_l, self.args.image_size, m1type=cv2.CV_32F)

        self.ori_crop_lx, self.ori_crop_ly = crop_args[0], crop_args[1]
        self.ori_crop_rx, self.ori_crop_ry = crop_args[2], crop_args[3]
        self.crop_w, self.crop_h = crop_args[4], crop_args[5]
            
        self.resize_w = int(self.resize_scale_x * self.crop_w)
        self.resize_h = int(self.resize_scale_y * self.crop_h)

        self.resize_crop_lx = int(self.ori_crop_lx * self.resize_scale_x)
        self.resize_crop_ly = int(self.ori_crop_ly * self.resize_scale_y)
        self.resize_crop_rx = int(self.ori_crop_rx * self.resize_scale_x)
        self.resize_crop_ry = int(self.ori_crop_ry * self.resize_scale_y)

        # 좌표변환을 위한 rect_info 저장 (스케일된 카메라 매트릭스 전달)
        self._store_rect_info(mtx_l, K_l, init_dist_l, mtx_r, K_r, init_dist_r, 
                             init_R, init_t, rect_l, proj_l, rect_r, proj_r, origin_size)

    def reinit(self, cal_file_path, depth_range, crop_args, config_file_path):
        print(f"{CurrentDateTime(0)} [Artis_AI][reinit] Depth Config & Model Re-initialize")
        self.load_model(cal_file_path, config_file_path)
        self.init_cam_params(depth_range, crop_args)

    def define_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--cal_config", default="/home/nvidia/JetsonMan/output/resource/camera/calibration_results.xml", help="kisan config file path")
        parser.add_argument("--restore_ckpt", default="./checkpoints/pretrained_RAFT_fast.pth", help="restore checkpoint")
        parser.add_argument("--mixed_precision", default=True, help="use mixed precision")
        parser.add_argument("--valid_iters", type=int, default=7, help="number of flow-field updates during forward pass")

        # Architecture choices
        parser.add_argument("--image_size", type=int, nargs="+", default=[480, 768], help="size of the random image crops used during training.")
        parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128]*3, help="hidden state and context dimensions")
        parser.add_argument("--corr_implementation", default="reg_cuda", help="correlation volume implementation")
        parser.add_argument("--shared_backbone", default=True, help="use a single backbone for the context and feature encoders")
        parser.add_argument("--corr_levels", type=int, default=4, help="number of levels in the correlation pyramid")
        parser.add_argument("--corr_radius", type=int, default=4, help="width of the correlation pyramid")
        parser.add_argument("--n_downsample", type=int, default=3, help="resolution of the disparity field (1/2^K)")
        parser.add_argument("--context_norm", type=str, default="batch", choices=["group", "batch", "instance", "none"], help="normalization of context encoder")
        parser.add_argument("--slow_fast_gru", default=True, help="iterate the low-res GRUs more frequently")
        parser.add_argument("--n_gru_layers", type=int, default=2, help="number of hidden GRU levels")

        parser.add_argument("--min_depth", type=float, default=400, help="depth clipping with min_depth")
        parser.add_argument("--max_depth", type=float, default=650, help="depth clipping with max_depth")
        
        #args = parser.parse_args()
        args, unknown = parser.parse_known_args()

        return args

    def load_calibration_results(self, depth_range):
        def read_matrix(matrix_elem):
            rows = int(matrix_elem.find("rows").text)
            cols = int(matrix_elem.find("cols").text)
            data_str = matrix_elem.find("data").text
            data = [float(x) for x in data_str.split()]
            return np.array(data).reshape(rows, cols)

        # 1) 메모리 캐시 우선 사용 (serial_calibration_data에 저장된 파라미터)
        try:
            from app.core.Artis_AI.camera.calibration_manager import serial_calibration_data
            for cal_data in serial_calibration_data.values():
                if cal_data.get("cal_file_path") == self.args.cal_config:
                    params = cal_data.get("stereo_cal_params")
                    if params:
                        mtx_l = params["mtx_l"]
                        mtx_r = params["mtx_r"]
                        K_l = params["K_l"]
                        K_r = params["K_r"]
                        dist_l = params["dist_l"]
                        dist_r = params["dist_r"]
                        R = params["R"]
                        t = params["t"]
                        origin_size = params.get("origin_size", [1920, 1200])
                        
                        # depth_range 우선 적용
                        if depth_range[0] > 0:
                            self.args.min_depth = depth_range[0]
                        if depth_range[1] > 0:
                            self.args.max_depth = depth_range[1]
                        
                        return mtx_l, K_l, dist_l, mtx_r, K_r, dist_r, R, t, origin_size
        except Exception:
            pass

        # 2) 캐시에 없으면 파일에서 로드
        tree = ElementTree()
        tree.parse(self.args.cal_config)
        root = tree.getroot()

        K_l = read_matrix(root.find("K_l"))
        dist_l = np.squeeze(read_matrix(root.find("dist_l")))
        K_r = read_matrix(root.find("K_r"))
        dist_r = np.squeeze(read_matrix(root.find("dist_r")))

        R = read_matrix(root.find("R"))
        t = read_matrix(root.find("t"))
        
        mtx_l = read_matrix(root.find("mtx_l"))
        mtx_r = read_matrix(root.find("mtx_r"))

        if depth_range[0] > 0:
            self.args.min_depth = depth_range[0]
        elif root.find("min_depth") != None:
            self.args.min_depth = float(root.find("min_depth").text)
        
        if depth_range[1] > 0:
            self.args.max_depth = depth_range[1]
        elif root.find("max_depth") != None:
            self.args.max_depth = float(root.find("max_depth").text)

        if root.find("img_width") != None:
            image_width = int(root.find("img_width").text)
        else:
            image_width = 1920
        
        if root.find("img_height") != None:
            image_height = int(root.find("img_height").text)
        else:
            image_height = 1200

        return mtx_l, K_l, dist_l, mtx_r, K_r, dist_r, R, t, [image_width, image_height]

    def scaleCameraMatrix(self, mtx, x_scale, y_scale):
        # 원본 배열 보호를 위해 복사본 생성 (메모리 캐시 데이터 변경 방지)
        mtx_copy = mtx.copy()
        mtx_copy[0,0] *= x_scale
        mtx_copy[0,2] *= x_scale
        mtx_copy[1,1] *= y_scale
        mtx_copy[1,2] *= y_scale

        return mtx_copy

    def warm_up(self, loop_max, dep_max):
        warmup_depth_height, warmup_depth_width = cc.image_resolution_depth[0], cc.image_resolution_depth[1]
        if self.is_depth_enable:
            loop_cnt = 0
            dep_time = 0
            while loop_cnt < loop_max:
                self.timer_starter.record()
                with torch.no_grad():
                    image1, image2 = self.load_image("./sample/Cam_2_Color.jpg", "./sample/Cam_1_Color.jpg")

                    padder = InputPadder(image1.shape, divis_by=32)
                    image1, image2 = padder.pad(image1, image2)

                    _, flow_up = self.model(image1, image2, iters=self.args.valid_iters, test_mode=True)
                    flow_up = padder.unpad(flow_up).squeeze()

                    disp = abs(flow_up).cpu().numpy().squeeze()
                    disp += 1e-8
                    depth = self.fx / disp * self.baseline
                    depth = np.clip(depth, self.args.min_depth, self.args.max_depth)

                    depth = cv2.remap(depth, self.unrect_map_l_x, self.unrect_map_l_y, cv2.INTER_NEAREST)
                    disp = cv2.remap(disp, self.unrect_map_l_x, self.unrect_map_l_y, cv2.INTER_NEAREST)

                    depth = depth[self.resize_crop_ly:self.resize_crop_ly + self.resize_h,
                            self.resize_crop_lx:self.resize_crop_lx + self.resize_w]

                self.timer_ender.record()
                torch.cuda.synchronize()
                dep_time = self.timer_starter.elapsed_time(self.timer_ender) / 1000
                print(f"{CurrentDateTime(0)} [Artis_AI] [{loop_cnt + 1}] Raft : {dep_time:.2f}")

                if dep_time < dep_max:
                    break

                loop_cnt += 1

            if loop_cnt == loop_max:
                return False, str(dep_max)

            return True, str(round(dep_time, 2))
        
        else:
            return True, str(0)

    def update_crop_settings(self, left_x, left_y, right_x, right_y):
        if self.ori_crop_lx != left_x or self.ori_crop_ly != left_y or self.ori_crop_rx != right_x or self.ori_crop_ry != right_y:
            self.ori_crop_lx = left_x
            self.ori_crop_ly = left_y
            self.ori_crop_rx = right_x
            self.ori_crop_ry = right_y

            self.resize_crop_lx = int(self.ori_crop_lx * self.resize_scale_x)
            self.resize_crop_ly = int(self.ori_crop_ly * self.resize_scale_y)
            self.resize_crop_rx = int(self.ori_crop_rx * self.resize_scale_x)
            self.resize_crop_ry = int(self.ori_crop_ry * self.resize_scale_y)

            cc.artis_ai_current_log = f'Updated crop settings: left_x({left_x}), left_y({left_y}), right_x({right_x}), right_y({right_y})'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

    def make_input_image(self, img, camera_type: CamInfo, bFlagLDC: bool):
        h, w = img.shape[:2]
        
        if camera_type == CamInfo.LEFT:
            if bFlagLDC:
                img = cv2.remap(img, self.undist_map_l_x, self.undist_map_l_y, cv2.INTER_LINEAR)
            nCropX = self.ori_crop_lx
            nCropY = self.ori_crop_ly
        elif camera_type == CamInfo.RIGHT:
            if bFlagLDC:
                img = cv2.remap(img, self.undist_map_r_x, self.undist_map_r_y, cv2.INTER_LINEAR)
            nCropX = self.ori_crop_rx
            nCropY = self.ori_crop_ry
        elif camera_type == CamInfo.SINGLE:
            center_x = w // 2
            center_y = h // 2
            nCropX = center_x - (self.crop_w // 2)
            nCropY = center_y - (self.crop_h // 2)
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")

        # Crop
        x1 = max(0, nCropX)
        y1 = max(0, nCropY)
        x2 = min(w, nCropX + self.crop_w)
        y2 = min(h, nCropY + self.crop_h)

        img = img[y1:y2, x1:x2]
        #img = cv2.resize(img, (1280, 960))

        return img

    def load_image(self, imfile_l, imfile_r):
        img_l = cv2.imread(imfile_l)
        img_r = cv2.imread(imfile_r)

        cc.artis_ai_json_config = cc.get_config(self.config_file_path, cc.artis_ai_json_config)
        self.update_crop_settings(cc.artis_ai_json_config["crop_lx"], cc.artis_ai_json_config["crop_ly"],
                                  cc.artis_ai_json_config["crop_rx"], cc.artis_ai_json_config["crop_ry"])

        img_l = cv2.resize(img_l, (self.resize_w, self.resize_h))
        img_r = cv2.resize(img_r, (self.resize_w, self.resize_h))

        img_l_resize = np.zeros((self.args.image_size[1], self.args.image_size[0], 3), dtype=np.uint8)
        img_r_resize = np.zeros((self.args.image_size[1], self.args.image_size[0], 3), dtype=np.uint8)

        img_l_resize[self.resize_crop_ly:self.resize_crop_ly+self.resize_h, self.resize_crop_lx:self.resize_crop_lx+self.resize_w] = img_l
        img_r_resize[self.resize_crop_ry:self.resize_crop_ry+self.resize_h, self.resize_crop_rx:self.resize_crop_rx+self.resize_w] = img_r

        undist_rect_img_l = cv2.remap(img_l_resize, self.rect_map_l_x, self.rect_map_l_y, cv2.INTER_LINEAR)
        undist_rect_img_r = cv2.remap(img_r_resize, self.rect_map_r_x, self.rect_map_r_y, cv2.INTER_LINEAR)
        
        undist_rect_img_l = cv2.cvtColor(undist_rect_img_l, cv2.COLOR_BGR2RGB)
        undist_rect_img_r = cv2.cvtColor(undist_rect_img_r, cv2.COLOR_BGR2RGB)

        undist_rect_img_l = np.array(undist_rect_img_l).astype(np.uint8)
        undist_rect_img_r = np.array(undist_rect_img_r).astype(np.uint8)

        undist_rect_img_l = torch.from_numpy(undist_rect_img_l).permute(2, 0, 1).float()
        undist_rect_img_r = torch.from_numpy(undist_rect_img_r).permute(2, 0, 1).float()

        return undist_rect_img_l[None].to(self.device), undist_rect_img_r[None].to(self.device)

    def _get_orig_to_rect_map(self, K, dist, R_rect, P_rect, image_size):
        """원본 이미지 좌표의 픽셀들이 Rectified 이미지의 어디에 해당하는지 매핑하는 Map 생성.
        
        Args:
            K: 원본 카메라 내부 파라미터
            dist: 원본 카메라 왜곡 계수
            R_rect: Rectification 회전 행렬
            P_rect: Rectification 투영 행렬
            image_size: (width, height)
        
        Returns:
            map_x, map_y: cv2.remap용 맵 (float32)
        """
        w, h = image_size
        # 1. 원본 이미지의 모든 픽셀 좌표 생성
        grid_y, grid_x = np.indices((h, w), dtype=np.float32)
        
        # (N, 1, 2) 형태로 변환 for undistortPoints
        pts_orig = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).reshape(-1, 1, 2)
        
        # 2. 원본 -> Rectified 좌표 변환
        # P_rect를 설정하면 Rectified 픽셀 좌표로 바로 나옴
        pts_rect = cv2.undistortPoints(pts_orig, K, dist, R=R_rect, P=P_rect)
        
        # 3. 다시 이미지 형태의 Map으로 변환
        map_x = pts_rect[:, 0, 0].reshape(h, w)
        map_y = pts_rect[:, 0, 1].reshape(h, w)
        
        return map_x, map_y

    def _store_rect_info(self, mtx_l, K_l, dist_l, mtx_r, K_r, dist_r, R, t, rect_l, proj_l, rect_r, proj_r, origin_size):
        """Rectification 정보를 저장 (좌표변환용)"""
        # P1에서 rectified 내부 파라미터 계산 (bbox_transformer에서 재사용)
        K_rect = proj_l[:, :3]
        fx = float(K_rect[0, 0])
        fy = float(K_rect[1, 1])
        cx = float(K_rect[0, 2])
        cy = float(K_rect[1, 2])
        
        self.rect_info = {
            # LEFT 카메라 파라미터
            "mtx_l": mtx_l,     # 원본 카메라 매트릭스 (bbox_transformer에서 K_left로 사용)
            "dist_l": dist_l,
            "R1": rect_l,       # LEFT rectification 회전 행렬
            "P1": proj_l,       # LEFT rectification 투영 행렬
            "fx": fx,           # LEFT rectified focal length (x)
            "fy": fy,           # LEFT rectified focal length (y)
            "cx": cx,           # LEFT rectified principal point (x)
            "cy": cy,           # LEFT rectified principal point (y)
            # RIGHT rectified → 원본 변환 맵
            "map2_orig_x": self.unrect_map_r_x,  # RIGHT rect → orig
            "map2_orig_y": self.unrect_map_r_y,
            "image_size": self.args.image_size,
            "origin_size": origin_size,
            "ori_crop_lx": int(self.ori_crop_lx),           # 원본 이미지에서의 크롭 오프셋 (x)
            "ori_crop_ly": int(self.ori_crop_ly),           # 원본 이미지에서의 크롭 오프셋 (y)
            "crop_w": int(self.crop_w),                     # 크롭된 이미지 너비
            "crop_h": int(self.crop_h),                     # 크롭된 이미지 높이
            "resize_scale_x": float(self.resize_scale_x),   # 원본 → RAFT 스케일 비율 (x)
            "resize_scale_y": float(self.resize_scale_y),   # 원본 → RAFT 스케일 비율 (y)
            "resize_crop_lx": int(self.resize_crop_lx),     # RAFT 캔버스에서의 크롭 영역 오프셋 (x)
            "resize_crop_ly": int(self.resize_crop_ly),     # RAFT 캔버스에서의 크롭 영역 오프셋 (y)
        }
    
    def _get_rect_info(self):
        """저장된 rect_info 반환"""
        return getattr(self, 'rect_info', None)

    def _calc_disp_from_left_flow(self, flow_left_to_right_rect: np.ndarray) -> np.ndarray:
        """Left 기준 RAFT flow -> disparity(px) (양수).  d = u_left - u_right = -du"""
        du = flow_left_to_right_rect[..., 0].astype(np.float32)
        disp = -du
        disp[~np.isfinite(disp)] = np.nan
        disp[disp <= 0] = np.nan
        return disp

    def _warp_left_depth_to_right_rect(self, depth_left: np.ndarray, disp_left: np.ndarray) -> np.ndarray:
        """
        LEFT rect depth를 RIGHT rect 좌표로 forward-splat.

        - 좌표 이동: u_right = u_left - disp
        - 충돌 처리: z-buffer로 더 가까운(depth 작은) 값 유지
        - 업데이트되지 않은 위치(occlusion)는 NaN으로 유지

        return: right rect depth (H, W) float32 (occlusion은 NaN)
        """
        h, w = depth_left.shape
        
        # disp_left shape 보정 (1D/2D 모두 허용)
        if disp_left.ndim == 1:
            # 1차원 배열이면 depth_left와 같은 shape으로 확장
            if disp_left.shape[0] == h:
                # 세로 방향 확장 (각 행 동일)
                disp_left = np.tile(disp_left[:, np.newaxis], (1, w))
            elif disp_left.shape[0] == w:
                # 가로 방향 확장 (각 열 동일)
                disp_left = np.tile(disp_left[np.newaxis, :], (h, 1))
            else:
                raise ValueError(f"disp_left shape {disp_left.shape} cannot be reshaped to match depth_left shape ({h}, {w})")
        elif disp_left.ndim == 2:
            if disp_left.shape != (h, w):
                import cv2
                disp_left = cv2.resize(disp_left, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"disp_left must be 1D or 2D array, got {disp_left.ndim}D with shape {disp_left.shape}")
        
        out_flat = np.full(h * w, np.inf, dtype=np.float32)

        ys, xs = np.indices((h, w), dtype=np.int32)
        # 유효한 depth/disp만 사용
        valid = (
            np.isfinite(depth_left) & (depth_left > 0) &
            np.isfinite(disp_left) & (disp_left > 0)
        )
        if not np.any(valid):
            out = np.full((h, w), np.nan, dtype=np.float32)
            return out

        xs_v = xs[valid]
        ys_v = ys[valid]
        z_v  = depth_left[valid].astype(np.float32)
        d_v  = disp_left[valid].astype(np.float32)

        # RIGHT 좌표로 이동 (정수 픽셀 반올림)
        xr = np.rint(xs_v.astype(np.float32) - d_v).astype(np.int32)
        yr = ys_v

        inb = (xr >= 0) & (xr < w)
        xr = xr[inb]
        yr = yr[inb]
        z_v = z_v[inb]

        # z-buffer: 동일 위치에는 더 작은 depth만 유지
        flat_idx = (yr.astype(np.int64) * w + xr.astype(np.int64))
        np.minimum.at(out_flat, flat_idx, z_v)

        out = out_flat.reshape(h, w)
        out[np.isinf(out)] = np.nan  # occlusion 영역을 NaN으로 설정
        return out

    def _interpolate_depth_holes(self, depth: np.ndarray) -> np.ndarray:
        """
        Depth 맵의 빈 영역(NaN)을 cv2.inpaint를 사용하여 보간합니다.
        """
        if depth is None or depth.size == 0:
            return depth
        
        # NaN/Inf 마스크 생성
        invalid_mask = ~np.isfinite(depth)
        invalid_count = np.count_nonzero(invalid_mask)
        if invalid_count == 0:
            # 유효하지 않은 픽셀이 없으면 그대로 반환
            return depth.copy()

        # invalid 픽셀 비율 체크: 너무 많으면 inpaint가 오래 걸리므로 스킵
        total_pixels = depth.size
        invalid_ratio = invalid_count / total_pixels
        max_invalid_ratio = 0.15
        if invalid_ratio > max_invalid_ratio:
            make_artis_ai_log(
                f'Depth 보간 스킵: invalid 비율 {invalid_ratio*100:.1f}% > {max_invalid_ratio*100:.0f}%',
                'warning', True
            )
            return depth.copy()

        try:
            # 고정 범위로 정규화 (min/max 계산 비용 제거)
            depth_min = self.args.min_depth
            depth_max = self.args.max_depth
            depth_range = depth_max - depth_min

            if depth_range < 1e-6:
                return depth.copy()

            # 정규화 준비: NaN/Inf는 계산 경고 방지를 위해 depth_min으로 임시 대체
            depth_clean = np.where(~invalid_mask, depth, depth_min)
            depth_normalized = ((depth_clean - depth_min) / depth_range * 255).astype(np.uint8)
            # 보간 대상(=invalid_mask) 픽셀의 입력 이미지는 0으로 통일해 결손부를 명확히 표시
            depth_normalized[invalid_mask] = 0
            
            # 보간 마스크: invalid_mask=True인 위치를 255로 표시 (0이 아닌 값은 모두 보간 대상)
            invalid_mask_uint8 = invalid_mask.astype(np.uint8) * 255
            
            # cv2.inpaint (가장 빠른 설정)
            depth_inpainted = cv2.inpaint(depth_normalized, invalid_mask_uint8, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
            
            # 역변환 및 적용 (NaN 영역만)
            depth_result = depth.copy()
            depth_result[invalid_mask] = (depth_inpainted[invalid_mask].astype(np.float32) / 255.0 * depth_range) + depth_min
            
            # 클리핑 (고정 범위)
            depth_result[invalid_mask] = np.maximum(depth_result[invalid_mask], depth_min)
            depth_result[invalid_mask] = np.minimum(depth_result[invalid_mask], depth_max)
            
            return depth_result
            
        except Exception as e:
            # 보간 과정에서 예상치 못한 에러 발생 시 원본 반환
            make_artis_ai_log(cc.artis_ai_current_log, f'Depth 보간: 예상치 못한 에러, 원본 반환: {e}', 'warning', True)
            return depth.copy()

    def _save_depth_bin(self, depth: np.ndarray, filename: str, header_info: tuple) -> bool:
        """
        Depth 데이터를 bin 파일로 저장
        """
        try:
            len_byte_header_depth_version, byte_header_depth_version, header_depth_version = header_info
            
            depth_uint16 = depth.astype(np.uint16)
            depth_uint16 = np.reshape(depth_uint16, (cc.image_resolution_depth[0], cc.image_resolution_depth[1]))
            
            with open(filename, 'wb') as depth_file:
                np.array([len_byte_header_depth_version], dtype=np.uint16).tofile(depth_file)
                depth_file.write(byte_header_depth_version)
                header_depth_version.tofile(depth_file)
                depth_uint16.tofile(depth_file)
            
            cc.artis_ai_current_log = f'Finish Save Depth Binary File : {filename}'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', False)
            return True
        except Exception as e:
            print(f"{CurrentDateTime(0)} [Artis_AI] Depth bin 저장 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_depth_jpg(self, depth: np.ndarray, filename: str, min_depth: float, max_depth: float) -> bool:
        """
        Depth 데이터를 JPG 파일로 저장 (컬러맵 적용)
        """
        try:
            depth_clipped = np.clip(depth, min_depth, max_depth)  # 클리핑
            depth_norm = 255 - ((depth_clipped - min_depth) / (max_depth - min_depth) * 255)  # 200~500mm → 255~0 으로 역변환
            
            depth_uint8 = depth_norm.astype(np.uint8)  # uint8 변환
            depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)  # 컬러맵 적용
            cv2.imwrite(filename, depth_color)
            
            cc.artis_ai_current_log = f'Finish Save Depth Image File : {filename}'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', False)
            return True
        except Exception as e:
            print(f"{CurrentDateTime(0)} [Artis_AI] Depth jpg 저장 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def inference(self, imgs, is_debug, save_path, run_mode, depth_offset):
        if not self.is_depth_enable:
            return 0
        
        with torch.no_grad():
            self.timer_starter.record()
            img_l, img_r = imgs[1], imgs[2]
            image1, image2 = self.load_image(img_l, img_r)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = self.model(image1, image2, iters=self.args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()

            disp = abs(flow_up).cpu().numpy().squeeze()
            disp += 1e-8
            depth = self.fx / disp * self.baseline
            depth = np.clip(depth, self.args.min_depth, self.args.max_depth)

            # 좌표변환을 위한 flow, rect_info, depth map 저장
            flow_np = flow_up.cpu().numpy()
            flow_np = np.stack([flow_np, np.zeros_like(flow_np)], axis=-1) # (H, W) -> (H, W, 2)로 확장
            self.last_flow_rect = flow_np  # rectified 좌표계의 flow (H, W, 2)
            self.last_rect_info = self._get_rect_info()  # rectification 정보
            self.last_depth_rect = depth.copy()  # rectified 좌표계의 depth map (offset 적용 전)

            # Depth 보정 값 적용
            if depth_offset != 0 and depth_offset is not None:
                valid_mask = (depth > self.args.min_depth) & (depth < self.args.max_depth)
                depth[valid_mask] = np.clip(depth[valid_mask] + depth_offset, self.args.min_depth, self.args.max_depth)
                cc.artis_ai_current_log = f'Depth 보정 적용 (rectified): {depth_offset}mm'
                make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

            # ========== 오른쪽 카메라 Depth 생성 및 보간 (LEFT rect → RIGHT rect warping) ==========
            depth_right_rect = None
            use_right_depth = cc.artis_ai_json_config.get("right_depth", False)
            if use_right_depth:
                warp_elapsed = 0.0
                interp_elapsed = 0.0

                # 오른쪽 Depth 생성
                warp_start_time = time.time()
                try:
                    disp_left_rect = self._calc_disp_from_left_flow(flow_np)
                    depth_right_rect = self._warp_left_depth_to_right_rect(depth, disp_left_rect)
                    warp_elapsed = (time.time() - warp_start_time) * 1000  # ms 단위
                except Exception as e:
                    print(f"{CurrentDateTime(0)} [Artis_AI] RIGHT Depth warp 실패 (무시하고 계속): {e}")
                    import traceback
                    traceback.print_exc()

                # 오른쪽 Depth 보간 적용
                if depth_right_rect is not None:
                    interp_start_time = time.time()
                    try:
                        depth_right_rect = self._interpolate_depth_holes(depth_right_rect)
                        interp_elapsed = (time.time() - interp_start_time) * 1000  # ms 단위
                    except Exception as interp_e:
                        print(f"{CurrentDateTime(0)} [Artis_AI] RIGHT Depth 보간 실패 (기존 로직 사용): {interp_e}")

                cc.artis_ai_current_log = f'오른쪽 Depth 생성 & 보간 완료: {warp_elapsed:.2f} + {interp_elapsed:.2f} ms'
                make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

            # ========== LEFT Depth 저장 (LEFT rect → LEFT original) ==========
            depth = cv2.remap(depth, self.unrect_map_l_x, self.unrect_map_l_y, cv2.INTER_NEAREST)
            disp = cv2.remap(disp, self.unrect_map_l_x, self.unrect_map_l_y, cv2.INTER_NEAREST)
            depth = depth[self.resize_crop_ly:self.resize_crop_ly+self.resize_h, self.resize_crop_lx:self.resize_crop_lx+self.resize_w]
            min_d, max_d = self.args.min_depth, self.args.max_depth

            cc.artis_ai_current_log = f'Depth Inference Finished'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', False)

            depth_uint16 = np.array(depth).astype(np.uint16)
            depth_uint16 = np.reshape(depth_uint16, (cc.image_resolution_depth[0], cc.image_resolution_depth[1]))
            
            # raw depth 저장 (좌표변환 및 이미지 합성 시 사용)
            self.raw_depth = depth_uint16.copy()

            # 헤더 정보 생성 (한 번만 생성하고 재사용)
            string_header_depth_version = 'Version'
            byte_header_depth_version = string_header_depth_version.encode('utf-8')
            len_byte_header_depth_version = len(byte_header_depth_version)
            header_depth_version = np.array([cc.artis_ai_depth_format_ver_major, cc.artis_ai_depth_format_ver_minor, cc.artis_ai_depth_format_ver_inner], dtype=np.uint16)
            header_info = (len_byte_header_depth_version, byte_header_depth_version, header_depth_version)

            # LEFT depth 저장 (JPG + BIN)
            filename = img_l.replace("_Color.jpg", "_Depth") #/run/JetsonMan/Cam_2_Depth
            filename_jpg = filename + ".jpg"
            filename_bin = filename + ".bin"
            self._save_depth_jpg(depth, filename_jpg, min_d, max_d)
            self._save_depth_bin(depth_uint16, filename_bin, header_info)

            # ========== RIGHT Depth 저장 (RIGHT rect → RIGHT original) ==========
            if depth_right_rect is not None:
                try:
                    depth_right = cv2.remap(depth_right_rect, self.unrect_map_r_x, self.unrect_map_r_y, cv2.INTER_NEAREST)
                    depth_right = depth_right[self.resize_crop_ry:self.resize_crop_ry+self.resize_h, 
                                              self.resize_crop_rx:self.resize_crop_rx+self.resize_w]
                    
                    # 보간 후에도 남은 NaN/Inf 처리 (max_depth로 변환)
                    depth_right[~np.isfinite(depth_right)] = max_d
                    depth_right = np.clip(depth_right, 0, 65535)
                    
                    depth_right_uint16 = depth_right.astype(np.uint16)
                    depth_right_uint16 = np.reshape(depth_right_uint16, (cc.image_resolution_depth[0], cc.image_resolution_depth[1]))
                    
                    # RIGHT depth 저장 (JPG + BIN)
                    filename_right = img_r.replace("_Color.jpg", "_Depth")
                    filename_right_jpg = filename_right + ".jpg"
                    filename_right_bin = filename_right + ".bin"
                    self._save_depth_jpg(depth_right, filename_right_jpg, min_d, max_d)
                    self._save_depth_bin(depth_right_uint16, filename_right_bin, header_info)
                    
                except Exception as e:
                    print(f"{CurrentDateTime(0)} [Artis_AI] RIGHT Depth 저장 실패 (무시하고 계속): {e}")
                    import traceback
                    traceback.print_exc()

            ret = "OK"
            self.timer_ender.record()
            torch.cuda.synchronize()
            dep_time = self.timer_starter.elapsed_time(self.timer_ender)

            cc.artis_ai_current_log = f'Depth 생성 {dep_time} ms'
            make_artis_ai_log(cc.artis_ai_current_log, 'info', True)

        return ret, dep_time


def crop_calibration_images(crop_settings, temp_dir):
    crop_lx = crop_settings.get("left_x", 180)
    crop_ly = crop_settings.get("left_y", 0)
    crop_rx = crop_settings.get("right_x", 130)
    crop_ry = crop_settings.get("right_y", 0)
    crop_width = crop_settings.get("width", 1600)
    crop_height = crop_settings.get("height", 1200)
    
    input_img_l = os.path.join(temp_dir, "Cal_left.jpg")
    input_img_r = os.path.join(temp_dir, "Cal_right.jpg")
    output_img_l = os.path.join(temp_dir, "Cam_2_Color.jpg")
    output_img_r = os.path.join(temp_dir, "Cam_1_Color.jpg")

    if not os.path.exists(input_img_l) or not os.path.exists(input_img_r):
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : Input image files not found in {temp_dir}")
        return None, None
    
    left_img_data = cv2.imread(input_img_l)
    right_img_data = cv2.imread(input_img_r)

    if left_img_data is None or right_img_data is None:
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : Failed to read image files")
        return None, None
    
    # 왼쪽 이미지 crop
    height, width = left_img_data.shape[:2]
    x1 = max(0, crop_lx)
    y1 = max(0, crop_ly)
    x2 = min(width, crop_lx + crop_width)
    y2 = min(height, crop_ly + crop_height)
    left_img_data = left_img_data[y1:y2, x1:x2]
    
    # 오른쪽 이미지 crop
    height, width = right_img_data.shape[:2]
    x1 = max(0, crop_rx)
    y1 = max(0, crop_ry)
    x2 = min(width, crop_rx + crop_width)
    y2 = min(height, crop_ry + crop_height)
    right_img_data = right_img_data[y1:y2, x1:x2]

    cv2.imwrite(output_img_l, left_img_data)
    cv2.imwrite(output_img_r, right_img_data)
    
    return output_img_l, output_img_r

def calculate_depth_offset(depth_file):
    if not os.path.exists(depth_file):
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : There is no depth file ({depth_file})")
        return None

    try:
        with open(depth_file, 'rb') as f:
            len_header = np.fromfile(f, dtype=np.uint16, count=1)[0]
            # 안전한 디코딩 시도
            try:
                header_str = f.read(len_header).decode('utf-8')
            except UnicodeDecodeError:
                try:
                    header_str = f.read(len_header).decode('utf-8-sig')
                except UnicodeDecodeError:
                    try:
                        header_str = f.read(len_header).decode('latin-1')
                    except UnicodeDecodeError:
                        header_str = f.read(len_header).decode('utf-8', errors='ignore')

            if header_str != 'Version':
                print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : Invalid depth header: {header_str}")
                return None

            _ = np.fromfile(f, dtype=np.uint16, count=3)
            depth = np.fromfile(f, dtype=np.uint16).reshape(cc.image_resolution_depth)
    except Exception as e:
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : Failed to read depth file ({depth_file})")
        return None

    # 중앙 100x100 영역에서 유효 Depth 값 추출
    window_size = 100
    ref = cc.depth_reference_distance
    min_valid, max_valid = ref - 30, ref + 30 # 440 ~ 500mm

    h, w = cc.image_resolution_depth
    cy, cx = h // 2, w // 2
    region = depth[cy - window_size//2:cy + window_size//2, cx - window_size//2:cx + window_size//2]
    valid_values = region[(region > min_valid) & (region < max_valid)]

    # 유효 Depth 값이 없으면 None 반환
    if valid_values.size == 0:
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : 유효한 Depth 값이 없습니다.")
        return None
    
    # 유효 Depth 값이 너무 적으면 None 반환
    if valid_values.size < window_size*window_size*0.5:  # 50% 미만이면
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : 유효한 Depth 값이 너무 적습니다. ({valid_values.size}/{window_size*window_size})")
        return None

    # 이상치 제거
    q1 = np.percentile(valid_values, 25)
    q3 = np.percentile(valid_values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_values = valid_values[(valid_values >= lower_bound) & (valid_values <= upper_bound)]

    if filtered_values.size == 0:
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 적용 X : 이상치 제거 후 유효한 값이 없습니다.")
        return None

    # 중앙값 계산 및 보정값 계산
    median_depth = np.median(filtered_values)
    offset = ref - median_depth

    # 보정값 범위 제한
    max_offset = 50  # 최대 50mm까지만 보정
    if abs(offset) > max_offset:
        print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정값이 너무 큽니다. ({offset:.1f}mm -> {max_offset}mm로 제한)")
        offset = max_offset if offset > 0 else -max_offset

    # 상세 로깅
    print(f"{CurrentDateTime(0)} [Artis_AI] Depth 보정 정보:")
    print(f"{CurrentDateTime(0)}   - 검사 영역: {window_size}x{window_size} (중앙)")
    print(f"{CurrentDateTime(0)}   - 유효 범위: {min_valid}~{max_valid}mm")
    print(f"{CurrentDateTime(0)}   - 유효값 개수: {valid_values.size}/{window_size*window_size}")
    print(f"{CurrentDateTime(0)}   - 중앙값: {median_depth:.1f}mm")
    print(f"{CurrentDateTime(0)}   - 보정값: {offset:.1f}mm")

    return offset

def save_depth_offset_to_file(depth_offset, temp_dir):
    depth_offset_file = os.path.join(temp_dir, "depth_offset.txt")
    
    try:
        with open(depth_offset_file, 'w') as f:
            f.write(str(depth_offset) + '\n')
        print(f"{CurrentDateTime(0)} [Artis_AI] depth_offset saved to file: {depth_offset_file}")
        return True
    except Exception as e:
        print(f"{CurrentDateTime(0)} [Artis_AI] Failed to save depth_offset file: {e}")
        return False

def calculate_and_save_depth_offset(crop_settings, temp_dir, cmd_args=None):
    crop_lx = crop_settings.get("left_x", 180)
    crop_ly = crop_settings.get("left_y", 0)
    crop_rx = crop_settings.get("right_x", 130)
    crop_ry = crop_settings.get("right_y", 0)
    crop_width = crop_settings.get("width", 1600)
    crop_height = crop_settings.get("height", 1200)
    
    depth_file = os.path.join(temp_dir, "Cam_2_Depth.bin")
    
    if not os.path.exists(depth_file):
        if cmd_args is None:
            print(f"{CurrentDateTime(0)} [Artis_AI] depth 파일이 없고 cmd_args도 없어 depth_offset 계산 불가")
            return None
        
        img_l, img_r = crop_calibration_images(crop_settings, temp_dir)
        
        if img_l is not None and img_r is not None:
            ret, dep_time = d1.inference([0, img_l, img_r], cmd_args[2], cmd_args[3], cmd_args[4], 0)
            depth_offset = calculate_depth_offset(depth_file)
            print(f"{CurrentDateTime(0)} [Artis_AI] depth_offset calculated: {depth_offset}")
        else:
            return None
    else:
        # depth 파일이 이미 있으면 바로 계산
        depth_offset = calculate_depth_offset(depth_file)
        print(f"{CurrentDateTime(0)} [Artis_AI] depth_offset calculated: {depth_offset}")
    
    if depth_offset is not None:
        save_depth_offset_to_file(depth_offset, temp_dir)
    
    return depth_offset


if __name__=="__main__":
    import common_config as cc
    config_file_path = "./kisan_config.json"
    #cal_file_path = "/home/nvidia/JetsonMan/resource/camera/calibration_results.xml"
    cal_file_path = "./camera/calibration/default/calibration_results.xml"
    min_depth, max_depth = 200, 500
    crop_lx, crop_ly, crop_rx, crop_ry = 180, 0, 130, 0
    crop_width, crop_height = 1600, 1200

    d1 = Depth(cal_file_path, [min_depth, max_depth], [crop_lx, crop_ly, crop_rx, crop_ry, crop_width, crop_height], config_file_path)    
    ret, dep_time = d1.warm_up(5, 1)

    cc.artis_ai_result_json['log_Artis_AI'] = []
    cam1_path = "./sample/Cam_1_Color.jpg"
    cam2_path = "./sample/Cam_2_Color.jpg"
    imgfile_single = None
    for i in range(1):
        ret, dep_time = d1.inference([cam2_path, cam2_path, cam1_path, imgfile_single], False, "./", None, None)
