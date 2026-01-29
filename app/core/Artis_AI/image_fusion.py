from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import os
import time


# =============================
# Calibration Memory Access
# =============================
def get_calibration_params(serial_number: Optional[str] = None) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    메모리에서 캘리브레이션 파라미터를 가져옴

    Args:
        serial_number: 시리얼 번호 (None이면 첫 번째 유효한 데이터 사용)

    Returns:
        (stereo_cal_params, single_cal_params) 튜플
        - stereo_cal_params: Left-Right 캘리브레이션 (없으면 None)
        - single_cal_params: Left-Single 캘리브레이션 (없으면 None)
    """
    try:
        from app.core.Artis_AI.camera.calibration_manager import serial_calibration_data

        if not serial_calibration_data:
            return None, None

        # 특정 시리얼 번호가 지정된 경우
        if serial_number and serial_number in serial_calibration_data:
            cal_data = serial_calibration_data[serial_number]
            return cal_data.get("stereo_cal_params"), cal_data.get("single_cal_params")

        # 시리얼 번호가 없으면 첫 번째 유효한 stereo_cal_params 찾기
        for cal_data in serial_calibration_data.values():
            stereo_params = cal_data.get("stereo_cal_params")
            if stereo_params:
                single_params = cal_data.get("single_cal_params")
                return stereo_params, single_params

        return None, None
    except Exception as e:
        print(f"[Artis_AI] 캘리브레이션 파라미터 로드 실패: {e}")
        return None, None




# =============================
# CONFIG: 빛반사 감지 및 교체 설정
# =============================

# 빛반사 감지 임계값
# 점수 공식: score = W_V × 밝기(V) + W_INV_S × (255 - 채도(S))
# 점수가 이 값 이상이면 빛반사로 판단
SPECULAR_THRESHOLD = 280

# 빛반사 점수 계산 가중치
SPECULAR_WEIGHTS = {
    'V': 0.7,      # 밝기 가중치
    'INV_S': 0.6,  # 채도 역가중치 (채도가 낮을수록 반사 가능성 높음)
}

# 교체 조건
MIN_SCORE_DIFF_RIGHT = 5.0   # Right가 Left보다 최소 이만큼 낮아야 교체
MIN_SCORE_DIFF_SINGLE = 10.0 # Single이 Left보다 최소 이만큼 낮아야 교체
MIN_REPLACE_AREA = 20        # 교체할 최소 픽셀 수 (이보다 작으면 스킵, MultiView_Fusion.py와 동일하게 작은 영역도 교체)

# 후처리 옵션
USE_MORPHOLOGY = True         # Morphological operations로 마스크 정리
MORPH_KERNEL_SIZE = 3         # Morphology 커널 크기
USE_BLENDING = True           # 반사 영역 경계 블렌딩
BLEND_WIDTH = 5               # 블렌딩 경계 폭 (픽셀)
EXPAND_REPLACE_MASK = 5       # 교체 마스크 확장 (픽셀, 경계가 티날 때 증가)

# 기타 설정
UNDISTORT = False             # True면 각 카메라를 undistort 후 처리
BORDER_VALUE = (0, 0, 0)      # 이미지 경계 채우기 값
INVALID_SCORE = 1e9           # 재투영 불가능한 영역의 점수 (교체 대상에서 제외)


# =============================
# Helper Functions
# =============================
def _normalize_dist(dist: np.ndarray) -> np.ndarray:
    """왜곡 계수를 올바른 shape으로 변환"""
    if dist.ndim == 1:
        return dist.reshape(-1, 1) if dist.size > 0 else np.zeros((5, 1), dtype=np.float64)
    return dist


def _normalize_t(t: np.ndarray) -> np.ndarray:
    """변환 벡터를 (3, 1) shape으로 변환"""
    if t.ndim == 1:
        return t.reshape(3, 1)
    elif t.shape == (1, 3):
        return t.reshape(3, 1)
    return t


def clip_bbox(b: List[float], w: int, h: int, pad: int = 0) -> Tuple[int, int, int, int]:
    """Bounding box 클리핑"""
    x1, y1, x2, y2 = b
    x1 = int(np.floor(x1)) - pad
    y1 = int(np.floor(y1)) - pad
    x2 = int(np.ceil(x2)) + pad
    y2 = int(np.ceil(y2)) + pad
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0, 0
    return x1, y1, x2, y2


def _normalize_contour(contour) -> List[Tuple[float, float]]:
    """
    다양한 형태의 컨투어를 [(x, y), ...] 형태로 정규화
    
    지원 형태:
        - numpy array: (n, 2) 또는 (n, 1, 2)
        - list: [[x, y], ...] 또는 [[[x, y]], ...] 또는 [x1, y1, x2, y2, ...]
    
    Args:
        contour: 정규화할 컨투어 데이터
    
    Returns:
        정규화된 컨투어 포인트 리스트 [(x, y), ...]
    """
    if contour is None or (hasattr(contour, '__len__') and len(contour) == 0):
        return []

    # numpy array인 경우
    if isinstance(contour, np.ndarray):
        contour = contour.squeeze()
        if contour.ndim == 1:
            # [x1, y1, x2, y2, ...] flatten된 형태
            if len(contour) % 2 == 0:
                return [(float(contour[i]), float(contour[i+1])) for i in range(0, len(contour), 2)]
            return []
        elif contour.ndim == 2:
            # [[x1, y1], [x2, y2], ...] 형태
            return [(float(p[0]), float(p[1])) for p in contour]
        return []

    # 리스트인 경우
    if isinstance(contour, list):
        if len(contour) == 0:
            return []

        first = contour[0]

        # [[[x, y]], [[x, y]], ...] 형태 (OpenCV 스타일)
        if isinstance(first, (list, np.ndarray)) and len(first) == 1:
            return [(float(p[0][0]), float(p[0][1])) for p in contour]

        # [[x, y], [x, y], ...] 형태
        if isinstance(first, (list, tuple, np.ndarray)) and len(first) >= 2:
            return [(float(p[0]), float(p[1])) for p in contour]

        # [x1, y1, x2, y2, ...] flatten된 형태
        if isinstance(first, (int, float, np.integer, np.floating)):
            if len(contour) % 2 == 0:
                return [(float(contour[i]), float(contour[i+1])) for i in range(0, len(contour), 2)]

    return []


def create_contour_mask(
    contours: List,
    img_shape: Tuple[int, int],
    bbox: Optional[Tuple[int, int, int, int]] = None
) -> np.ndarray:
    """
    컨투어로 마스크 생성 (다양한 컨투어 형태 지원)
    
    Args:
        contours: 컨투어 리스트 (다양한 형태 지원)
        img_shape: 이미지 크기 (h, w)
        bbox: 마스크를 제한할 bbox 영역 (x1, y1, x2, y2), None이면 전체 이미지
    
    Returns:
        마스크 배열 (h, w) uint8, 255=내부, 0=외부
    """
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if not contours:
        return mask

    for contour in contours:
        # 컨투어 정규화
        normalized = _normalize_contour(contour)
        if len(normalized) < 3:
            continue

        pts = np.array([[int(p[0]), int(p[1])] for p in normalized], dtype=np.int32)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            pts[:, 0] = np.clip(pts[:, 0], x1, x2 - 1)
            pts[:, 1] = np.clip(pts[:, 1], y1, y2 - 1)

        cv2.fillPoly(mask, [pts], 255)

    return mask


def _filter_small_regions(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    최소 연결 영역 크기 이상인 영역만 유지

    Args:
        mask: 입력 마스크 (bool 또는 uint8)
        min_area: 유지할 최소 영역 크기 (픽셀 수)

    Returns:
        필터링된 마스크 (bool)
    """
    if not np.any(mask):
        return mask.astype(bool) if mask.dtype != bool else mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep_mask = np.zeros_like(mask, dtype=bool)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            keep_mask[labels == label_id] = True
    return keep_mask


def _filter_and_check_regions(
    mask: np.ndarray,
    min_noise_area: int = 20,
    min_replace_area: int = 500
) -> Tuple[np.ndarray, bool]:
    """
    작은 노이즈 영역을 제거하고, 충분히 큰 영역이 있는지 확인

    connectedComponentsWithStats를 한 번만 호출하여 두 가지 작업을 수행:
    1. min_noise_area 미만의 작은 영역 제거
    2. min_replace_area 이상의 큰 영역 존재 여부 확인

    Args:
        mask: 입력 마스크 (bool 또는 uint8)
        min_noise_area: 노이즈로 간주할 최소 영역 크기
        min_replace_area: 교체를 진행할 최소 영역 크기

    Returns:
        (filtered_mask, has_large_region) 튜플
        - filtered_mask: 노이즈가 제거된 마스크 (bool)
        - has_large_region: min_replace_area 이상 영역 존재 여부
    """
    if not np.any(mask):
        return mask.astype(bool) if mask.dtype != bool else mask, False

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    keep_mask = np.zeros_like(mask, dtype=bool)
    has_large_region = False

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_noise_area:
            keep_mask[labels == label_id] = True
        if area >= min_replace_area:
            has_large_region = True

    return keep_mask, has_large_region


# =============================
# Rectified-depth aware mapping
# =============================
def _orig_to_rect_uv(
    uu: np.ndarray,
    vv: np.ndarray,
    K_l: np.ndarray,
    D_l: np.ndarray,
    R1: np.ndarray,
    P1: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Left 원본 좌표계 픽셀 -> Left rectified 좌표계 픽셀 변환

    Args:
        uu, vv: 원본 좌표계 픽셀 그리드 (동일한 shape)
        K_l: Left 카메라 내부 파라미터
        D_l: Left 카메라 왜곡 계수
        R1: Left rectification rotation matrix
        P1: Left rectification projection matrix

    Returns:
        u_rect, v_rect: rectified 좌표계 픽셀 그리드 (동일한 shape)
    """
    pts = np.stack([uu, vv], axis=-1).reshape(-1, 1, 2).astype(np.float32)
    rect_pts = cv2.undistortPoints(pts, K_l, D_l, R=R1, P=P1)
    u_rect = rect_pts[:, 0, 0].reshape(uu.shape).astype(np.float32)
    v_rect = rect_pts[:, 0, 1].reshape(vv.shape).astype(np.float32)
    return u_rect, v_rect


def _sample_depth_bilinear(depth_rect: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bilinear interpolation으로 depth 값 샘플링
    
    Args:
        depth_rect: rectified 좌표계의 depth 맵 (H, W) float32
        u, v: 샘플링할 rectified 좌표 (동일한 shape)
    
    Returns:
        Z: 샘플링된 depth 값 (동일한 shape) float32
        valid: 유효한 픽셀 마스크 (동일한 shape) bool
    """
    h, w = depth_rect.shape
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    valid = (u0 >= 0) & (v0 >= 0) & (u1 < w) & (v1 < h)

    u0c = np.clip(u0, 0, w - 1)
    v0c = np.clip(v0, 0, h - 1)
    u1c = np.clip(u1, 0, w - 1)
    v1c = np.clip(v1, 0, h - 1)

    Ia = depth_rect[v0c, u0c]
    Ib = depth_rect[v0c, u1c]
    Ic = depth_rect[v1c, u0c]
    Id = depth_rect[v1c, u1c]

    wa = (u1 - u) * (v1 - v)
    wb = (u - u0) * (v1 - v)
    wc = (u1 - u) * (v - v0)
    wd = (u - u0) * (v - v0)

    Z = (wa * Ia + wb * Ib + wc * Ic + wd * Id).astype(np.float32)
    valid &= (Z > 0)
    return Z, valid


# =============================
# Core: Reprojection + Fusion
# =============================
def build_reproject_map_left_to_target(
    roi_xyxy: Tuple[int, int, int, int],
    depth_rect: np.ndarray,
    R_lt: np.ndarray,
    t_lt: np.ndarray,
    K_t: np.ndarray,
    target_shape: Tuple[int, int],
    stereo_params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Left 원본 좌표계 ROI -> Target 카메라 재투영 맵 생성

    좌표 변환 흐름:
        1. 원본 좌표 -> Rectified 좌표 (u_rect, v_rect) via cv2.undistortPoints
        2. Rectified 좌표에서 depth 샘플링 (depth 해상도로 스케일링)
        3. P1로 3D 역투영 (Rectified 좌표계)
        4. Rectified 3D -> 원본 3D 변환 (R1.T)
        5. 원본 3D -> Target 3D 변환 (R, t)
        6. Target 카메라로 2D 투영 (K_t)

    Args:
        roi_xyxy: 원본 좌표계의 ROI 좌표 (x1, y1, x2, y2)
        depth_rect: Rectified 좌표계의 depth 맵 (H, W) mm 단위 (예: 768x480)
        R_lt: Left -> Target 회전 행렬 (3x3)
        t_lt: Left -> Target 변환 벡터 (3x1) mm 단위
        K_t: Target 카메라 내부 파라미터 (3x3)
        target_shape: Target 이미지 크기 (H, W)
        stereo_params: 스테레오 캘리브레이션 파라미터
            - K_l: Left 카메라 내부 파라미터 (원본)
            - dist_l: Left 카메라 왜곡 계수
            - R1: Left rectification rotation matrix
            - P1: Left rectification projection matrix

    Returns:
        u_t: Target 이미지의 u 좌표 맵 (roi_h, roi_w) float32
        v_t: Target 이미지의 v 좌표 맵 (roi_h, roi_w) float32
        valid: 유효한 픽셀 마스크 (roi_h, roi_w) bool
    """
    # 파라미터 추출
    K_l = stereo_params["K_l"].astype(np.float64)
    D_l = _normalize_dist(stereo_params["dist_l"]).astype(np.float64)
    R1 = stereo_params["R1"].astype(np.float64)
    P1 = stereo_params["P1"].astype(np.float64)

    depth_h, depth_w = depth_rect.shape[:2]
    x1, y1, x2, y2 = roi_xyxy
    roi_w = x2 - x1
    roi_h = y2 - y1
    th, tw = target_shape

    # ========== (1) 원본 좌표 그리드 생성 ==========
    u = np.arange(x1, x2, dtype=np.float32)
    v = np.arange(y1, y2, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # ========== (2) 원본 좌표 -> Rectified 좌표 변환 ==========
    u_rect, v_rect = _orig_to_rect_uv(uu, vv, K_l, D_l, R1, P1)

    # ========== (3) Rectified 좌표를 depth 해상도로 스케일링 ==========
    # P1은 원본 해상도(예: 1920x1200) 기준, depth는 추론 해상도(768x480)
    rect_cx, rect_cy = float(P1[0, 2]), float(P1[1, 2])
    rect_w_estimated = rect_cx * 2
    rect_h_estimated = rect_cy * 2
    scale_rect_to_depth_x = depth_w / rect_w_estimated
    scale_rect_to_depth_y = depth_h / rect_h_estimated

    u_rect_scaled = u_rect * scale_rect_to_depth_x
    v_rect_scaled = v_rect * scale_rect_to_depth_y

    # ========== (4) Rectified 좌표에서 depth 샘플링 (bilinear) ==========
    Z, valid = _sample_depth_bilinear(depth_rect.astype(np.float32), u_rect_scaled, v_rect_scaled)

    # ========== (5) P1로 3D 역투영 (Rectified 좌표계) ==========
    fx, fy = float(P1[0, 0]), float(P1[1, 1])
    cx, cy = float(P1[0, 2]), float(P1[1, 2])

    Xr = (u_rect - cx) * Z / fx
    Yr = (v_rect - cy) * Z / fy

    P_rect = np.stack([Xr, Yr, Z], axis=0).reshape(3, -1)  # (3, N)

    # ========== (6) Rectified 3D -> 원본 3D 변환 ==========
    P_cam2 = R1.T @ P_rect  # (3, N)

    # ========== (7) 원본 Left -> Target 변환 ==========
    t_lt = _normalize_t(t_lt)
    P_t = (R_lt @ P_cam2) + t_lt  # (3, N)

    Xt = P_t[0, :].reshape(roi_h, roi_w)
    Yt = P_t[1, :].reshape(roi_h, roi_w)
    Zt = P_t[2, :].reshape(roi_h, roi_w)

    valid &= (Zt > 1e-6)

    # ========== (8) K_t로 Target 2D 투영 ==========
    fx_t, fy_t = float(K_t[0, 0]), float(K_t[1, 1])
    cx_t, cy_t = float(K_t[0, 2]), float(K_t[1, 2])

    u_t = (Xt * fx_t / Zt) + cx_t
    v_t = (Yt * fy_t / Zt) + cy_t

    # 이미지 경계 확인
    valid &= (u_t >= 0) & (u_t <= (tw - 1)) & (v_t >= 0) & (v_t <= (th - 1))

    return u_t.astype(np.float32), v_t.astype(np.float32), valid


def undistort_if_needed(img: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    필요시 이미지 왜곡 보정 적용
    
    Args:
        img: 입력 이미지
        K: 카메라 내부 파라미터
        D: 왜곡 계수
    
    Returns:
        왜곡 보정된 이미지 (UNDISTORT=False면 원본 반환)
    """
    if not UNDISTORT:
        return img
    return cv2.undistort(img, K, D, None, K)


def build_reproject_map_left_to_target(
    roi_xyxy: Tuple[int, int, int, int],
    depth_mm: np.ndarray,
    R_lt: np.ndarray,
    t_lt: np.ndarray,
    K_t: np.ndarray,
    target_shape: Tuple[int, int],
    stereo_params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Left 카메라 ROI -> Target 카메라 재투영 맵 생성

    재투영 과정:
        1. Left 원본 좌표 -> Left rectified 좌표 변환
        2. Rectified 좌표에서 depth 샘플링
        3. Rectified 3D 좌표로 역투영
        4. Left 원본 3D 좌표로 변환
        5. Target 카메라 3D 좌표로 변환
        6. Target 카메라 2D 좌표로 투영

    Args:
        roi_xyxy: Left 이미지의 ROI 좌표 (x1, y1, x2, y2)
        depth_mm: Left 카메라의 rectified depth 맵 (H, W) mm 단위
        R_lt: Left -> Target 회전 행렬 (3x3)
        t_lt: Left -> Target 변환 벡터 (3x1) mm 단위
        K_t: Target 카메라 내부 파라미터 (3x3)
        target_shape: Target 이미지 크기 (H, W)
        stereo_params: 스테레오 캘리브레이션 파라미터 (serial_calibration_data에서 가져온 값)
            - K_l: Left 카메라 내부 파라미터
            - dist_l: Left 카메라 왜곡 계수
            - R1: Left rectification rotation matrix
            - P1: Left rectification projection matrix

    Returns:
        u_t: Target 이미지의 u 좌표 맵 (roi_h, roi_w) float32
        v_t: Target 이미지의 v 좌표 맵 (roi_h, roi_w) float32
        valid: 유효한 픽셀 마스크 (roi_h, roi_w) bool
    """
    # stereo_params에서 rectify 정보 추출
    K_l = stereo_params["K_l"].astype(np.float64)
    D_l = _normalize_dist(stereo_params["dist_l"]).astype(np.float64)
    R1 = stereo_params["R1"].astype(np.float64)
    P1 = stereo_params["P1"].astype(np.float64)

    x1, y1, x2, y2 = roi_xyxy
    roi_w = x2 - x1
    roi_h = y2 - y1
    th, tw = target_shape

    # Left 원본 좌표계에서 ROI 픽셀 그리드 생성
    u = np.arange(x1, x2, dtype=np.float32)
    v = np.arange(y1, y2, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # (A) Left 원본 좌표 -> Left rectified 좌표 변환
    u_rect, v_rect = _orig_to_rect_uv(uu, vv, K_l, D_l, R1, P1)

    # (B) Rectified 좌표에서 depth 샘플링 (bilinear interpolation)
    Z, valid = _sample_depth_bilinear(depth_mm, u_rect, v_rect)

    # (C) Rectified 좌표계에서 3D 좌표로 역투영 (P1 내부 파라미터 사용)
    fx, fy = float(P1[0, 0]), float(P1[1, 1])
    cx, cy = float(P1[0, 2]), float(P1[1, 2])

    Xr = (u_rect - cx) * Z / fx
    Yr = (v_rect - cy) * Z / fy

    P_rect = np.stack([Xr, Yr, Z], axis=0).reshape(3, -1)

    # (D) Rectified 3D 좌표 -> Left 원본 3D 좌표 변환
    P_cam2 = (R1.T @ P_rect)

    # (E) Left 원본 3D 좌표 -> Target 카메라 3D 좌표 변환
    t_lt = _normalize_t(t_lt)
    P_t = (R_lt @ P_cam2) + t_lt

    Xt = P_t[0, :].reshape(roi_h, roi_w)
    Yt = P_t[1, :].reshape(roi_h, roi_w)
    Zt = P_t[2, :].reshape(roi_h, roi_w)

    valid &= (Zt > 1e-6)

    # (F) Target 카메라 3D 좌표 -> Target 이미지 2D 좌표 투영
    fx_t, fy_t = float(K_t[0, 0]), float(K_t[1, 1])
    cx_t, cy_t = float(K_t[0, 2]), float(K_t[1, 2])

    u_t = (Xt * fx_t / Zt) + cx_t
    v_t = (Yt * fy_t / Zt) + cy_t

    # 이미지 경계 내부인지 확인
    valid &= (u_t >= 0) & (u_t <= (tw - 1)) & (v_t >= 0) & (v_t <= (th - 1))

    return u_t.astype(np.float32), v_t.astype(np.float32), valid


def specular_score_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    빛반사 점수 계산
    
    점수 공식: score = W_V × 밝기(V) + W_INV_S × (255 - 채도(S))
    - 밝기가 높을수록 반사 가능성 높음
    - 채도가 낮을수록 반사 가능성 높음
    
    Args:
        img_bgr: BGR 이미지 (H, W, 3) uint8
    
    Returns:
        빛반사 점수 맵 (H, W) float32, 클수록 반사 가능성 높음
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)  # 채도 (Saturation)
    V = hsv[:, :, 2].astype(np.float32)   # 밝기 (Value)

    score = (SPECULAR_WEIGHTS['V'] * V) + \
            (SPECULAR_WEIGHTS['INV_S'] * (255.0 - S))
    return score.astype(np.float32)


def fuse_roi_reflection_free(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    depth_rect: np.ndarray,
    roi_xyxy: Tuple[int, int, int, int],
    stereo_params: Dict[str, Any],
    crop_params: Dict[str, Any],
    single_bgr: Optional[np.ndarray] = None,
    single_params: Optional[Dict[str, Any]] = None,
    contour_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Left ROI 기준으로 Right/Single 카메라에서 재투영하여 빛반사 영역만 선택적으로 교체

    처리 과정:
        1. Left ROI에서 빛반사 영역 감지 (specular score >= threshold)
        2. Right/Single 카메라로 재투영하여 해당 영역의 이미지 가져오기
        3. 각 후보의 빛반사 점수 비교하여 가장 낮은 점수의 픽셀 선택
        4. Right 우선 교체, Single은 Right로 교체되지 않은 영역에서만 고려

    Args:
        left_bgr: Left 카메라 BGR 이미지 (전체 이미지, 원본 좌표계)
        right_bgr: Right 카메라 BGR 이미지 (전체 이미지, 원본 해상도)
        depth_rect: Rectified 좌표계 depth 맵 (H, W) mm 단위 (예: 768x480)
        roi_xyxy: 원본 좌표계의 ROI 좌표 (x1, y1, x2, y2)
        stereo_params: Left-Right 스테레오 캘리브레이션 파라미터
            - K_l, K_r: Left/Right 카메라 내부 파라미터 (원본 좌표계)
            - dist_l, dist_r: Left/Right 카메라 왜곡 계수
            - R, t: Left->Right 회전/변환
            - R1, P1: Left rectification 행렬
        crop_params: 크롭 파라미터 (호환성 유지용, 사용 안 함)
        single_bgr: Single 카메라 BGR 이미지 (전체 이미지), None이면 Right에서만 합성
        single_params: Left-Single 캘리브레이션 파라미터 (single_bgr가 있을 때만 필요)
            - K_s: Single 카메라 내부 파라미터
            - dist_s: Single 카메라 왜곡 계수
            - R_s, t_s: Left->Single 회전/변환
        contour_mask: ROI 영역의 컨투어 마스크 (전체 이미지 크기), None이면 전체 ROI 사용

    Returns:
        (fused, replaced_mask, specular_mask) 튜플:
        - fused: 빛반사가 제거된 ROI 이미지 (roi_h, roi_w, 3) uint8
        - replaced_mask: 실제 교체된 영역 마스크 (roi_h, roi_w) bool (디버그: 파란색)
        - specular_mask: 빛반사가 감지된 영역 전체 마스크 (roi_h, roi_w) bool (디버그: 빨간색)
    """
    x1, y1, x2, y2 = roi_xyxy
    roi_left = left_bgr[y1:y2, x1:x2]
    roi_h_rgb = y2 - y1
    roi_w_rgb = x2 - x1

    # 컨투어 마스크 준비 (전체 이미지 크기에서 ROI 영역만 추출)
    if contour_mask is not None:
        roi_contour_mask = contour_mask[y1:y2, x1:x2] > 0
    else:
        roi_contour_mask = np.ones((roi_h_rgb, roi_w_rgb), dtype=bool)

    # RGB 좌표계 bbox를 원본 좌표계로 변환 (재투영 맵 생성용)
    # crop_params에서 변환 정보 가져오기
    if crop_params:
        rgb_w = crop_params.get("rgb_w", left_bgr.shape[1])
        rgb_h = crop_params.get("rgb_h", left_bgr.shape[0])
        crop_w = crop_params.get("crop_w", rgb_w)
        crop_h = crop_params.get("crop_h", rgb_h)
        ori_crop_lx = crop_params.get("ori_crop_lx", 0)
        ori_crop_ly = crop_params.get("ori_crop_ly", 0)
        
        # RGB 좌표 -> 원본 좌표 변환
        scale_x = crop_w / rgb_w
        scale_y = crop_h / rgb_h
        x1_orig = x1 * scale_x + ori_crop_lx
        y1_orig = y1 * scale_y + ori_crop_ly
        x2_orig = x2 * scale_x + ori_crop_lx
        y2_orig = y2 * scale_y + ori_crop_ly
        roi_xyxy_orig = (int(x1_orig), int(y1_orig), int(x2_orig), int(y2_orig))
        
        # 원본 좌표계 크기 계산 (재투영 맵 생성용)
        orig_w = ori_crop_lx + crop_w
        orig_h = ori_crop_ly + crop_h
        target_shape_orig = (orig_h, orig_w)  # 원본 좌표계 크기
    else:
        # crop_params가 없으면 그대로 사용 (이전 동작 유지)
        roi_xyxy_orig = roi_xyxy
        target_shape_orig = right_bgr.shape[:2]

    # Left-Right 캘리브레이션 파라미터 추출
    R_lr = stereo_params["R"].astype(np.float64)
    t_lr = _normalize_t(stereo_params["t"]).astype(np.float64)
    K_r = stereo_params["K_r"].astype(np.float64)

    # Single 카메라 사용 가능 여부 확인
    use_single = single_bgr is not None and single_params is not None

    # ========== 1. Left->Right 재투영 맵 생성 및 이미지 재매핑 ==========
    map_rx, map_ry, valid_r = build_reproject_map_left_to_target(
        roi_xyxy_orig, depth_rect, R_lr, t_lr, K_r,  # roi_xyxy_orig 사용
        target_shape=target_shape_orig,  # 원본 좌표계 크기 사용
        stereo_params=stereo_params
    )
    
    # right_bgr은 원본 좌표계이므로 원본 좌표계 크기의 맵으로 remap
    roi_right_orig = cv2.remap(
        right_bgr, map_rx, map_ry, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=BORDER_VALUE
    )
    
    # remap된 이미지와 valid_r을 RGB 좌표계 크기로 리사이즈
    if crop_params and (roi_right_orig.shape[0] != roi_h_rgb or roi_right_orig.shape[1] != roi_w_rgb):
        roi_right = cv2.resize(roi_right_orig, (roi_w_rgb, roi_h_rgb), interpolation=cv2.INTER_LINEAR)
        valid_r = cv2.resize(valid_r.astype(np.float32), (roi_w_rgb, roi_h_rgb), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        roi_right = roi_right_orig

    # ========== 2. Left->Single 재투영 맵 생성 및 이미지 재매핑 (Single이 있는 경우만) ==========
    if use_single:
        R_ls = single_params["R_s"].astype(np.float64)
        t_ls = _normalize_t(single_params["t_s"]).astype(np.float64)
        K_s = single_params["K_s"].astype(np.float64)

        # Single 카메라도 원본 좌표계 크기 사용
        if crop_params:
            target_shape_single_orig = target_shape_orig
        else:
            target_shape_single_orig = single_bgr.shape[:2] if single_bgr is not None else None
        
        map_sx, map_sy, valid_s = build_reproject_map_left_to_target(
            roi_xyxy_orig, depth_rect, R_ls, t_ls, K_s,  # roi_xyxy_orig 사용
            target_shape=target_shape_single_orig,  # 원본 좌표계 크기 사용
            stereo_params=stereo_params
        )
        
        # single_bgr도 원본 좌표계이므로 원본 좌표계 크기의 맵으로 remap
        roi_single_orig = cv2.remap(
            single_bgr, map_sx, map_sy, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=BORDER_VALUE
        )
        
        # remap된 이미지와 valid_s를 RGB 좌표계 크기로 리사이즈
        if crop_params and (roi_single_orig.shape[0] != roi_h_rgb or roi_single_orig.shape[1] != roi_w_rgb):
            roi_single = cv2.resize(roi_single_orig, (roi_w_rgb, roi_h_rgb), interpolation=cv2.INTER_LINEAR)
            valid_s = cv2.resize(valid_s.astype(np.float32), (roi_w_rgb, roi_h_rgb), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            roi_single = roi_single_orig
    else:
        roi_single = None
        valid_s = np.zeros((roi_h_rgb, roi_w_rgb), dtype=bool)

    # ========== 3. 빛반사 점수 계산 ==========
    s_left = specular_score_bgr(roi_left)
    s_right = specular_score_bgr(roi_right)
    s_single = specular_score_bgr(roi_single) if use_single else np.full_like(s_left, INVALID_SCORE)

    # 재투영 불가능한 영역은 매우 큰 점수로 설정 (교체 대상에서 제외)
    s_right = np.where(valid_r, s_right, INVALID_SCORE)
    s_single = np.where(valid_s, s_single, INVALID_SCORE) if use_single else s_single

    # ========== Left depth 유효성 체크 (원본 좌표계에서 rectified 좌표로 변환하여 depth_rect에서 확인) ==========
    # 이전 코드: valid_l = depth_mm[y1:y2, x1:x2] > 0.0 (원본 좌표계에서 직접 인덱싱)
    # 현재: depth_rect는 rectified 해상도이므로 원본 좌표를 rectified 좌표로 변환 후 샘플링
    K_l = stereo_params["K_l"].astype(np.float64)
    D_l = _normalize_dist(stereo_params["dist_l"]).astype(np.float64)
    R1 = stereo_params["R1"].astype(np.float64)
    P1 = stereo_params["P1"].astype(np.float64)

    # RGB 좌표계 bbox를 원본 좌표계로 변환하여 depth 샘플링
    if crop_params:
        # 원본 좌표 그리드 (재투영 맵 생성과 동일한 변환)
        x1_orig, y1_orig, x2_orig, y2_orig = roi_xyxy_orig
        u_orig = np.arange(x1_orig, x2_orig, dtype=np.float32)
        v_orig = np.arange(y1_orig, y2_orig, dtype=np.float32)
        uu_orig, vv_orig = np.meshgrid(u_orig, v_orig)
        
        # 원본 좌표 -> Rectified 좌표 변환
        u_rect, v_rect = _orig_to_rect_uv(uu_orig, vv_orig, K_l, D_l, R1, P1)
    else:
        # crop_params가 없으면 기존 방식 사용
        u = np.arange(x1, x2, dtype=np.float32)
        v = np.arange(y1, y2, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        u_rect, v_rect = _orig_to_rect_uv(uu, vv, K_l, D_l, R1, P1)

    # Rectified 좌표를 depth 해상도로 스케일링
    depth_h, depth_w = depth_rect.shape[:2]
    rect_cx, rect_cy = float(P1[0, 2]), float(P1[1, 2])
    rect_w_estimated = rect_cx * 2
    rect_h_estimated = rect_cy * 2
    scale_rect_to_depth_x = depth_w / rect_w_estimated
    scale_rect_to_depth_y = depth_h / rect_h_estimated

    u_rect_scaled = u_rect * scale_rect_to_depth_x
    v_rect_scaled = v_rect * scale_rect_to_depth_y

    # depth_rect에서 depth 샘플링 및 > 0 확인
    Z, valid_depth_sample = _sample_depth_bilinear(depth_rect.astype(np.float32), u_rect_scaled, v_rect_scaled)
    valid_l = (Z > 0.0) & valid_depth_sample
    
    # valid_l을 RGB 좌표계 크기로 리사이즈 (크기가 다른 경우만)
    if crop_params and (valid_l.shape[0] != roi_h_rgb or valid_l.shape[1] != roi_w_rgb):
        valid_l = cv2.resize(valid_l.astype(np.float32), (roi_w_rgb, roi_h_rgb), interpolation=cv2.INTER_NEAREST).astype(bool)

    # 기본값 설정
    fused = roi_left.copy()
    empty_mask = np.zeros((roi_h_rgb, roi_w_rgb), dtype=bool)

    # ========== 4. 빛반사 영역 감지 ==========
    s_left_above_threshold = s_left >= SPECULAR_THRESHOLD
    specular_mask_raw = s_left_above_threshold & roi_contour_mask  # 컨투어 내부의 반사 영역
    specular_mask = specular_mask_raw & valid_l  # depth가 있는 반사 영역만 (재투영 가능)

    if not np.any(specular_mask):
        return fused, empty_mask, empty_mask

    # ========== 5. 교체 후보 선택 (Right 우선 전략) ==========
    score_improvement_right = s_left - s_right
    replace_from_right = specular_mask & valid_r & (score_improvement_right >= MIN_SCORE_DIFF_RIGHT)

    # Right로 교체되지 않은 영역에서만 Single 고려 (Single이 있는 경우만)
    if use_single:
        remaining_specular = specular_mask & ~replace_from_right
        score_improvement_single = s_left - s_single
        replace_from_single = remaining_specular & valid_s & (score_improvement_single >= MIN_SCORE_DIFF_SINGLE)
        replace_from_single = replace_from_single & ~replace_from_right  # Right 우선 사용
    else:
        replace_from_single = np.zeros((y2 - y1, x2 - x1), dtype=bool)

    # ========== 6. 작은 노이즈 영역 제거 + 교체 영역 크기 체크 (통합) ==========
    # connectedComponentsWithStats 1회 호출로 두 가지 작업 수행:
    # 1) 20px 미만 노이즈 제거  2) MIN_REPLACE_AREA 이상 영역 존재 확인
    replace_from_right = _filter_small_regions(replace_from_right, min_area=20)
    if use_single:
        replace_from_single = _filter_small_regions(replace_from_single, min_area=20)

    replace_total = replace_from_right | replace_from_single

    if not np.any(replace_total):
        candidate_mask = specular_mask
        return fused, empty_mask, candidate_mask

    # 노이즈 제거 + 큰 영역 존재 여부를 한 번에 확인
    replace_total, has_large_region = _filter_and_check_regions(
        replace_total, min_noise_area=20, min_replace_area=MIN_REPLACE_AREA
    )

    if not has_large_region:
        candidate_mask = specular_mask
        return fused, empty_mask, candidate_mask

    # ========== 8. Morphological operations로 마스크 정리 ==========
    if USE_MORPHOLOGY:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

        def clean_mask(mask, valid_mask):
            """Morphological open/close로 마스크 정리"""
            if not np.any(mask):
                return mask
            cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            return cleaned.astype(bool) & valid_mask

        replace_from_right = clean_mask(replace_from_right, valid_r)
        if use_single:
            replace_from_single = clean_mask(replace_from_single, valid_s)

    # ========== 9. 교체 마스크 확장 (경계가 티나지 않도록) ==========
    if EXPAND_REPLACE_MASK > 0:
        expand_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (EXPAND_REPLACE_MASK * 2 + 1, EXPAND_REPLACE_MASK * 2 + 1)
        )
        replace_from_right = cv2.dilate(
            replace_from_right.astype(np.uint8), expand_kernel, iterations=1
        ).astype(bool) & valid_r
        if use_single:
            replace_from_single = cv2.dilate(
                replace_from_single.astype(np.uint8), expand_kernel, iterations=1
            ).astype(bool) & valid_s

    # ========== 10. 픽셀 교체 (블렌딩 여부에 따라 분기) ==========
    if USE_BLENDING and BLEND_WIDTH > 0:
        # 블렌딩 사용: replaced 배열에 교체 후 블렌딩 적용
        replace_mask_uint8 = (replace_from_right | replace_from_single).astype(np.uint8)

        if np.any(replace_mask_uint8 > 0):
            # Gaussian blur로 블렌딩 가중치 생성
            blend_weight = replace_mask_uint8.astype(np.float32)
            blend_weight = cv2.GaussianBlur(blend_weight, (BLEND_WIDTH * 2 + 1, BLEND_WIDTH * 2 + 1), 0)
            blend_weight = np.clip(blend_weight, 0, 1)
            blend_weight_3d = np.stack([blend_weight] * 3, axis=2)

            # 교체된 이미지 준비
            replaced = roi_left.copy()
            replaced[replace_from_right] = roi_right[replace_from_right]
            if use_single:
                replaced[replace_from_single] = roi_single[replace_from_single]

            # 블렌딩 적용
            fused = (roi_left * (1 - blend_weight_3d) + replaced * blend_weight_3d).astype(np.uint8)
    else:
        # 블렌딩 미사용: 직접 픽셀 교체
        fused[replace_from_right] = roi_right[replace_from_right]
        if use_single:
            fused[replace_from_single] = roi_single[replace_from_single]

    # ========== 12. 반환 마스크 생성 ==========
    # 실제 교체된 영역
    replaced_mask = replace_from_right | replace_from_single
    
    # 빛반사가 감지된 영역 전체 반환 (디버그용: 빨간색으로 표시)

    return fused, replaced_mask, specular_mask


def fuse_reflection_free_image(
    imgs: List[str],
    bbox_output: List,
    serial_number: Optional[str] = None,
    depth_instance: Optional[Any] = None,
) -> Optional[np.ndarray]:
    """
    여러 bbox에 대해 빛반사 제거 퓨전 수행 (간소화된 인터페이스)

    자동 처리 항목:
        - depth: Depth 인스턴스에서 가져옴 (last_depth_rect - rectified 좌표계 768x480)
        - 캘리브레이션: calibration_manager의 serial_calibration_data에서 자동으로 가져옴
        - 이미지 로드 및 bbox/contour 파싱을 내부에서 처리
        - 디버그 이미지 자동 저장 (교체된 영역: 파란색, 후보지만 못한 영역: 빨간색)

    Args:
        imgs: 이미지 경로 리스트
            - imgs[0]: depth 경로 (사용 안 함, 호환성 유지용)
            - imgs[1]: Left 카메라 이미지 경로
            - imgs[2]: Right 카메라 이미지 경로
            - imgs[3]: Single 카메라 이미지 경로 (선택적)
        bbox_output: Detection 결과 리스트
            - 각 요소: [x1, y1, x2, y2, cls, score, valid, contour, ...]
            - valid=True인 bbox만 처리
            - contour는 선택적 (있으면 해당 영역만 처리)
        serial_number: 시리얼 번호 (None이면 첫 번째 유효한 캘리브레이션 사용)
        depth_instance: Depth 클래스 인스턴스 (last_depth_rect, 크롭 파라미터 속성 필요)

    Returns:
        빛반사가 제거된 전체 이미지 (H, W, 3) uint8, 실패 시 None
    """
    # 성능 측정 시작
    start_time = time.time()
    import common_config as cc

    # ========== 1. Depth 맵 및 크롭 파라미터 가져오기 (Depth 인스턴스에서) ==========
    if depth_instance is None:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] depth_instance가 없어서 fusion 스킵")
        return None

    # rectified 좌표계의 depth 사용 (offset 보정 전, 768x480)
    depth_rect_raw = getattr(depth_instance, 'last_depth_rect', None)
    if depth_rect_raw is None:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] depth_instance.last_depth_rect가 없어서 fusion 스킵")
        return None

    # 크롭 파라미터 가져오기
    ori_crop_lx = getattr(depth_instance, 'ori_crop_lx', None)
    ori_crop_ly = getattr(depth_instance, 'ori_crop_ly', None)
    crop_w = getattr(depth_instance, 'crop_w', None)
    crop_h = getattr(depth_instance, 'crop_h', None)

    if None in [ori_crop_lx, ori_crop_ly, crop_w, crop_h]:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] depth_instance에 크롭 파라미터가 없어서 fusion 스킵")
        return None

    rgb_w, rgb_h = cc.image_resolution_rgb[1], cc.image_resolution_rgb[0]

    # crop_params 딕셔너리 생성
    crop_params = {
        "ori_crop_lx": ori_crop_lx,
        "ori_crop_ly": ori_crop_ly,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "rgb_w": rgb_w,
        "rgb_h": rgb_h,
    }

    # depth는 rectified 좌표계 그대로 사용 (768x480)
    depth_rect = depth_rect_raw.astype(np.float32)

    # ========== 2. Valid bbox 및 contour 추출 ==========
    if not bbox_output or len(bbox_output) == 0:
        return None

    # bbox 입력 검증 강화
    valid_bboxes = []
    invalid_bbox_count = 0
    for b in bbox_output:
        try:
            # 최소 길이 검증 (x1, y1, x2, y2, cls, score, valid)
            if len(b) < 7:
                invalid_bbox_count += 1
                continue
            # valid 필드 검증
            if not b[6]:
                continue
            # 좌표값 검증 (숫자 타입 확인)
            if not all(isinstance(b[i], (int, float, np.integer, np.floating)) for i in range(4)):
                invalid_bbox_count += 1
                continue
            valid_bboxes.append(b)
        except (IndexError, TypeError) as e:
            invalid_bbox_count += 1
            continue

    if len(valid_bboxes) == 0:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] 유효한 bbox가 없어서 fusion 스킵 (전체: {len(bbox_output)}, 무효: {invalid_bbox_count})")
        return None

    # bbox 좌표 추출 (안전하게)
    bboxes = []
    for b in valid_bboxes:
        try:
            bboxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        except (IndexError, ValueError, TypeError):
            continue

    if len(bboxes) == 0:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] bbox 좌표 추출 실패")
        return None

    # contour 추출 (안전하게)
    contours = []
    for b in valid_bboxes:
        try:
            contour = b[7] if len(b) > 7 and b[7] is not None else None
            contours.append(contour)
        except (IndexError, TypeError):
            contours.append(None)

    # 컨투어 분석 로그
    contour_count = sum(1 for c in contours if c is not None)
    has_contour = contour_count > 0
    contour_status = "있음" if has_contour else "없음"
    print(f"{cc.CurrentDateTime(0)} [Artis_AI] ===============================================")

    # ========== 3. 이미지 로드 ==========
    if len(imgs) < 3:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] imgs 배열이 부족합니다 (최소 3개 필요, 현재: {len(imgs)})")
        return None

    left_bgr = cv2.imread(imgs[1])
    right_bgr = cv2.imread(imgs[2])

    if left_bgr is None or right_bgr is None:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] 필수 이미지 로드 실패 (Left: {left_bgr is not None}, Right: {right_bgr is not None})")
        return None

    # Single 이미지 (있으면 로드)
    single_bgr = None
    if len(imgs) > 3 and imgs[3]:
        single_bgr = cv2.imread(imgs[3])
        if single_bgr is None:
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] Single 이미지 로드 실패 (선택적, 계속 진행): {imgs[3]}")

    # 해상도 로그 출력
    img_h, img_w = left_bgr.shape[:2]
    depth_h, depth_w = depth_rect.shape[:2]
    single_size_str = f"{single_bgr.shape[1]}x{single_bgr.shape[0]}" if single_bgr is not None else "없음"
    print(f"{cc.CurrentDateTime(0)} [Artis_AI] RGB(Left): {img_w}x{img_h}, Depth(Rect): {depth_w}x{depth_h}, Single: {single_size_str}")

    # ========== 4. 캘리브레이션 파라미터 가져오기 ==========
    stereo_params, single_params = get_calibration_params(serial_number)

    if stereo_params is None:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] stereo_cal_params가 없어서 fusion 스킵")
        return None

    # rectify 정보 확인 (R1, P1이 있어야 함)
    if "R1" not in stereo_params or "P1" not in stereo_params:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] stereo_params에 rectify 정보(R1, P1)가 없어서 fusion 스킵")
        return None

    # ========== 5. Rectified depth 좌표 변환 확인 로그 ==========
    P1 = stereo_params["P1"]

    # ========== 6. 이미지 왜곡 보정 (선택적) ==========
    K_l = stereo_params["K_l"].astype(np.float64)
    D_l = _normalize_dist(stereo_params["dist_l"]).astype(np.float64)
    K_r = stereo_params["K_r"].astype(np.float64)
    D_r = _normalize_dist(stereo_params["dist_r"]).astype(np.float64)

    left_u = undistort_if_needed(left_bgr, K_l, D_l)
    right_u = undistort_if_needed(right_bgr, K_r, D_r)

    # Single 카메라 처리
    use_single = single_bgr is not None and single_params is not None
    if use_single:
        K_s = single_params["K_s"].astype(np.float64)
        D_s = _normalize_dist(single_params["dist_s"]).astype(np.float64)
        single_u = undistort_if_needed(single_bgr, K_s, D_s)
    else:
        single_u = None
        single_params = None
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] Single 카메라 미사용: Right 카메라만 사용")

    # ========== 7. Fusion 수행 ==========
    fused_full = left_u.copy()
    h, w = left_u.shape[:2]

    # 전체 이미지 크기의 마스크 수집용
    full_replaced_mask = np.zeros((h, w), dtype=bool)  # 교체된 영역 (디버그: 파란색)
    full_specular_mask = np.zeros((h, w), dtype=bool)  # 빛반사 감지된 영역 전체 (디버그: 빨간색)

    skipped_bbox_count = 0
    contour_mask_fail_count = 0

    for idx, bbox in enumerate(bboxes):
        roi = clip_bbox(bbox, w, h, pad=0)
        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            skipped_bbox_count += 1
            continue

        # 컨투어 마스크 생성 (있으면 해당 영역만, 없으면 전체 ROI)
        contour_mask = None
        if contours is not None and idx < len(contours) and contours[idx] is not None:
            try:
                normalized = _normalize_contour(contours[idx])
                if len(normalized) >= 3:
                    contour_mask = create_contour_mask([normalized], img_shape=(h, w), bbox=roi)
                    if contour_mask is None:
                        contour_mask_fail_count += 1
                else:
                    contour_mask_fail_count += 1
            except Exception as e:
                contour_mask_fail_count += 1
                # 폴백: bbox 전체 영역 사용

        # ROI별 fusion 수행
        fused_roi, replaced_mask_roi, specular_mask_roi = fuse_roi_reflection_free(
            left_u, right_u, depth_rect, roi,
            stereo_params, crop_params,
            single_bgr=single_u,
            single_params=single_params,
            contour_mask=contour_mask
        )

        fused_full[y1:y2, x1:x2] = fused_roi

        # 마스크 수집 (전체 이미지 크기로 누적)
        full_replaced_mask[y1:y2, x1:x2] |= replaced_mask_roi
        full_specular_mask[y1:y2, x1:x2] |= specular_mask_roi

    # ========== 전체 이미지 레벨 빛반사 감지 통계 ==========
    total_specular_px = np.sum(full_specular_mask)
    total_replaced_px = np.sum(full_replaced_mask)
    print(f"{cc.CurrentDateTime(0)} [Artis_AI] 전체 빛반사 영역: {total_specular_px}px, 교체된 영역: {total_replaced_px}px")
    if total_specular_px == 0:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] ⚠️ 빛반사 영역이 감지되지 않았습니다")

    # 스킵된 bbox 로그
    if skipped_bbox_count > 0:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] 유효하지 않은 ROI로 인해 {skipped_bbox_count}개 bbox 스킵")

    # 컨투어 마스크 생성 실패 로그
    if contour_mask_fail_count > 0:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] 컨투어 마스크 생성 실패 {contour_mask_fail_count}개 (bbox 전체 영역 사용)")

    # ========== 8. 디버그 이미지 저장 ==========
    try:
        img_dir = os.path.dirname(imgs[1])
        debug_image = fused_full.copy()

        # 빨간색: 빛반사가 감지된 영역 전체
        if np.any(full_specular_mask):
            red_overlay = debug_image.copy()
            red_overlay[full_specular_mask] = [0, 0, 255]  # BGR 빨간색
            alpha = 0.5
            debug_image = cv2.addWeighted(debug_image, 1.0 - alpha, red_overlay, alpha, 0)

        # 파란색: 실제 교체된 영역 (빨간색 위에 오버레이)
        if np.any(full_replaced_mask):
            blue_overlay = debug_image.copy()  # 원본 이미지로 복사
            blue_overlay[full_replaced_mask] = [255, 0, 0]  # BGR 파란색
            alpha = 0.5
            debug_image = cv2.addWeighted(debug_image, 1.0 - alpha, blue_overlay, alpha, 0)

        fused_path = os.path.join(img_dir, cc.artis_fusion_img_file)
        success = cv2.imwrite(fused_path, debug_image)
    except Exception as e:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] 디버그 이미지 저장 실패: {e}")

    # 성능 측정 및 로그 출력
    elapsed_time = (time.time() - start_time) * 1000
    print(f"{cc.CurrentDateTime(0)} [Artis_AI] ==============================================> 이미지 합성 완료 ({elapsed_time:.2f} ms)")

    return fused_full
