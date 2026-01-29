import os
import numpy as np
import cv2
from typing import Tuple, Optional, List, Union

try:
    import common_config as cc
except (ImportError, AttributeError):
    raise ImportError("common_config 모듈을 찾을 수 없습니다. common_config.py가 필요합니다.")

# RIGHT 카메라 변환 관련
DV_THRESHOLD = 2.0  # rectified stereo에서 y좌표 차이 임계값
PATCH_RADIUS = 2  # Flow 추정 시 사용할 patch 반경 (1 → 2로 증가하여 노이즈에 덜 민감하게)

# SINGLE 카메라 변환 관련
BBOX_MARGIN = 0.03  # bbox 확장 마진 (3%)

# 공통
PERCENTILE_LOW = 2  # 낮은 percentile (outlier 제거용)
PERCENTILE_HIGH = 98  # 높은 percentile (outlier 제거용)
IQR_MULTIPLIER = 1.5  # IQR 기반 outlier 제거 배수


# ===========================
# 공통 함수
# ===========================
def orig_to_rect_point(x_orig: float, y_orig: float, K, dist, R_rect, P_rect) -> Tuple[float, float]:
    """원본 이미지 좌표를 rectified 이미지 좌표로 변환.
    
    Args:
        x_orig, y_orig: 원본 이미지 좌표
        K: 카메라 내부 파라미터
        dist: 왜곡 계수
        R_rect: Rectification 회전 행렬
        P_rect: Rectification 투영 행렬
    
    Returns:
        (u, v): Rectified 이미지 좌표
    """
    pts = np.array([[[float(x_orig), float(y_orig)]]], dtype=np.float32)
    rect = cv2.undistortPoints(pts, K, dist, R=R_rect, P=P_rect)
    u, v = rect[0, 0]
    return float(u), float(v)

def _remove_outliers_iqr_1d(values: np.ndarray, min_size: int = 4) -> np.ndarray:
    """1D 배열에서 IQR 기반 outlier를 제거.
    
    Args:
        values: 1D numpy 배열
        min_size: outlier 제거를 수행하기 위한 최소 샘플 수
    
    Returns:
        필터링된 배열 (outlier 제거 후에도 유효한 값이 없으면 원본 반환)
    """
    if len(values) < min_size:
        return values
    
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    if iqr > 0:
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        filtered = values[(values >= lower_bound) & (values <= upper_bound)]
        return filtered if filtered.size > 0 else values
    
    return values

def _remove_outliers_iqr_2d(
    xs: np.ndarray,
    ys: np.ndarray,
    min_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """2D 좌표 배열에서 IQR 기반 outlier를 제거.
    
    Args:
        xs: x 좌표 배열
        ys: y 좌표 배열
        min_size: outlier 제거를 수행하기 위한 최소 샘플 수
    
    Returns:
        (filtered_xs, filtered_ys): 필터링된 좌표 배열
    """
    if len(xs) < min_size:
        return xs, ys
    
    q1_x, q3_x = np.percentile(xs, [25, 75])
    iqr_x = q3_x - q1_x
    q1_y, q3_y = np.percentile(ys, [25, 75])
    iqr_y = q3_y - q1_y
    
    if iqr_x > 0 and iqr_y > 0:
        x_mask = (xs >= q1_x - IQR_MULTIPLIER * iqr_x) & (xs <= q3_x + IQR_MULTIPLIER * iqr_x)
        y_mask = (ys >= q1_y - IQR_MULTIPLIER * iqr_y) & (ys <= q3_y + IQR_MULTIPLIER * iqr_y)
        mask = x_mask & y_mask
        if np.sum(mask) >= 2:
            return xs[mask], ys[mask]
    elif iqr_x > 0:
        x_mask = (xs >= q1_x - IQR_MULTIPLIER * iqr_x) & (xs <= q3_x + IQR_MULTIPLIER * iqr_x)
        if np.sum(x_mask) >= 2:
            return xs[x_mask], ys[x_mask]
    elif iqr_y > 0:
        y_mask = (ys >= q1_y - IQR_MULTIPLIER * iqr_y) & (ys <= q3_y + IQR_MULTIPLIER * iqr_y)
        if np.sum(y_mask) >= 2:
            return xs[y_mask], ys[y_mask]
    
    return xs, ys


# ===========================
# 왼쪽 & 오른쪽 카메라 변환 관련
# ===========================
def rect_to_orig_point_bilinear(u_rect: float, v_rect: float, map_x, map_y) -> Optional[Tuple[float, float]]:
    """Rectified 이미지 좌표를 원본 이미지 좌표로 변환.
    
    Args:
        u_rect, v_rect: Rectified 이미지 좌표
        map_x, map_y: Remap 맵 (rect→orig)
    
    Returns:
        (x, y): 원본 이미지 좌표 또는 None (범위 밖인 경우)
    """
    h, w = map_x.shape
    if not (0 <= u_rect < w and 0 <= v_rect < h):
        return None

    x0 = int(np.floor(u_rect))
    y0 = int(np.floor(v_rect))
    x1 = min(w - 1, x0 + 1)
    y1 = min(h - 1, y0 + 1)

    wx = float(u_rect - x0)
    wy = float(v_rect - y0)

    # 4개 샘플(각 샘플은 원본 이미지 좌표)
    ox00, oy00 = float(map_x[y0, x0]), float(map_y[y0, x0])
    ox10, oy10 = float(map_x[y0, x1]), float(map_y[y0, x1])
    ox01, oy01 = float(map_x[y1, x0]), float(map_y[y1, x0])
    ox11, oy11 = float(map_x[y1, x1]), float(map_y[y1, x1])

    ox = (1 - wx) * (1 - wy) * ox00 + wx * (1 - wy) * ox10 + (1 - wx) * wy * ox01 + wx * wy * ox11
    oy = (1 - wx) * (1 - wy) * oy00 + wx * (1 - wy) * oy10 + (1 - wx) * wy * oy01 + wx * wy * oy11
    return float(ox), float(oy)


def sample_flow_bilinear(x: float, y: float, flow) -> Optional[Tuple[float, float]]:
    """Bilinear 보간으로 flow 샘플링 (서브픽셀 정확도).
    
    Args:
        x, y: 샘플링할 좌표 (float)
        flow: Flow map (H, W) 또는 (H, W, C)
    
    Returns:
        (du, dv): Flow 값 또는 None (범위 밖이거나 유효하지 않은 경우)
    """
    if flow.ndim == 2:
        h, w = flow.shape
        channels = 1
    elif flow.ndim == 3:
        h, w, channels = flow.shape
    else:
        return None
    
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return None
    
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    
    wx = x - x0
    wy = y - y0
    
    if channels == 1:
        f00, f10 = flow[y0, x0], flow[y0, x1]
        f01, f11 = flow[y1, x0], flow[y1, x1]
        if not all(np.isfinite([f00, f10, f01, f11])):
            return None
        val = (1-wx)*(1-wy)*f00 + wx*(1-wy)*f10 + (1-wx)*wy*f01 + wx*wy*f11
        return float(val), 0.0
    else:
        f00, f10 = flow[y0, x0, :], flow[y0, x1, :]
        f01, f11 = flow[y1, x0, :], flow[y1, x1, :]
        if not (np.isfinite(f00).all() and np.isfinite(f10).all() and 
                np.isfinite(f01).all() and np.isfinite(f11).all()):
            return None
        val = (1-wx)*(1-wy)*f00 + wx*(1-wy)*f10 + (1-wx)*wy*f01 + wx*wy*f11
        return float(val[0]), float(val[1])


def estimate_flow_at_point(pt_r: Tuple[float, float], flow, patch_radius: int = 1) -> Optional[Tuple[float, float]]:
    """Rectified Cam2 기준 포인트 주변의 flow 추정.
    
    Args:
        pt_r: (x, y) rectified 좌표
        flow: Flow map (H, W) 또는 (H, W, 2)
        patch_radius: Patch 반경
    
    Returns:
        (du, dv): Flow 값 또는 None
    """
    if flow.ndim == 2:
        h, w = flow.shape
        channels = 1
    elif flow.ndim == 3:
        h, w, channels = flow.shape
    else:
        raise ValueError(f"flow 차원이 이상합니다: shape={flow.shape}")

    x, y = pt_r
    
    # bilinear 보간 우선 시도
    if patch_radius <= 1:
        result = sample_flow_bilinear(x, y, flow)
        if result is not None:
            return result
    
    # fallback: patch median 사용
    xi = int(round(x))
    yi = int(round(y))

    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return None

    x1 = max(0, xi - patch_radius)
    x2 = min(w - 1, xi + patch_radius)
    y1 = max(0, yi - patch_radius)
    y2 = min(h - 1, yi + patch_radius)

    if channels == 1:
        roi = flow[y1:y2+1, x1:x2+1]
        roi_flat = roi.reshape(-1)
        valid = roi_flat[np.isfinite(roi_flat)]
        if valid.size == 0:
            return None
        disp = float(np.median(valid))
        return disp, 0.0

    elif channels == 2:
        roi = flow[y1:y2+1, x1:x2+1, :]
        roi_flat = roi.reshape(-1, 2)
        valid_mask = np.isfinite(roi_flat).all(axis=1)
        valid = roi_flat[valid_mask]
        if valid.shape[0] == 0:
            return None
        # trimmed mean
        if valid.shape[0] >= 5:
            sorted_du = np.sort(valid[:, 0])
            sorted_dv = np.sort(valid[:, 1])
            trim = max(1, int(len(sorted_du) * 0.1))
            du = float(np.mean(sorted_du[trim:-trim]))
            dv = float(np.mean(sorted_dv[trim:-trim]))
        else:
            du = float(np.median(valid[:, 0]))
            dv = float(np.median(valid[:, 1]))
        return du, dv

    else:
        raise ValueError(f"지원하지 않는 flow 채널 수: {channels}")


def transform_bbox_cam2_to_cam1(
    bbox_cam2_orig: List[float],
    flow_left_to_right_rect,
    rect_info: dict,
    img_cam1_shape: Tuple[int, int],
) -> Optional[List[float]]:
    """Cam2(LEFT) bbox를 Cam1(RIGHT) bbox로 변환.
    
    Args:
        bbox_cam2_orig: Cam2 RAFT 좌표계의 bbox [x1, y1, x2, y2] (RAFT 해상도 기준: 768x480)
        flow_left_to_right_rect: LEFT→RIGHT rectified flow map (RAFT 해상도 기준: 768x480)
        rect_info: Rectification 정보 딕셔너리
        img_cam1_shape: Cam1 이미지 크기
    
    Returns:
        bbox_cam1: Cam1 RAFT 좌표계의 bbox [x1, y1, x2, y2] (RAFT 해상도 기준: 768x480) 또는 None
    
    Note:
        좌표계 변환(RGB ↔ RAFT)은 호출자(transform_all_bboxes)에서 처리됩니다.
        이 함수는 RAFT 좌표계 기준으로 동작합니다.
    """
    # 카메라 내부 파라미터 추출 (rect_info 검증은 호출자에서 수행)
    K_left = rect_info["mtx_l"]
    dist_left = rect_info["dist_l"]
    R1 = rect_info["R1"]
    P_left = rect_info["P1"]

    # 좌표변환 맵 추출
    map_right_x = rect_info["map2_orig_x"]
    map_right_y = rect_info["map2_orig_y"]

    # 이미지 크기 추출
    h_rect, w_rect = flow_left_to_right_rect.shape[:2]
    h1, w1 = img_cam1_shape[:2]

    # 크롭 정보 추출
    resize_crop_lx = rect_info.get("resize_crop_lx", 0)
    resize_crop_ly = rect_info.get("resize_crop_ly", 0)
    
    def raft_to_scaled_origin(x_raft, y_raft):
        """RAFT 좌표계를 스케일된 원본 좌표계(768x480)로 변환.
        
        mtx_l은 스케일된 원본 좌표계에 맞춰져 있으므로,
        orig_to_rect_point에 전달할 좌표는 스케일된 원본 좌표계여야 함.
        """
        return x_raft - resize_crop_lx, y_raft - resize_crop_ly

    # Cam2 RAFT 좌표계 bbox 추출
    x1o_raft, y1o_raft, x2o_raft, y2o_raft = bbox_cam2_orig
    w_bbox = x2o_raft - x1o_raft
    h_bbox = y2o_raft - y1o_raft
    
    # bbox 내부 픽셀 기반 샘플링 (RAFT 좌표계)
    sample_points_raft = []
    
    # 최소 샘플링 포인트 수 보장
    min_sample_points = 50
    bbox_area = w_bbox * h_bbox
    
    # bbox 크기에 따른 적응적 샘플링 간격 조정
    if bbox_area < 100:  # 매우 작은 bbox
        step = max(1, min(3, int(min(w_bbox, h_bbox) / 5)))
    elif bbox_area < 500:  # 작은 bbox
        step = max(2, min(5, int(min(w_bbox, h_bbox) / 8)))
    else:  # 일반 bbox
        step = max(3, min(10, int(min(w_bbox, h_bbox) / 10)))
    
    # 최소 샘플링 포인트 수 보장을 위한 step 재조정
    y_steps = len(list(np.arange(y1o_raft, y2o_raft + 1, step)))
    x_steps = len(list(np.arange(x1o_raft, x2o_raft + 1, step)))
    current_points = y_steps * x_steps
    
    if current_points < min_sample_points:
        # step을 줄여서 더 많은 포인트 생성
        step = max(1, int(np.sqrt(bbox_area / min_sample_points)))
    
    # bbox 내부 전체를 그리드로 샘플링 (RAFT 좌표계)
    for y in np.arange(y1o_raft, y2o_raft + 1, step):
        for x in np.arange(x1o_raft, x2o_raft + 1, step):
            sample_points_raft.append((float(x), float(y)))
    
    # 코너와 에지는 반드시 포함
    cx_raft = (x1o_raft + x2o_raft) / 2.0
    cy_raft = (y1o_raft + y2o_raft) / 2.0
    essential_points = [
        (cx_raft, cy_raft),  # 센터
        (x1o_raft, y1o_raft), (x2o_raft, y1o_raft), (x2o_raft, y2o_raft), (x1o_raft, y2o_raft),  # 코너
        (cx_raft, y1o_raft), (cx_raft, y2o_raft), (x1o_raft, cy_raft), (x2o_raft, cy_raft),  # 에지 중점
    ]
    for pt in essential_points:
        if pt not in sample_points_raft:
            sample_points_raft.append(pt)

    # RIGHT rect 대응점 계산
    pts_right_rect = []
    
    for (x_raft, y_raft) in sample_points_raft:
        # RAFT 좌표계 → 스케일된 원본 좌표계 변환 후 orig_to_rect_point 호출
        # mtx_l은 스케일된 원본 좌표계(768x480)에 맞춰진 카메라 매트릭스
        x_scaled, y_scaled = raft_to_scaled_origin(x_raft, y_raft)
        uL, vL = orig_to_rect_point(x_scaled, y_scaled, K_left, dist_left, R1, P_left)
        if not (0 <= uL < w_rect and 0 <= vL < h_rect):
            continue

        flow_vec = estimate_flow_at_point((uL, vL), flow_left_to_right_rect, patch_radius=PATCH_RADIUS)
        if flow_vec is None:
            continue
        du, dv = flow_vec
        
        # rectified stereo에서 y좌표는 동일해야 함
        if abs(dv) > DV_THRESHOLD:
            continue
        
        # Rectified stereo에서: u_right = u_left + du, v_right = v_left (epipolar constraint)
        uR = uL + du
        vR = vL

        if not (0 <= uR < w_rect and 0 <= vR < h_rect):
            continue

        pts_right_rect.append([uR, vR])

    if len(pts_right_rect) < 2:
        return None

    # RIGHT rect → RIGHT orig 변환
    # rect_to_orig_point_bilinear 결과는 스케일된 원본 좌표계(768x480)이므로
    # RAFT 좌표계로 변환하려면 오프셋만 추가하면 됨
    cam1_points_raft = []
    for uRr, vRr in pts_right_rect:
        orig_pt = rect_to_orig_point_bilinear(uRr, vRr, map_right_x, map_right_y)
        if orig_pt is None:
            continue
        x_orig_scaled, y_orig_scaled = orig_pt
        x_raft = x_orig_scaled + resize_crop_lx
        y_raft = y_orig_scaled + resize_crop_ly
        if 0 <= x_raft < w1 and 0 <= y_raft < h1:
            cam1_points_raft.append((x_raft, y_raft))

    if len(cam1_points_raft) < 2:
        return None

    xs = np.array([p[0] for p in cam1_points_raft])
    ys = np.array([p[1] for p in cam1_points_raft])

    # IQR 기반 outlier 제거
    xs, ys = _remove_outliers_iqr_2d(xs, ys)

    # Bbox 생성: 포인트 수와 bbox 크기에 따라 percentile 또는 min/max 사용
    bbox_area = w_bbox * h_bbox
    use_percentile = len(xs) >= 10 and bbox_area > 500
    
    if use_percentile:
        x1_new = max(0, int(np.floor(np.percentile(xs, PERCENTILE_LOW))))
        x2_new = min(w1 - 1, int(np.ceil(np.percentile(xs, PERCENTILE_HIGH))))
        y1_new = max(0, int(np.floor(np.percentile(ys, PERCENTILE_LOW))))
        y2_new = min(h1 - 1, int(np.ceil(np.percentile(ys, PERCENTILE_HIGH))))
    else:
        x1_new = max(0, int(np.floor(np.min(xs))))
        y1_new = max(0, int(np.floor(np.min(ys))))
        x2_new = min(w1 - 1, int(np.ceil(np.max(xs))))
        y2_new = min(h1 - 1, int(np.ceil(np.max(ys))))

    if x2_new <= x1_new or y2_new <= y1_new:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] [Right] BBox Invalid: [{x1_new}, {y1_new}, {x2_new}, {y2_new}] (Input: {bbox_cam2_orig})")
        return None

    print(f"{cc.CurrentDateTime(0)} [Artis_AI] [Right] BBox Transformed: {bbox_cam2_orig} -> [{x1_new}, {y1_new}, {x2_new}, {y2_new}]")
    return [x1_new, y1_new, x2_new, y2_new]


# ===========================
# 왼쪽 & 싱글 카메라 변환 관련
# ===========================
def _filter_depth_with_iqr(depth_valids: np.ndarray) -> float:
    """IQR 기반 필터링을 사용하여 depth 값 추정.
    
    Args:
        depth_valids: 유효한 depth 값 배열
        
    Returns:
        필터링된 depth 값 (median)
    """
    if depth_valids.size == 0:
        return 0.0
    
    if depth_valids.size <= 4:
        return float(np.median(depth_valids))
    
    q1 = np.percentile(depth_valids, 25)
    q3 = np.percentile(depth_valids, 75)
    iqr = q3 - q1
    
    if iqr > 0:
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        depth_valids_filtered = depth_valids[(depth_valids >= lower_bound) & (depth_valids <= upper_bound)]
        if depth_valids_filtered.size > 0:
            return float(np.median(depth_valids_filtered))
    
    return float(np.median(depth_valids))


def _estimate_depth_at_point(
    u_rect: float,
    v_rect: float,
    depth_map: np.ndarray,
    depth_map_raw: np.ndarray,
    w_bbox: float,
    h_bbox: float,
    patch_radius: Optional[int] = None,
    use_weighted_average: bool = False,
) -> Optional[float]:
    """Rectified 좌표계의 특정 포인트에서 depth 값을 추정.
    
    Args:
        u_rect, v_rect: Rectified 좌표계 좌표
        depth_map: 필터링된 depth map
        depth_map_raw: 원본 depth map
        w_bbox, h_bbox: bbox 크기 (patch_radius 계산용)
        patch_radius: Patch 반경 (None이면 자동 계산)
        use_weighted_average: True면 거리 역가중치 기반 가중 평균 사용
        
    Returns:
        추정된 depth 값 또는 None (유효한 depth가 없는 경우)
    """
    h_rect, w_rect = depth_map.shape[:2]
    
    if not (0 <= u_rect < w_rect and 0 <= v_rect < h_rect):
        return None
    
    ui = int(round(u_rect))
    vi = int(round(v_rect))
    
    if patch_radius is None:
        patch_radius = max(5, int(min(w_bbox, h_bbox) / 15))
    
    x1 = max(0, ui - patch_radius)
    x2 = min(w_rect - 1, ui + patch_radius)
    y1 = max(0, vi - patch_radius)
    y2 = min(h_rect - 1, vi + patch_radius)
    
    depth_patch = depth_map[y1:y2+1, x1:x2+1]
    depth_valids = depth_patch[(np.isfinite(depth_patch)) & (depth_patch > 0)]
    
    if depth_valids.size == 0:
        depth_patch_raw = depth_map_raw[y1:y2+1, x1:x2+1]
        depth_valids = depth_patch_raw[(np.isfinite(depth_patch_raw)) & (depth_patch_raw > 0)]
        if depth_valids.size == 0:
            return None
    
    if use_weighted_average and depth_valids.size > 8:
        # 거리 역가중치 기반 가중 평균
        center_y_global = (y1 + y2) / 2
        center_x_global = (x1 + x2) / 2
        weights = []
        
        for py in range(y1, y2 + 1):
            for px in range(x1, x2 + 1):
                if 0 <= py < h_rect and 0 <= px < w_rect:
                    d_val = depth_map[py, px] if depth_map[py, px] > 0 else depth_map_raw[py, px]
                    if np.isfinite(d_val) and d_val > 0:
                        dist = np.sqrt((px - center_x_global)**2 + (py - center_y_global)**2)
                        weight = 1.0 / (1.0 + dist)
                        weights.append((d_val, weight))
        
        if weights:
            depths_w = [w[0] for w in weights]
            weights_w = [w[1] for w in weights]
            return float(np.average(depths_w, weights=weights_w))
    
    return _filter_depth_with_iqr(depth_valids)


def _generate_sample_points(
    x1o: float,
    y1o: float,
    x2o: float,
    y2o: float,
) -> List[Tuple[float, float, str]]:
    """Bbox에서 샘플링 포인트 생성.
    
    Args:
        x1o, y1o, x2o, y2o: Bbox 좌표
        
    Returns:
        샘플링 포인트 리스트 [(x, y, point_type), ...]
    """
    w_bbox = x2o - x1o
    h_bbox = y2o - y1o
    cx_center = (x1o + x2o) / 2.0
    cy_center = (y1o + y2o) / 2.0
    
    sample_points = []
    sample_points.append((cx_center, cy_center, 'center'))
    sample_points.append((x1o, y1o, 'corner'))
    sample_points.append((x2o, y1o, 'corner'))
    sample_points.append((x2o, y2o, 'corner'))
    sample_points.append((x1o, y2o, 'corner'))
    
    if w_bbox > 15 and h_bbox > 15:
        grid_size = max(3, min(5, int(min(w_bbox, h_bbox) / 30)))
        for i in range(1, grid_size):
            for j in range(1, grid_size):
                x = x1o + (w_bbox * i / grid_size)
                y = y1o + (h_bbox * j / grid_size)
                sample_points.append((x, y, 'grid'))
    
    if w_bbox > 10:
        sample_points.append((cx_center, y1o, 'edge'))
        sample_points.append((cx_center, y2o, 'edge'))
    if h_bbox > 10:
        sample_points.append((x1o, cy_center, 'edge'))
        sample_points.append((x2o, cy_center, 'edge'))
    
    return sample_points


def project_point(
    P: np.ndarray,
    point_3d: Union[Tuple[float, float, float], List[float], np.ndarray],
) -> Optional[Tuple[float, float]]:
    """3D 점을 투영 행렬 P(3x4)로 이미지 좌표로 투영.
    
    Args:
        P: 투영 행렬 (3x4 numpy array)
        point_3d: 3D 점 (X, Y, Z) - Tuple, List 또는 numpy array
    
    Returns:
        (u, v): 이미지 좌표 또는 None (Z값이 0에 가까운 경우)
    """
    X, Y, Z = point_3d
    X_h = np.array([[X], [Y], [Z], [1.0]], dtype=np.float64)
    uvw = P @ X_h
    if abs(uvw[2, 0]) < 1e-9:
        return None
    u = uvw[0, 0] / uvw[2, 0]
    v = uvw[1, 0] / uvw[2, 0]
    return (float(u), float(v))


def transform_3d_point(
    P_3d: Union[Tuple[float, float, float], List[float], np.ndarray],
    R: np.ndarray,
    t: Union[np.ndarray, List[float], Tuple[float, float, float]],
) -> np.ndarray:
    """3D 포인트를 다른 카메라 좌표계로 변환(강체변환).
    
    Args:
        P_3d: 3D 포인트 (X, Y, Z) - Tuple, List 또는 numpy array
        R: 회전 행렬 (3x3 numpy array)
        t: 이동 벡터 (3x1 또는 1x3 numpy array, 또는 List/Tuple)
    
    Returns:
        변환된 3D 포인트 (numpy array, shape: (3,))
    """
    P_3d = np.asarray(P_3d).reshape(3, 1)
    t = np.asarray(t).reshape(3, 1)
    
    P_transformed = R @ P_3d + t
    return P_transformed.reshape(3)


def transform_bbox_cam2_to_single(
    bbox_cam2_orig: List[float],
    flow_left_to_right_rect,
    rect_info: dict,
    R_cam2_to_single,
    t_cam2_to_single,
    K_single,
    img_cam2_shape: Tuple[int, int],
    img_single_shape: Tuple[int, int],
    min_depth: float = 200.0,
    max_depth: float = 500.0,
    depth_map_rect: Optional[np.ndarray] = None,
) -> Optional[List[float]]:
    """Cam2(LEFT) bbox를 Single 카메라 bbox로 변환.
    
    변환 과정:
    1) depth map에서 bbox 영역의 Z값 추정
    2) 2D → 3D 역투영 (rectified 좌표계)
    3) rectified → Cam2 원본 좌표계 (R1.T 적용)
    4) Cam2 → Single 좌표계 (R_cam2_to_single, t_cam2_to_single)
    5) 3D → Single 이미지 2D 투영
    
    Args:
        bbox_cam2_orig: Cam2 RAFT 좌표계의 bbox [x1, y1, x2, y2] (RAFT 해상도 기준: 768x480)
        flow_left_to_right_rect: LEFT→RIGHT rectified flow map (RAFT 해상도 기준: 768x480)
        rect_info: Rectification 정보 딕셔너리
        R_cam2_to_single: Cam2→Single 회전 행렬
        t_cam2_to_single: Cam2→Single 이동 벡터
        K_single: Single 카메라 내부 파라미터
        img_cam2_shape: Cam2 이미지 크기
        img_single_shape: Single 이미지 크기
        min_depth: 최소 depth (mm)
        max_depth: 최대 depth (mm)
        depth_map_rect: rectified 좌표계의 depth map (None이면 좌표변환 스킵)
    
    Returns:
        bbox_single: Single RAFT 좌표계의 bbox [x1, y1, x2, y2] (RAFT 해상도 기준: 768x480) 또는 None
    """
    h_cam2, w_cam2 = img_cam2_shape[:2]
    h_single, w_single = img_single_shape[:2]

    # 카메라 내부 파라미터 추출 (rect_info 검증은 호출자에서 수행)
    K_left = rect_info["mtx_l"]
    dist_left = rect_info["dist_l"]
    R1 = rect_info["R1"]
    P1_rect = rect_info["P1"]
    R_rect_cam2 = R1

    # 저장된 depth map이 없으면 좌표변환 불가
    if depth_map_rect is None:
        return None
    
    # Cam2 원본 bbox 추출 (이미 Depth 해상도 기준)
    x1o, y1o, x2o, y2o = bbox_cam2_orig
    w_bbox = x2o - x1o
    h_bbox = y2o - y1o
    
    # 저장된 depth map 사용
    depth_map_raw = depth_map_rect
    h_rect, w_rect = depth_map_raw.shape[:2]
    
    # 샘플링 포인트 생성
    sample_points_orig = _generate_sample_points(x1o, y1o, x2o, y2o)
    cx_center = (x1o + x2o) / 2.0
    cy_center = (y1o + y2o) / 2.0
    
    # orig_to_rect_point 결과 캐싱
    orig_to_rect_cache = {}
    
    # depth map 필터링
    depth_map = depth_map_raw.copy()
    depth_map[~np.isfinite(depth_map)] = 0.0
    depth_map[depth_map <= 0] = 0.0
    depth_map[depth_map < min_depth] = 0.0
    depth_map[depth_map > max_depth] = 0.0

    # Rectified 카메라 내부 파라미터
    fx = rect_info["fx"]
    fy = rect_info["fy"]
    cx = rect_info["cx"]
    cy = rect_info["cy"]
    
    # 크롭 정보 추출
    resize_scale_x = rect_info.get("resize_scale_x", 0.4)
    resize_scale_y = rect_info.get("resize_scale_y", 0.4)
    resize_crop_lx = rect_info.get("resize_crop_lx", 0)
    resize_crop_ly = rect_info.get("resize_crop_ly", 0)
    
    def raft_to_scaled_origin(x_raft, y_raft):
        """RAFT 좌표계를 스케일된 원본 좌표계(768x480)로 변환.
        
        mtx_l은 스케일된 원본 좌표계에 맞춰져 있으므로,
        orig_to_rect_point에 전달할 좌표는 스케일된 원본 좌표계여야 함.
        """
        return x_raft - resize_crop_lx, y_raft - resize_crop_ly
    
    def origin_to_raft(x_orig, y_orig):
        """실제 원본 좌표계(1920x1200)를 RAFT 좌표계로 변환.
        
        project_point 결과가 실제 원본 좌표계로 나올 경우 사용.
        """
        x_scaled = x_orig * resize_scale_x
        y_scaled = y_orig * resize_scale_y
        return x_scaled + resize_crop_lx, y_scaled + resize_crop_ly
    
    # 중심점 처리 (가중치 기반 depth 추정)
    center_projected = None
    center_3d = None
    
    cx_raft, cy_raft = cx_center, cy_center
    cx_scaled, cy_scaled = raft_to_scaled_origin(cx_raft, cy_raft)
    u2, v2 = orig_to_rect_point(cx_scaled, cy_scaled, K_left, dist_left, R1, P1_rect)
    
    Z = _estimate_depth_at_point(
        u2, v2, depth_map, depth_map_raw, w_bbox, h_bbox,
        patch_radius=max(5, int(min(w_bbox, h_bbox) / 15)),
        use_weighted_average=True
    )
    
    if Z is not None:
        # 3D 역투영: Rectified 좌표계에서 3D 포인트 계산
        X_rect = (u2 - cx) * Z / fx
        Y_rect = (v2 - cy) * Z / fy
        P_3d_rect = np.array([X_rect, Y_rect, Z], dtype=np.float64)
        
        # Rectified → Cam2 원본 좌표계
        if R_rect_cam2 is not None:
            P_3d_cam2 = (R_rect_cam2.T @ P_3d_rect.reshape(3, 1)).reshape(3)
        else:
            P_3d_cam2 = P_3d_rect
        
        # Cam2 → Single 좌표계 변환 및 투영
        P_3d_single = transform_3d_point(P_3d_cam2, R_cam2_to_single, t_cam2_to_single)
        K_single_proj = K_single @ np.hstack([np.eye(3), np.zeros((3, 1))])
        proj_p = project_point(K_single_proj, P_3d_single)
        
        if proj_p is not None:
            up, vp = proj_p
            if np.isfinite(up) and np.isfinite(vp):
                up_raft, vp_raft = origin_to_raft(up, vp)
                center_projected = (up_raft, vp_raft)
                center_3d = P_3d_single
    
    projected_points = []
    if center_projected is not None:
        projected_points.append(center_projected)
    
    # 다른 샘플링 포인트 처리
    for cx_raft, cy_raft, point_type in sample_points_orig:
        if point_type == 'center':
            continue
        
        cx_scaled, cy_scaled = raft_to_scaled_origin(cx_raft, cy_raft)
        cache_key = (cx_scaled, cy_scaled)
        if cache_key in orig_to_rect_cache:
            u2, v2 = orig_to_rect_cache[cache_key]
        else:
            u2, v2 = orig_to_rect_point(cx_scaled, cy_scaled, K_left, dist_left, R1, P1_rect)
            orig_to_rect_cache[cache_key] = (u2, v2)
        
        if not (0 <= u2 < w_rect and 0 <= v2 < h_rect):
            continue
        
        # 포인트 타입별 patch_radius 계산
        if point_type == 'corner':
            patch_radius = max(3, int(min(w_bbox, h_bbox) / 25))
        elif point_type == 'edge':
            patch_radius = max(3, int(min(w_bbox, h_bbox) / 30))
        else:  # grid
            patch_radius = max(2, int(min(w_bbox, h_bbox) / 35))
        
        Z = _estimate_depth_at_point(
            u2, v2, depth_map, depth_map_raw, w_bbox, h_bbox, patch_radius
        )
        
        if Z is None:
            continue
        
        # 3D 역투영 및 좌표계 변환
        X_rect = (u2 - cx) * Z / fx
        Y_rect = (v2 - cy) * Z / fy
        P_3d_rect = np.array([X_rect, Y_rect, Z], dtype=np.float64)
        
        if R_rect_cam2 is not None:
            P_3d_cam2 = (R_rect_cam2.T @ P_3d_rect.reshape(3, 1)).reshape(3)
        else:
            P_3d_cam2 = P_3d_rect
        
        P_3d_single = transform_3d_point(P_3d_cam2, R_cam2_to_single, t_cam2_to_single)
        K_single_proj = K_single @ np.hstack([np.eye(3), np.zeros((3, 1))])
        proj_p = project_point(K_single_proj, P_3d_single)
        
        if proj_p is None:
            continue
        
        up, vp = proj_p
        if not np.isfinite(up) or not np.isfinite(vp):
            continue
        
        # Single 이미지 좌표를 RAFT 좌표계로 변환
        up_raft, vp_raft = origin_to_raft(up, vp)
        if not (0 <= up_raft < w_single and 0 <= vp_raft < h_single):
            up_raft = up + resize_crop_lx
            vp_raft = vp + resize_crop_ly
        
        projected_points.append((up_raft, vp_raft))
    
    if len(projected_points) < 2:
        return None

    # 투영된 포인트들의 실제 분포를 기반으로 bbox 생성
    xs = [p[0] for p in projected_points]
    ys = [p[1] for p in projected_points]
    
    # Outlier 제거 (IQR 방법)
    if len(xs) > 4:
        q1_x = np.percentile(xs, 25)
        q3_x = np.percentile(xs, 75)
        iqr_x = q3_x - q1_x
        q1_y = np.percentile(ys, 25)
        q3_y = np.percentile(ys, 75)
        iqr_y = q3_y - q1_y
        
        if iqr_x > 0 and iqr_y > 0:
            lower_x = q1_x - IQR_MULTIPLIER * iqr_x
            upper_x = q3_x + IQR_MULTIPLIER * iqr_x
            lower_y = q1_y - IQR_MULTIPLIER * iqr_y
            upper_y = q3_y + IQR_MULTIPLIER * iqr_y
            
            filtered_points = [(x, y) for x, y in projected_points 
                             if lower_x <= x <= upper_x and lower_y <= y <= upper_y]
            if len(filtered_points) >= 2:
                xs = [p[0] for p in filtered_points]
                ys = [p[1] for p in filtered_points]
                projected_points = filtered_points
    
    # percentile 사용으로 경계 outlier 방지
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    
    if len(xs_arr) >= 10:
        x_min = np.percentile(xs_arr, PERCENTILE_LOW)
        x_max = np.percentile(xs_arr, PERCENTILE_HIGH)
        y_min = np.percentile(ys_arr, PERCENTILE_LOW)
        y_max = np.percentile(ys_arr, PERCENTILE_HIGH)
    else:
        x_min = np.min(xs_arr)
        x_max = np.max(xs_arr)
        y_min = np.min(ys_arr)
        y_max = np.max(ys_arr)
    
    # 중심점이 있으면 중심점을 우선 사용하되, 실제 포인트 분포를 반영
    if center_projected is not None:
        center_x, center_y = center_projected
        
        # 중심점 기준으로 각 포인트의 offset 계산
        offsets_x = [x - center_x for x in xs]
        offsets_y = [y - center_y for y in ys]
        
        # 비대칭 확장: 각 방향별로 독립적인 offset 적용
        max_offset_x_pos = max([ox for ox in offsets_x if ox >= 0], default=0)
        max_offset_x_neg = abs(min([ox for ox in offsets_x if ox < 0], default=0))
        max_offset_y_pos = max([oy for oy in offsets_y if oy >= 0], default=0)
        max_offset_y_neg = abs(min([oy for oy in offsets_y if oy < 0], default=0))
        
        # fallback: offset이 0이면 min/max 기반으로 계산
        if max_offset_x_pos == 0 and max_offset_x_neg == 0:
            max_offset_x_pos = max_offset_x_neg = (x_max - x_min) * 0.5
        if max_offset_y_pos == 0 and max_offset_y_neg == 0:
            max_offset_y_pos = max_offset_y_neg = (y_max - y_min) * 0.5
        
        # 마진 추가
        x1s = int(center_x - max_offset_x_neg * (1 + BBOX_MARGIN))
        y1s = int(center_y - max_offset_y_neg * (1 + BBOX_MARGIN))
        x2s = int(center_x + max_offset_x_pos * (1 + BBOX_MARGIN))
        y2s = int(center_y + max_offset_y_pos * (1 + BBOX_MARGIN))
    else:
        # 중심점이 없으면 min/max 사용
        w_margin = (x_max - x_min) * BBOX_MARGIN
        h_margin = (y_max - y_min) * BBOX_MARGIN
        x1s = int(np.floor(x_min - w_margin))
        y1s = int(np.floor(y_min - h_margin))
        x2s = int(np.ceil(x_max + w_margin))
        y2s = int(np.ceil(y_max + h_margin))
    
    # 이미지 범위 검증 (범위 초과 시 실패 처리)
    if not (0 <= x1s < w_single and 0 <= y1s < h_single and 
            0 <= x2s < w_single and 0 <= y2s < h_single):
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] [Single] BBox 범위 초과: [{x1s}, {y1s}, {x2s}, {y2s}] (이미지 크기: {w_single}x{h_single}) (Input: {bbox_cam2_orig})")
        return None
    
    if x2s <= x1s or y2s <= y1s:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] [Single] BBox Invalid: [{x1s}, {y1s}, {x2s}, {y2s}] (Input: {bbox_cam2_orig})")
        return None

    print(f"{cc.CurrentDateTime(0)} [Artis_AI] [Single] BBox Transformed: {bbox_cam2_orig} -> [{x1s}, {y1s}, {x2s}, {y2s}]")
    return [x1s, y1s, x2s, y2s]


# ===========================
# 통합 좌표변환 함수
# ===========================
def transform_all_bboxes(
    left_bbox_dict: dict,
    flow_rect: np.ndarray,
    rect_info: dict,
    min_depth: float = 200.0,
    max_depth: float = 500.0,
    depth_map_rect: Optional[np.ndarray] = None,
    skip_single: bool = False,
) -> tuple[dict, dict]:
    """LEFT 기준 bbox 딕셔너리를 RIGHT와 SINGLE로 변환.
    
    Args:
        left_bbox_dict: LEFT 카메라 기준 bbox 딕셔너리 {obj_idx: [x1, y1, x2, y2], ...}
        flow_rect: LEFT→RIGHT rectified flow map (H, W, 2)
        rect_info: Rectification 정보 딕셔너리 (None이면 RIGHT 변환 스킵)
        min_depth: 최소 depth (mm) - 기본값 200.0
        max_depth: 최대 depth (mm) - 기본값 500.0
        depth_map_rect: rectified 좌표계의 depth map (optional, 있으면 재사용)
        skip_single: True이면 SINGLE 변환 스킵 (싱글 이미지가 없는 경우)
    
    Returns:
    (right_bbox_dict, single_bbox_dict): 변환된 bbox 딕셔너리들
    """
    right_bbox_dict = {}
    single_bbox_dict = {}

    if not left_bbox_dict:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] [TransformAll] LEFT bbox가 비어있어 변환 스킵")
        return right_bbox_dict, single_bbox_dict
    
    # rect_info 유효성 검사
    if rect_info is None:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] [TransformAll] rect_info가 None이어서 변환 스킵")
        return right_bbox_dict, single_bbox_dict
    
    # rect_info 필수 키 확인 (모든 하위 함수에서 사용하는 키 포함)
    required_keys = ["image_size", "mtx_l", "dist_l", "R1", "P1", "map2_orig_x", "map2_orig_y", "fx", "fy", "cx", "cy"]
    for key in required_keys:
        if key not in rect_info:
            return right_bbox_dict, single_bbox_dict
    
    # 크롭 정보 확인 (좌표변환에 필수)
    crop_keys = ["ori_crop_lx", "ori_crop_ly", "crop_w", "crop_h", "resize_scale_x", "resize_scale_y", "resize_crop_lx", "resize_crop_ly"]
    has_crop_info = all(key in rect_info for key in crop_keys)
    if not has_crop_info:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] [TransformAll] 크롭 정보가 없어서 좌표변환 스킵")
        return right_bbox_dict, single_bbox_dict
    
    # 이미지 크기 정보
    img_cam1_shape = (rect_info['image_size'][1], rect_info['image_size'][0])  # (H, W)
    img_cam2_shape = (rect_info['image_size'][1], rect_info['image_size'][0])  # (H, W)
    
    # 실제 flow_rect 해상도 확인 (RAFT 모델 실제 해상도)
    if flow_rect is not None:
        flow_rect_h, flow_rect_w = flow_rect.shape[:2]  # (H, W)
        raft_resolution_w = flow_rect_w  # 실제 RAFT width (예: 768)
        raft_resolution_h = flow_rect_h  # 실제 RAFT height (예: 480)
    else:
        # flow_rect가 없으면 rect_info의 image_size 사용
        raft_resolution_w = rect_info['image_size'][0]  # width
        raft_resolution_h = rect_info['image_size'][1]  # height
    
    # 크롭 정보 추출
    ori_crop_lx = rect_info['ori_crop_lx']
    ori_crop_ly = rect_info['ori_crop_ly']
    crop_w = rect_info['crop_w']
    crop_h = rect_info['crop_h']
    resize_scale_x = rect_info['resize_scale_x']
    resize_scale_y = rect_info['resize_scale_y']
    resize_crop_lx = rect_info['resize_crop_lx']
    resize_crop_ly = rect_info['resize_crop_ly']
    
    # RGB bbox (1280x960) → RAFT 좌표계 (768x480) 변환
    # 변환 경로: RGB → 크롭된 원본 → 원본 → 스케일된 원본 → RAFT 캔버스
    def rgb_bbox_to_raft_bbox(x_rgb, y_rgb):
        """RGB bbox 좌표를 RAFT 좌표계로 변환"""
        # 1. RGB (1280x960) → 크롭된 원본 (1600x1200)
        x_crop = x_rgb * (crop_w / cc.image_resolution_rgb[1])  # width 기준
        y_crop = y_rgb * (crop_h / cc.image_resolution_rgb[0])  # height 기준
        
        # 2. 크롭된 원본 → 원본 (1920x1200)
        x_orig = x_crop + ori_crop_lx
        y_orig = y_crop + ori_crop_ly
        
        # 3. 원본 → 스케일된 원본
        x_scaled = x_orig * resize_scale_x
        y_scaled = y_orig * resize_scale_y
        
        # 4. 스케일된 원본 → RAFT 캔버스 (오프셋 추가)
        x_raft = x_scaled + resize_crop_lx
        y_raft = y_scaled + resize_crop_ly
        
        return x_raft, y_raft
    
    def raft_bbox_to_rgb_bbox(x_raft, y_raft):
        """RAFT 좌표계를 RGB bbox 좌표로 역변환"""
        # 1. RAFT 캔버스 → 스케일된 원본 (오프셋 제거)
        x_scaled = x_raft - resize_crop_lx
        y_scaled = y_raft - resize_crop_ly
        
        # 2. 스케일된 원본 → 원본
        x_orig = x_scaled / resize_scale_x
        y_orig = y_scaled / resize_scale_y
        
        # 3. 원본 → 크롭된 원본
        x_crop = x_orig - ori_crop_lx
        y_crop = y_orig - ori_crop_ly
        
        # 4. 크롭된 원본 → RGB (1280x960)
        x_rgb = x_crop * (cc.image_resolution_rgb[1] / crop_w)  # width 기준
        y_rgb = y_crop * (cc.image_resolution_rgb[0] / crop_h)  # height 기준
        
        return x_rgb, y_rgb
    
    # RGB bbox를 RAFT 좌표계로 변환
    left_bbox_dict_scaled = {}
    for obj_idx, bbox_left in left_bbox_dict.items():
        if isinstance(bbox_left, list) and len(bbox_left) == 4:
            x1_rgb, y1_rgb, x2_rgb, y2_rgb = bbox_left
            x1_raft, y1_raft = rgb_bbox_to_raft_bbox(x1_rgb, y1_rgb)
            x2_raft, y2_raft = rgb_bbox_to_raft_bbox(x2_rgb, y2_rgb)
            left_bbox_dict_scaled[obj_idx] = [x1_raft, y1_raft, x2_raft, y2_raft]
    
    # ===========================
    # RIGHT 변환: LEFT → RIGHT (flow_rect가 유효한 경우만)
    # ===========================
    if flow_rect is not None:
        right_success_count = 0
        right_fail_count = 0
        
        for obj_idx, bbox_left_scaled in left_bbox_dict_scaled.items():
            if isinstance(bbox_left_scaled, list) and len(bbox_left_scaled) == 4:
                bbox_right_scaled = transform_bbox_cam2_to_cam1(
                    bbox_left_scaled,
                    flow_rect,
                    rect_info,
                    img_cam1_shape
                )
                
                if bbox_right_scaled is not None:
                    # RAFT 좌표계 결과를 RGB 좌표계로 역변환
                    x1_raft, y1_raft, x2_raft, y2_raft = bbox_right_scaled
                    x1_rgb, y1_rgb = raft_bbox_to_rgb_bbox(x1_raft, y1_raft)
                    x2_rgb, y2_rgb = raft_bbox_to_rgb_bbox(x2_raft, y2_raft)
                    right_bbox_dict[obj_idx] = [
                        int(x1_rgb),
                        int(y1_rgb),
                        int(x2_rgb),
                        int(y2_rgb)
                    ]
                    right_success_count += 1
                else:
                    right_fail_count += 1
        
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] [TransformAll] RIGHT 변환 완료: 성공 {right_success_count}개, 실패 {right_fail_count}개")
    else:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] [TransformAll] flow_rect가 None이어서 RIGHT 변환 스킵")
    
    # ===========================
    # SINGLE 변환: LEFT → SINGLE (메모리에서 싱글 캘리브레이션 파라미터 가져오기)
    # ===========================
    if skip_single:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] [TransformAll] 싱글 이미지가 없어서 SINGLE 변환 스킵")
    elif flow_rect is not None and depth_map_rect is not None:
        try:
            from app.core.Artis_AI.camera.calibration_manager import serial_calibration_data
            
            # 메모리에서 single_cal_params가 있는 첫 번째 항목 찾기
            single_cal_params = None
            for serial_num, cal_data in serial_calibration_data.items():
                single_cal_params = cal_data.get("single_cal_params")
                if single_cal_params:
                    break
            
            # 메모리에 파라미터가 있는 경우만 SINGLE 변환 수행
            if single_cal_params:
                single_success_count = 0
                single_fail_count = 0
                
                mtx_s = single_cal_params["mtx_s"]
                K_s = single_cal_params["K_s"]
                dist_s = single_cal_params["dist_s"]
                R_s = single_cal_params["R_s"]
                t_s = single_cal_params["t_s"]
                
                # R_s, t_s는 이미 LEFT 기준 relative pose
                R_cam2_to_single = R_s
                t_cam2_to_single = t_s
                
                # SINGLE 이미지 크기 (rect_info의 image_size 사용, 실제 SINGLE 크기와 다를 수 있음)
                img_single_shape = (rect_info['image_size'][1], rect_info['image_size'][0])  # (H, W)
                
                for obj_idx, bbox_left_scaled in left_bbox_dict_scaled.items():
                    if isinstance(bbox_left_scaled, list) and len(bbox_left_scaled) == 4:
                        bbox_single_scaled = transform_bbox_cam2_to_single(
                            bbox_left_scaled,
                            flow_rect,
                            rect_info,
                            R_cam2_to_single,
                            t_cam2_to_single,
                            K_s,
                            img_cam2_shape,
                            img_single_shape,
                            min_depth=min_depth,
                            max_depth=max_depth,
                            depth_map_rect=depth_map_rect
                        )
                        
                        if bbox_single_scaled is not None:
                            # RAFT 좌표계 결과를 RGB 좌표계로 역변환
                            x1_raft, y1_raft, x2_raft, y2_raft = bbox_single_scaled
                            x1_rgb, y1_rgb = raft_bbox_to_rgb_bbox(x1_raft, y1_raft)
                            x2_rgb, y2_rgb = raft_bbox_to_rgb_bbox(x2_raft, y2_raft)
                            single_bbox_dict[obj_idx] = [
                                int(x1_rgb),
                                int(y1_rgb),
                                int(x2_rgb),
                                int(y2_rgb)
                            ]
                            single_success_count += 1
                        else:
                            single_fail_count += 1
                
                print(f"{cc.CurrentDateTime(0)} [Artis_AI] [TransformAll] SINGLE 변환 완료: 성공 {single_success_count}개, 실패 {single_fail_count}개")
            else:
                print(f"{cc.CurrentDateTime(0)} [Artis_AI] [TransformAll] single_cal_params가 없어서 SINGLE 변환 스킵")
        except Exception as single_error:
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] [TransformAll] SINGLE 변환 중 오류 발생: {single_error}")
            import traceback
            traceback.print_exc()
    else:
        if flow_rect is None:
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] [TransformAll] flow_rect가 None이어서 SINGLE 변환 스킵")
        elif depth_map_rect is None:
            print(f"{cc.CurrentDateTime(0)} [Artis_AI] [TransformAll] depth_map_rect가 None이어서 SINGLE 변환 스킵")
    
    return right_bbox_dict, single_bbox_dict


# ===========================
# 디버깅 이미지 생성 함수
# ===========================
def create_bbox_debug_image(img_cam1_path, img_cam2_path, img_single_path,
                            left_bbox_dict, right_bbox_dict, single_bbox_dict, output_path):
    """
    좌표변환 디버깅 이미지를 생성합니다.
    """
    try:
        img_cam1 = cv2.imread(img_cam1_path) if img_cam1_path and os.path.exists(img_cam1_path) else None
        img_cam2 = cv2.imread(img_cam2_path) if img_cam2_path and os.path.exists(img_cam2_path) else None
        img_single = cv2.imread(img_single_path) if img_single_path and os.path.exists(img_single_path) else None
        
        if img_cam1 is None or img_cam2 is None:
            return False
        
        left_bbox_dict = left_bbox_dict or {}
        right_bbox_dict = right_bbox_dict or {}
        single_bbox_dict = single_bbox_dict or {}
        
        # 이미지 리스트 및 원본 크기 저장
        images = {
            'cam1': img_cam1,
            'cam2': img_cam2,
            'single': img_single
        }
        orig_sizes = {}
        for key, img in images.items():
            if img is not None:
                orig_sizes[key] = (img.shape[0], img.shape[1])  # (H, W)
        
        # 통일할 크기 결정 (최대 너비와 높이)
        max_width = max([w for h, w in orig_sizes.values()])
        max_height = max([h for h, w in orig_sizes.values()])
        
        # 이미지 리사이즈 및 스케일 계산 (비율 유지)
        resized_images = {}
        scales = {}
        
        for key, img in images.items():
            if img is None:
                resized_images[key] = None
                scales[key] = 1.0
                continue
            
            h, w = img.shape[:2]
            scale_w = max_width / w
            scale_h = max_height / h
            scale = min(scale_w, scale_h)  # 비율 유지
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 리사이즈
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 패딩 추가 (중앙 정렬)
            pad_w = max_width - new_w
            pad_h = max_height - new_h
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            
            resized = cv2.copyMakeBorder(
                resized, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            
            resized_images[key] = resized
            scales[key] = scale
        
        # bbox 좌표 조정 함수 (리사이즈된 이미지에 맞게)
        def adjust_bbox(bbox_dict, scale):
            if not bbox_dict or scale == 1.0:
                return bbox_dict
            return {
                obj_idx: [int(x * scale) for x in bbox]
                for obj_idx, bbox in bbox_dict.items()
                if isinstance(bbox, list) and len(bbox) == 4
            }
        
        # bbox 그리기 함수
        def draw_bboxes(img, bbox_dict, color):
            if img is None or not bbox_dict:
                return
            
            h, w = img.shape[:2]
            for obj_idx, bbox in bbox_dict.items():
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = [int(c) for c in bbox]
                
                # 이미지 범위 내로 클램핑
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))
                
                # 유효성 검증
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # bbox 그리기
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # 각 이미지에 bbox 그리기
        draw_bboxes(resized_images['cam2'], adjust_bbox(left_bbox_dict, scales['cam2']), (0, 0, 255))
        draw_bboxes(resized_images['cam1'], adjust_bbox(right_bbox_dict, scales['cam1']), (255, 0, 0))
        draw_bboxes(resized_images['single'], adjust_bbox(single_bbox_dict, scales['single']), (0, 255, 0))
        
        # 이미지들을 수직으로 합치기
        images_to_combine = [
            resized_images['cam1'],
            resized_images['cam2']
        ]
        if resized_images['single'] is not None:
            images_to_combine.append(resized_images['single'])
        
        combined_image = np.vstack(images_to_combine)
        cv2.imwrite(output_path, combined_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        return True
        
    except Exception as e:
        print(f"{cc.CurrentDateTime(0)} [Artis_AI] create_bbox_debug_image 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
