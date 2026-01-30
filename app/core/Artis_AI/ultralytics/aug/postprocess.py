import cv2
import numpy as np

## 루프 감지 ##
def _make_edge_mask_from_contour(cnt, pad=8, thickness=2, close_ksize=3):
    """
    contour 좌표만으로 edge(경계) 마스크 생성.
    - pad: ROI 여유
    - thickness: 폴리라인 두께(1~3 권장)
    - close_ksize: 작은 끊김을 메우기 위한 closing (0이면 생략)
    """
    cnt = as_contour(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    W = w + 2 * pad
    H = h + 2 * pad

    # ROI 좌표로 shift
    shifted = cnt.copy()
    shifted[:, 0, 0] = shifted[:, 0, 0] - x + pad
    shifted[:, 0, 1] = shifted[:, 0, 1] - y + pad

    edge = np.zeros((H, W), dtype=np.uint8)

    # contour는 폐곡선이므로 polylines로 경계 생성
    cv2.polylines(edge, [shifted], isClosed=True, color=255, thickness=thickness, lineType=cv2.LINE_8)

    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, k)

    return edge, (x - pad, y - pad)  # 원본 좌표로 되돌릴 때 쓸 offset

def _count_enclosed_background_regions(edge_mask):
    """
    edge_mask(경계=255)에서 배경(0) 연결요소를 세고,
    border(영상 가장자리)와 연결되지 않은(=폐영역) 배경 컴포넌트 개수를 반환.
    - 단순 닫힌 외곽선: 폐영역 1개
    - 도넛(외곽+내곽): 폐영역 2개 이상
    """
    bg = (edge_mask == 0).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(bg, connectivity=4)

    H, W = labels.shape
    # border에 닿는 label들을 exterior로 마킹
    border_labels = set(np.unique(np.concatenate([
        labels[0, :], labels[H-1, :], labels[:, 0], labels[:, W-1]
    ])))

    enclosed = 0
    for lab in range(1, num_labels):  # 0은 배경이지만 connectedComponents에서 0은 사실 "라벨 없음"이 아님. 여기서는 1.. 기준
        if lab in border_labels:
            continue
        enclosed += 1

    return enclosed

def contours_and_hierarchy_from_edge_mask(edge_mask, min_area=10):
    """
    edge_mask에서 RETR_TREE로 contour/hierarchy 복원.
    edge는 선이므로 area가 작게 나올 수 있어 min_area는 낮게.
    """
    cnts, hier = cv2.findContours(edge_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or hier is None:
        return [], None

    # 너무 작은 노이즈 제거
    keep = []
    keep_idx = []
    for i, c in enumerate(cnts):
        a = abs(cv2.contourArea(c))
        if a >= float(min_area):
            keep.append(c)
            keep_idx.append(i)

    if not keep:
        return [], None

    return keep, hier

def has_internal_loop(
    cnt,
    concave_thr=0.85,               # 기존 오목함 감지 유지(보조)
    pad=10,
    thickness=2,
    close_ksize=3,
    enclosed_regions_thr=2,        # 폐영역이 2개 이상이면 내부 루프(홀) 존재로 판단
    return_debug=False
):
    """
    contour 좌표만으로 내부 루프(홀) 존재 여부 판정.
    - 1) 기존 concavity(soliditiy) 기반(보조)
    - 2) contour를 edge mask로 만들고 '폐영역 개수'로 hole 판정(핵심)

    return:
      has_loop (bool)
      (옵션) debug dict
    """
    cnt = as_contour(cnt)

    # (A) 기존 오목함 기반 (보조)
    area = abs(cv2.contourArea(cnt))
    hull = cv2.convexHull(cnt)
    hull_area = abs(cv2.contourArea(hull)) + 1e-9
    concave_loop = area < float(concave_thr) * hull_area
    #print(area, hull_area, area/hull_area)

    # (B) mask 기반 폐영역(holes) 판정 (핵심)
    edge_mask, offset = _make_edge_mask_from_contour(cnt, pad=pad, thickness=thickness, close_ksize=close_ksize)
    enclosed_cnt = _count_enclosed_background_regions(edge_mask)
    hole_loop = enclosed_cnt >= int(enclosed_regions_thr)

    has_loop = bool(concave_loop or hole_loop)

    if not return_debug:
        return has_loop

    # 계층 contour도 같이 뽑아 드림(디버깅용)
    tree_contours, tree_hier = contours_and_hierarchy_from_edge_mask(edge_mask, min_area=10)

    debug = {
        "area": float(area),
        "hull_area": float(hull_area),
        "concave_loop": bool(concave_loop),
        "enclosed_regions": int(enclosed_cnt),
        "hole_loop": bool(hole_loop),
        "edge_mask_shape": tuple(edge_mask.shape),
        "offset_xy": offset,
        "tree_contours_count": len(tree_contours),
        "tree_hierarchy_is_none": (tree_hier is None),
    }
    return has_loop, debug, edge_mask, tree_contours, tree_hier


## 구멍 채우기 ##
def visualize_contours_and_hull(contours, group_indices, hull, canvas_shape=None, margin=40):
    """
    contour만으로도 보이도록 로컬 캔버스 생성.
    canvas_shape=(H,W,3) 알면 그대로 사용 가능.
    """
    all_pts = np.vstack([contours[i].reshape(-1,2) for i in group_indices]).astype(np.int32)
    x, y, w, h = cv2.boundingRect(all_pts.reshape(-1,1,2))

    if canvas_shape is None:
        W = w + 2 * margin
        H = h + 2 * margin
        offset = np.array([margin - x, margin - y], dtype=np.int32)
        canvas = np.zeros((H, W, 3), np.uint8)
    else:
        canvas = np.zeros(canvas_shape, np.uint8)
        offset = np.array([0, 0], dtype=np.int32)

    # 그룹 contour(분홍)
    for i in group_indices:
        c = (contours[i].reshape(-1,2) + offset).reshape(-1,1,2)
        cv2.polylines(canvas, [c], True, (255, 0, 255), 2)

    # hull(노랑)
    if hull is not None:
        h2 = (hull.reshape(-1,2) + offset).reshape(-1,1,2)
        cv2.polylines(canvas, [h2], True, (0, 255, 255), 3)

    return canvas

def contour_area(cnt):
    return float(abs(cv2.contourArea(as_contour(cnt))))

def contour_bbox(cnt):
    x, y, w, h = cv2.boundingRect(as_contour(cnt))
    return (x, y, x + w, y + h)

def contour_centroid(cnt):
    cnt = as_contour(cnt)
    m = cv2.moments(cnt)
    if abs(m["m00"]) < 1e-9:
        xy = cnt.reshape(-1, 2).astype(np.float32)
        c = xy.mean(axis=0)
        return float(c[0]), float(c[1])
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])

def points_in_contour_ratio(region_cnt, pts_xy, margin_px=2.0):
    """
    pts_xy: (N,2) float/int
    region_cnt 내부(경계 포함, margin 허용)인 비율
    """
    region = as_contour(region_cnt)
    inside = 0
    for p in pts_xy:
        d = cv2.pointPolygonTest(region, (float(p[0]), float(p[1])), True)
        if d >= -float(margin_px):
            inside += 1
    return inside / max(1, len(pts_xy))

def is_seed_contained_by_candidate(seed_cnt, cand_cnt,
                                  use_seed_hull=True,
                                  bbox_pad=10,
                                  sample_step=3,
                                  ratio_thr=0.95,
                                  margin_px=2.0):
    """
    cand가 seed를 '감싸는지' 판정.
    - 1차: cand_bbox가 seed_bbox(또는 seed_hull_bbox)를 포함하면 후보
    - 2차: seed(또는 seed_hull) 포인트들이 cand 내부에 ratio_thr 이상이면 contained
    """
    seed = as_contour(seed_cnt)
    cand = as_contour(cand_cnt)

    seed_region = cv2.convexHull(seed) if use_seed_hull else seed
    seed_bbox = contour_bbox(seed_region)
    cand_bbox = contour_bbox(cand)

    if not bbox_contains(cand_bbox, seed_bbox, pad=bbox_pad):
        return False

    seed_xy = seed_region.reshape(-1, 2).astype(np.float32)
    if sample_step > 1:
        seed_xy = seed_xy[::sample_step]

    r = points_in_contour_ratio(cand, seed_xy, margin_px=margin_px)
    return r >= float(ratio_thr)


def bbox_contains(outer, inner, pad=0):
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    ox1 -= pad; oy1 -= pad; ox2 += pad; oy2 += pad
    return (ox1 <= ix1) and (oy1 <= iy1) and (ox2 >= ix2) and (oy2 >= iy2)

def union_bbox(b1, b2, pad=0):
    x1 = min(b1[0], b2[0]) - pad
    y1 = min(b1[1], b2[1]) - pad
    x2 = max(b1[2], b2[2]) + pad
    y2 = max(b1[3], b2[3]) + pad
    return x1, y1, x2, y2

def inside_ratio_by_points(seed_cnt, cand_cnt, sample_step=2, margin=0.0):
    """
    cand의 점들 중 seed 내부(>= -margin)인 점 비율을 반환.
    - sample_step: 1이면 모든 점, 2~6이면 다운샘플링(속도↑)
    - margin: 경계 허용 오차(px). 예: margin=1.5면 경계 밖 1.5px까지 내부로 인정
      (OpenCV pointPolygonTest는 measureDist=True일 때 거리 반환이지만,
       여기서는 measureDist=False이므로 margin 사용을 위해 measureDist=True로 계산)
    """
    seed = as_contour(seed_cnt)
    cand_xy = as_contour(cand_cnt).reshape(-1, 2).astype(np.float32)

    if len(cand_xy) == 0:
        return 0.0

    if sample_step > 1:
        cand_xy = cand_xy[::sample_step]

    inside = 0
    total = len(cand_xy)

    # margin을 쓰려면 measureDist=True로 signed distance를 받는 게 가장 깔끔합니다.
    use_dist = (margin > 0)

    for p in cand_xy:
        if use_dist:
            d = cv2.pointPolygonTest(seed, (float(p[0]), float(p[1])), True)  # signed distance
            if d >= -float(margin):
                inside += 1
        else:
            v = cv2.pointPolygonTest(seed, (float(p[0]), float(p[1])), False) # -1/0/1
            if v >= 0:
                inside += 1

    return inside / float(total)

def point_in_contour(cnt, pt):
    cnt = as_contour(cnt)
    return cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), False) >= 0

def point_in_contour_margin(cnt, pt, margin_px=1.5):
    """
    margin_px: 경계 밖으로 margin_px까지는 내부로 인정
    """
    cnt = as_contour(cnt)
    d = cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), True)  # signed distance
    return d >= -float(margin_px), d  # (bool, signed distance)

def point_in_loop_region(seed_cnt, pt, mode="hull", margin_px=1.5):
    """
    mode:
      - "seed": 원본 seed polygon 내부(문제 있었던 방식)
      - "hull": seed convex hull 내부(루프/오목에서 안정적)
    """
    seed = as_contour(seed_cnt)
    if mode == "hull":
        region = cv2.convexHull(seed)
    else:
        region = seed
    inside, dist = point_in_contour_margin(region, pt, margin_px=margin_px)
    return inside, dist, as_contour(region)

def outside_points_of_seed(seed_cnt, cand_cnt, sample_step=1):
    """
    cand 점들 중 seed 밖인 점만 추출
    """
    seed_cnt = as_contour(seed_cnt)
    cand_xy = as_contour(cand_cnt).reshape(-1, 2).astype(np.int32)
    if sample_step > 1:
        cand_xy = cand_xy[::sample_step]

    outs = []
    for p in cand_xy:
        if cv2.pointPolygonTest(seed_cnt, (float(p[0]), float(p[1])), False) < 0:
            outs.append(p)
    if not outs:
        return None
    return np.asarray(outs, dtype=np.int32)

def wrapping_hull_contour(
    raw_inputs, seed_idx,

    # "외부/약한 겹침" 제외 기준
    min_inside_ratio=0.10,   # x% (cand가 seed 내부에 포함되는 비율이 이 값 이하이면 제외)

    # "감싸는 contour" 제외 기준
    contain_bbox_pad=10,
    contain_check_seed_centroid=True,

    # outside 포인트 샘플링(속도/정확도)
    outside_sample_step=1,

    # 최종 다각형 각짐 완화(선택)
    approx_eps_ratio=0.003,

    img_w=1200, img_h=960
):
    """
    Exclude:
      - except_indices
      - disjoint or weak-touch: inside_ratio <= x%
      - contains(seed) (감싸는 contour)
      - larger-than-seed (면적 기준)

    Include:
      - centroid가 seed 내부인 contour만 포함
      - 포함 contour의 seed 외부 포인트는 hull 입력에 추가(곡선이 그 포인트 따라가게)
    """

    contours = []
    except_indices = []
    for idx, raw in enumerate(raw_inputs):
        contours.append(raw[7])
        if int(raw[4]) in [9999997]:
            except_indices.append(idx)
    except_indices = set(except_indices)

    seed = as_contour(contours[seed_idx])
    seed_bbox = contour_bbox(seed)
    seed_cent = contour_centroid(seed)

    used = [seed_idx]
    merged_pts = [seed.reshape(-1, 2)]  # 기본 seed 점들

    for j, cand in enumerate(contours):
        if j == seed_idx:
            continue
        if j in except_indices:
            continue

        cand = as_contour(cand)
        cand_bbox = contour_bbox(cand)

        # 1) "루프 contour를 감싸는 contour"만 제외 (면적 조건 제거)
        if is_seed_contained_by_candidate(
                seed_cnt=seed,
                cand_cnt=cand,
                use_seed_hull=True,  # 보정 전 seed 문제 완화
                bbox_pad=10,
                sample_step=3,
                ratio_thr=0.95,
                margin_px=2.0
        ):
            continue

        if bbox_contains(cand_bbox, seed_bbox, pad=contain_bbox_pad):
            # bbox상으로 seed를 감쌀 가능성이 매우 큼
            ##print(seed_idx, j, "bbox_contains FALSE")
            if (not contain_check_seed_centroid) or point_in_contour(cand, seed_cent):
                continue

        # 2) "외부(분리 or 약한 접촉)" 제외: inside_ratio <= x%
        inside_ratio = inside_ratio_by_points(seed, cand, sample_step=3, margin=1.5)
        ##print(seed_idx, j, "inside_ratio", inside_ratio)
        if inside_ratio <= float(min_inside_ratio):
            # 완전 분리도 inside_ratio=0으로 여기서 걸림
            continue

        # 3) 포함 조건: centroid가 seed 내부인 것만
        cand_cent = contour_centroid(cand)
        inside, dist, _ = point_in_loop_region(seed, cand_cent, mode="hull", margin_px=2.0)
        ##print(seed_idx, j, "contour_centroid", cand_cent, inside, dist)
        #if not point_in_contour(seed, cand_cent):
        if not inside:
            continue

        # 포함 확정
        used.append(j)

        # 4) 포함 contour가 seed 밖으로 나간 포인트가 있으면 그 포인트로 hull 보정
        outs = outside_points_of_seed(seed, cand, sample_step=outside_sample_step)
        ##print(seed_idx, j, "outside_points_of_seed", len(outs) if outs is not None else "None")
        if outs is not None and len(outs) > 0:
            merged_pts.append(outs)
        # 완전히 내부면 hull에 영향 없으니 추가 안 함

    merged_xy = np.vstack(merged_pts).astype(np.int32).reshape(-1, 1, 2)

    raw_hull = cv2.convexHull(merged_xy, returnPoints=True)
    raw_hull = as_contour(raw_hull)

    x1, y1, w, h = cv2.boundingRect(raw_hull)
    x2, y2 = min(x1 + w, img_w), min(y1 + h, img_h)

    '''eps = approx_eps_ratio * cv2.arcLength(raw_hull, True)
    refined = cv2.approxPolyDP(raw_hull, eps, True)
    refined = as_contour(refined)

    return raw_hull, refined, used'''
    return raw_hull.tolist(), x1, y1, x2, y2, used

## 구멍 채우기

## Spike 검출 및 날리기
def as_contour(contour):
    """입력이 리스트든 ndarray든 (N,1,2) int32 contour로 정규화"""
    contour = np.asarray(contour)
    if contour.ndim == 2 and contour.shape[1] == 2:      # (N,2)
        contour = contour.reshape(-1, 1, 2)
    return contour.astype(np.int32)

def contour_to_filled_mask(cnt, pad=12, draw_thickness=2, close_ksize=3):
    """
    contour -> polylines(edge) -> floodFill -> filled mask (ROI 좌표계)
    self-intersection/꼬임에서 drawContours(thickness=-1)보다 안정적.
    """
    cnt = as_contour(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    W = w + 2 * pad
    H = h + 2 * pad

    shifted = cnt.copy()
    shifted[:, 0, 0] = shifted[:, 0, 0] - x + pad
    shifted[:, 0, 1] = shifted[:, 0, 1] - y + pad

    edge = np.zeros((H, W), dtype=np.uint8)
    cv2.polylines(edge, [shifted], isClosed=True, color=255,
                  thickness=draw_thickness, lineType=cv2.LINE_8)

    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, k)

    ff = edge.copy()
    flood_mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
    cv2.floodFill(ff, flood_mask, seedPoint=(0, 0), newVal=128)

    outside = (ff == 128)
    filled = np.zeros_like(edge)
    filled[~outside] = 255  # 내부+경계

    # 원본 좌표로 복원할 때 쓸 offset (ROI->orig: +offset)
    offset = (x - pad, y - pad)
    return filled, offset

def morphology_detach_tail(mask,
                           open_ksize=7, open_iters=1,
                           erode_ksize=3, erode_iters=1,
                           post_close_ksize=3, post_close_iters=1):
    """
    thick tail이 main에 흡수되는 것을 막기 위한 모폴로지:
    - OPEN: 얇은 목/돌기 제거
    - ERODE: 목이 두꺼워도 더 끊기게(주의: main도 줄어듦)
    - POST CLOSE: main의 작은 틈 복구(선택)
    """
    out = mask.copy()

    if open_ksize and open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=int(open_iters))

    if erode_ksize and erode_ksize > 1 and erode_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
        out = cv2.erode(out, k, iterations=int(erode_iters))

    if post_close_ksize and post_close_ksize > 1 and post_close_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (post_close_ksize, post_close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=int(post_close_iters))

    return out

def filter_contours_by_area(contours, min_area=200):
    if not contours:
        return []

    # 모든 contour 면적 계산 (절댓값)
    areas = [abs(cv2.contourArea(c)) for c in contours]

    keep = []
    for c, a in zip(contours, areas):
        if a >= min_area:
            #print("filter_contours_by_area", min_area, a)
            keep.append(c)

    return keep

def merge_contours_to_single(contours, shape_hw,
                             force_single=True,
                             approx_eps_ratio=0.002):
    """
    유지된 contours를 하나의 contour로 합치기:
    - contours를 union mask로 합친 후 RETR_EXTERNAL로 외곽선 추출
    - 외곽선이 여러 개면:
        force_single=True  -> 모든 외곽점의 convex hull로 단일 contour 생성(가장 안정적으로 1개 보장)
        force_single=False -> 가장 큰 외곽선 1개 선택
    """
    H, W = shape_hw
    union = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(union, contours, -1, 255, thickness=-1)

    ext, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ext:
        return None, union

    if len(ext) == 1:
        merged = ext[0]
    else:
        if force_single:
            pts = np.vstack([c.reshape(-1, 2) for c in ext]).reshape(-1, 1, 2).astype(np.int32)
            merged = cv2.convexHull(pts)
        else:
            areas = [abs(cv2.contourArea(c)) for c in ext]
            merged = ext[int(np.argmax(areas))]

    merged = as_contour(merged)

    if approx_eps_ratio and approx_eps_ratio > 0 and len(merged) >= 5:
        eps = float(approx_eps_ratio) * cv2.arcLength(merged, True)
        merged = cv2.approxPolyDP(merged, eps, True)
        merged = as_contour(merged)

    return merged, union

def spike_check_by_merged_contour(original_cnt, merged_cnt_roi, offset_xy,
                                 outside_margin_px=2.0,
                                 spike_ratio_thr=0.02,
                                 spike_count_thr=8,
                                 spike_maxdist_thr=8.0):
    """
    merged contour(ROI 좌표계) 기준으로 원본 contour 포인트가 외부에 존재하는지 검사
    """
    original_cnt = as_contour(original_cnt)
    pts = original_cnt.reshape(-1, 2).astype(np.float32)

    ox, oy = offset_xy
    pts_roi = pts - np.array([ox, oy], dtype=np.float32)

    merged = as_contour(merged_cnt_roi)

    signed = np.array([cv2.pointPolygonTest(merged, (float(p[0]), float(p[1])), True)
                       for p in pts_roi], dtype=np.float32)

    outside = signed < -float(outside_margin_px)
    out_idx = np.where(outside)[0]
    out_cnt = int(len(out_idx))
    n = int(len(pts))

    if out_cnt == 0:
        return False, {
            "outside_count": 0,
            "outside_ratio": 0.0,
            "outside_maxdist": 0.0,
            "n_points": n
        }

    out_dist = (-signed[outside]).astype(np.float32)
    out_ratio = out_cnt / max(1, n)
    out_maxdist = float(np.max(out_dist))

    is_spike = (out_ratio >= float(spike_ratio_thr)) or \
               (out_cnt >= int(spike_count_thr)) or \
               (out_maxdist >= float(spike_maxdist_thr))

    return bool(is_spike), {
        "outside_count": out_cnt,
        "outside_ratio": float(out_ratio),
        "outside_maxdist": float(out_maxdist),
        "n_points": n
    }

def roi_to_original(cnt_roi, offset_xy):
    cnt_roi = as_contour(cnt_roi)
    ox, oy = offset_xy
    out = cnt_roi.copy()
    out[:, 0, 0] += int(ox)
    out[:, 0, 1] += int(oy)
    return out

def detect_spike_and_get_merged_contour(
    original_cnt,

    # mask 생성
    pad=12,
    draw_thickness=2,
    close_ksize=3,

    # tail 흡수 방지 모폴로지 (중요)
    open_ksize=7, open_iters=1,
    erode_ksize=3, erode_iters=1,
    post_close_ksize=3, post_close_iters=1,

    # 계층 contour 필터링
    min_keep_area=200,

    # merge
    force_single=True,
    merge_approx_eps_ratio=0.002,

    # spike 판정
    outside_margin_px=2.0,
    spike_ratio_thr=0.02,
    spike_count_thr=8,
    spike_maxdist_thr=8.0,

    return_debug=False
):
    # 1) mask
    filled, offset = contour_to_filled_mask(
        original_cnt, pad=pad, draw_thickness=draw_thickness, close_ksize=close_ksize
    )

    # 2) morphology(두꺼운 tail이 내부로 흡수되는 것 방지)
    morphed = morphology_detach_tail(
        filled,
        open_ksize=open_ksize, open_iters=open_iters,
        erode_ksize=erode_ksize, erode_iters=erode_iters,
        post_close_ksize=post_close_ksize, post_close_iters=post_close_iters
    )

    # 3) hierarchy contours
    cnts, hier = cv2.findContours(morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        info = {"reason": "no_contours_from_mask", "offset": offset}
        if return_debug:
            return False, info, original_cnt, filled, morphed, None, hier
        return False, info, original_cnt

    kept = filter_contours_by_area(cnts, min_area=min_keep_area)
    if not kept:
        info = {"reason": "no_contours_over_min_keep_area", "offset": offset, "min_keep_area": float(min_keep_area)}
        if return_debug:
            return False, info, original_cnt, filled, morphed, None, hier
        return False, info, original_cnt
    elif len(kept) > 2:
        info = {"reason": "has_loop", "offset": offset, "min_keep_area": float(min_keep_area)}
        if return_debug:
            return False, info, original_cnt, filled, morphed, None, hier
        return False, info, original_cnt

    # 4) merge
    merged_roi, union_mask = merge_contours_to_single(
        kept, morphed.shape, force_single=force_single, approx_eps_ratio=merge_approx_eps_ratio
    )
    if merged_roi is None:
        info = {"reason": "merge_failed", "offset": offset}
        if return_debug:
            return False, info, original_cnt, filled, morphed, union_mask, hier
        return False, info, original_cnt

    # 5) spike 판정 (merged 기준 외부 포인트)
    is_spike, stats = spike_check_by_merged_contour(
        original_cnt, merged_roi, offset,
        outside_margin_px=outside_margin_px,
        spike_ratio_thr=spike_ratio_thr,
        spike_count_thr=spike_count_thr,
        spike_maxdist_thr=spike_maxdist_thr
    )

    merged_original = roi_to_original(merged_roi, offset)

    info = {
        "offset": offset,
        "kept_count": int(len(kept)),
        "kept_min_area": float(min_keep_area),
        "force_single": bool(force_single),
        "merged_area": float(abs(cv2.contourArea(merged_roi))),
        **stats
    }

    if return_debug:
        return is_spike, info, merged_original, filled, morphed, union_mask, hier

    return is_spike, info, merged_original
## Spike 검출 및 날리기

def wrapper_detect_spike(cnt):
    is_spike, info, merged_cnt = detect_spike_and_get_merged_contour(
        cnt,
        open_ksize=5, open_iters=1,
        erode_ksize=3, erode_iters=1,
        post_close_ksize=3, post_close_iters=1,
        min_keep_area=500,
        outside_margin_px=2.0,
        spike_ratio_thr=0.05,
        spike_count_thr=6,
        spike_maxdist_thr=8.0,
        return_debug=False
    )

    return is_spike, info, as_contour(merged_cnt).tolist()

