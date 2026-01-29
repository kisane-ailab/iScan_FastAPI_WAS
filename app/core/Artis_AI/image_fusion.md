# Image Fusion Module (image_fusion.py)

스테레오/멀티 카메라 시스템에서 **빛반사(Specular Reflection)** 문제를 해결하기 위한 이미지 퓨전 모듈입니다.

---

## 개요

Left, Right, Single 세 개의 카메라 이미지를 활용하여 빛반사 영역을 감지하고, 반사가 적은 카메라의 이미지로 해당 영역을 교체합니다.

### 처리 흐름

```
Left 이미지 (기준)
    ↓
빛반사 영역 감지 (HSV 기반 점수 계산)
    ↓
Right/Single 카메라로 재투영 (depth 기반)
    ↓
점수 비교 후 최적 픽셀 선택
    ↓
경계 블렌딩 및 후처리
    ↓
퓨전된 이미지 출력
```

---

## 주요 설정값 (CONFIG)

### 빛반사 감지

| 설정 | 값 | 설명 |
|------|-----|------|
| `SPECULAR_THRESHOLD` | 280 | 빛반사 판단 임계값 (점수가 이 값 이상이면 반사로 판단) |
| `SPECULAR_WEIGHTS['V']` | 0.7 | 밝기(Value) 가중치 |
| `SPECULAR_WEIGHTS['INV_S']` | 0.6 | 채도 역가중치 (채도가 낮을수록 반사 가능성 높음) |

**점수 공식**: `score = 0.7 × V + 0.6 × (255 - S)`

### 교체 조건

| 설정 | 값 | 설명 |
|------|-----|------|
| `MIN_SCORE_DIFF_RIGHT` | 5.0 | Right가 Left보다 최소 이만큼 낮아야 교체 |
| `MIN_SCORE_DIFF_SINGLE` | 10.0 | Single이 Left보다 최소 이만큼 낮아야 교체 |
| `MIN_REPLACE_AREA` | 20 | 교체할 최소 픽셀 수 (이보다 작으면 스킵, 작은 영역도 교체 가능) |

### 후처리 옵션

| 설정 | 값 | 설명 |
|------|-----|------|
| `USE_MORPHOLOGY` | True | Morphological operations으로 마스크 정리 |
| `MORPH_KERNEL_SIZE` | 3 | Morphology 커널 크기 |
| `USE_BLENDING` | True | 반사 영역 경계 블렌딩 |
| `BLEND_WIDTH` | 5 | 블렌딩 경계 폭 (픽셀) |
| `EXPAND_REPLACE_MASK` | 5 | 교체 마스크 확장 (경계가 티날 때 증가) |

---

## 캘리브레이션 데이터 구조

시리얼 넘버별로 `calibration_manager.serial_calibration_data`에 캘리브레이션 파라미터가 저장됩니다.
부팅 시 `stereoRectify` 결과(R1, P1)도 함께 계산하여 메모리에 저장하므로, 런타임에 추가 계산이 불필요합니다.

### `stereo_cal_params` 구조

```python
stereo_cal_params = {
    # 기본 파라미터
    "K_l": np.ndarray,       # Left 카메라 내부 파라미터 (3×3)
    "K_r": np.ndarray,       # Right 카메라 내부 파라미터 (3×3)
    "dist_l": np.ndarray,    # Left 왜곡 계수
    "dist_r": np.ndarray,    # Right 왜곡 계수
    "R": np.ndarray,         # Left→Right 회전 행렬 (3×3)
    "t": np.ndarray,         # Left→Right 이동 벡터 (3×1)
    "origin_size": list,     # 원본 이미지 크기 [width, height]
    # Rectify 정보 (부팅 시 미리 계산됨)
    "R1": np.ndarray,        # Left rectification rotation matrix
    "P1": np.ndarray,        # Left rectification projection matrix
    "img_size": tuple,       # 이미지 크기 (width, height)
}
```

---

## 주요 함수

### 캘리브레이션 관련

#### `get_calibration_params(serial_number=None)`
메모리에서 시리얼 넘버에 해당하는 캘리브레이션 파라미터를 가져옵니다.

- **입력**: `serial_number` - 시리얼 번호 (None이면 첫 번째 유효한 데이터 사용)
- **출력**: `(stereo_cal_params, single_cal_params)` 튜플

**시리얼 넘버별 자동 적용**: startcam 시 시리얼 넘버가 변경되면 해당 시리얼의 캘리브레이션 파라미터가 자동으로 적용됩니다.

---

### 빛반사 점수 계산

#### `specular_score_bgr(img_bgr)`
HSV 색공간 기반으로 빛반사 점수를 계산합니다.

```python
score = 0.7 × V + 0.6 × (255 - S)
```

- 밝기(V)가 높을수록 반사 가능성 높음
- 채도(S)가 낮을수록 반사 가능성 높음

---

### 재투영 맵 생성

#### `build_reproject_map_left_to_target(roi_xyxy, depth_mm, R_lt, t_lt, K_t, target_shape, stereo_params)`

Left 카메라 ROI → Target 카메라 재투영 맵을 생성합니다.

- `stereo_params`에서 R1, P1, K_l, dist_l을 직접 사용합니다.

**재투영 과정**:
1. Left 원본 좌표 → Left rectified 좌표 변환
2. Rectified 좌표에서 depth 샘플링
3. Rectified 3D 좌표로 역투영
4. Left 원본 3D 좌표로 변환
5. Target 카메라 3D 좌표로 변환
6. Target 카메라 2D 좌표로 투영

---

### ROI 단위 퓨전

#### `fuse_roi_reflection_free(...)`

Left ROI 기준으로 Right/Single 카메라에서 재투영하여 빛반사 영역만 선택적으로 교체합니다.

**처리 과정**:
1. Left ROI에서 빛반사 영역 감지
2. Right/Single 카메라로 재투영하여 해당 영역의 이미지 가져오기
3. 각 후보의 빛반사 점수 비교하여 가장 낮은 점수의 픽셀 선택
4. **Right 우선 교체**, Single은 Right로 교체되지 않은 영역에서만 고려

**출력**:
- `fused`: 빛반사가 제거된 ROI 이미지
- `specular_mask`: 감지된 모든 빛반사 영역 마스크 (디버그: 빨간색)
- `replaced_mask`: 실제 교체된 영역 마스크 (디버그: 파란색, specular_mask 내부에 표시)

---

### 전체 이미지 퓨전 (메인 인터페이스)

#### `fuse_reflection_free_image(imgs, bbox_output, serial_number=None, depth_instance=None)`

여러 bbox에 대해 빛반사 제거 퓨전을 수행하는 간소화된 인터페이스입니다.

**자동 처리 항목**:
- `depth_rect`: `depth_instance`에서 자동으로 가져옴 (없으면 depth.py의 전역 변수 사용)
- 캘리브레이션: calibration_manager의 serial_calibration_data에서 자동으로 가져옴
- 이미지 로드 및 bbox/contour 파싱을 내부에서 처리
- 디버그 이미지 자동 저장

**파라미터**:
- `imgs`: 이미지 경로 리스트
- `bbox_output`: bbox 리스트
- `serial_number`: 시리얼 번호 (선택적, None이면 첫 번째 유효한 데이터 사용)
- `depth_instance`: Depth 인스턴스 (선택적, None이면 전역 변수 사용)

**입력**:
```python
imgs = [
    depth_path,   # [0] depth 경로 (호환성 유지용, 미사용)
    left_path,    # [1] Left 카메라 이미지 경로
    right_path,   # [2] Right 카메라 이미지 경로
    single_path   # [3] Single 카메라 이미지 경로 (선택적)
]

bbox_output = [
    [x1, y1, x2, y2, cls, score, valid, contour, ...],
    ...
]
```

**출력**: 빛반사가 제거된 전체 이미지 (H, W, 3) uint8

---

## 헬퍼 함수

| 함수 | 설명 |
|------|------|
| `_normalize_dist(dist)` | 왜곡 계수를 올바른 shape으로 변환 |
| `_normalize_t(t)` | 변환 벡터를 (3, 1) shape으로 변환 |
| `clip_bbox(b, w, h, pad)` | Bounding box 클리핑 |
| `_normalize_contour(contour)` | 다양한 형태의 컨투어를 `[(x, y), ...]` 형태로 정규화 |
| `create_contour_mask(contours, img_shape, bbox)` | 컨투어로 마스크 생성 |
| `_sample_depth_bilinear(depth_rect, u, v)` | Bilinear interpolation으로 depth 값 샘플링 |
| `_orig_to_rect_uv(uu, vv, K_l, D_l, R1, P1)` | Left 원본 좌표 → Left rectified 좌표 변환 |
| `undistort_if_needed(img, K, D)` | 필요시 이미지 왜곡 보정 적용 |

---

## 디버그 출력

퓨전 완료 후 디버그 이미지가 자동 저장됩니다:
- **빨간색 오버레이**: 감지된 모든 빛반사 영역 (specular_mask)
- **파란색 오버레이**: 실제 교체된 영역 (replaced_mask, 빨간색 위에 표시)

저장 경로: `{이미지 디렉토리}/{cc.artis_fusion_img_file}`

---

## 의존성

### 내부 모듈
- `app.core.Artis_AI.camera.calibration_manager` - 캘리브레이션 데이터
- `app.core.Artis_AI.depth.Depth` - Depth 인스턴스 (last_depth_rect 속성 사용)
- `common_config` - 설정 및 유틸리티

### 외부 라이브러리
- `cv2` (OpenCV)
- `numpy`

---

## 교체 전략

1. **Right 우선 전략**: Right 카메라에서 먼저 교체 시도
2. **Single 보조**: Right로 교체되지 않은 영역에서만 Single 고려
3. **점수 차이 조건**: 단순히 낮은 점수가 아닌, 최소 점수 차이 이상이어야 교체
4. **영역 크기 필터**: 너무 작은 교체 영역은 노이즈로 간주하여 스킵
