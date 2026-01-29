# DepthValidation 모듈 설명서

## 개요

`depth_validation.py`는 Depth 센서 데이터를 기반으로 객체 검사를 수행하는 모듈입니다. RGB 이미지에서 검출된 객체(bbox)와 Depth 이미지를 분석하여 비정상적인 상황을 감지합니다.

## 클래스 구조

```
DepthValidation
├── __init__(config_file_path)
├── load_depth_class_info()
├── init_depth_check(json_config)
├── check_validation(bbox, depth_object, result_json, config)
├── _check_inference_result(bbox, depth, min_depth)
├── 좌표 변환 함수
│   ├── _convert_bbox_coordinates(bbox_list, to_depth)  # 통합 함수
│   ├── _convert_rgb_to_depth_bbox(bbox_list)           # 래퍼
│   └── _convert_depth_to_rgb_bbox(bbox_list)           # 래퍼
├── 검사 함수
│   ├── _detect_max_depth(depth, min_depth)
│   ├── _detect_standing_objects(depth, bbox_list)
│   └── _detect_overlapped_object(bbox_list, depth_array)
└── 유틸리티 함수
    ├── _calculate_iou(box1, box2)
    ├── _calculate_iom(box1, box2)
    ├── _check_bbox_overlap_by_iou_iom(bbox_list, depth_array)
    └── make_error_code(result_json, error_code_current, error_reason_current)
```

## 주요 기능

### 1. 카메라 근접 물체 검출 (`max_depth_check`)

**목적**: 카메라에 너무 가까운 물체를 감지

**동작 방식**:
1. Depth 이미지에서 가장자리 100px 마진을 제외한 중앙 영역 분석
2. `min_depth` 이하인 픽셀의 비율 계산
3. 비율이 20% 이상이면 근접 물체로 판정
4. OpenCV `findContours`로 가장 큰 영역의 bbox 반환

**설정**:
```json
{
  "depth_check": {
    "max_depth_check": {
      "use": true
    }
  }
}
```

### 2. 세워진 사물 검출 (`stand_item_check`)

**목적**: 선반 위에 세워져 있는 물체 감지 (예: 세워진 음료병)

**동작 방식**:
1. 각 bbox 영역의 Depth 값 분석
2. 기준 거리(`depth_reference_distance`)에서 일정 depth 이상 떨어진 픽셀 계산
3. 다음 조건 충족 시 "세워진 물체"로 판정:
   - 면적 비율 50% 이상 + 픽셀 수 8000개 이하 + bbox 크기 200px 이하
   - 또는 픽셀 수 초과 시 depth 높이 20cm 이상 + 픽셀 수 30000개 미만

**임계값**:
| 상수 | 값 | 설명 |
|------|-----|------|
| MAX_PIXEL_COUNT | 8000 | 픽셀 수 임계값 |
| MAX_DEPTH | 100mm | 최대 depth 임계값 (10cm) |
| MIN_DEPTH | 30mm | 최소 depth 임계값 (3cm) |
| MIN_RATIO | 0.5 | 면적 비율 임계값 (50%) |
| MAX_BBOX_SIZE | 200px | bbox 최대 크기 |

**설정**:
```json
{
  "depth_check": {
    "stand_item_check": {
      "use": true
    }
  }
}
```

### 3. 겹친 사물 검출 (`overlap_item_check`)

**목적**: 물체가 다른 물체 위에 겹쳐져 있는 상황 감지

**동작 방식**:

#### 단계 1: IOU/IOM 기반 검사
1. 모든 bbox 쌍에 대해 IOU(Intersection over Union) 및 IOM(Intersection over Minimum) 계산
2. IOU 또는 IOM이 임계값(`iou_threshold`) 초과 시 교집합 영역 분석
3. 교집합 영역에서 유효한 Depth 비율이 70% 초과 시 겹침으로 판정
4. 겹친 bbox들을 하나의 큰 bbox로 병합

#### 단계 2: Depth 특성값 기반 검사
1. IOU/IOM에서 검출되지 않은 bbox에 대해 추가 검사
2. 클래스별 기준 Depth 값(median, range)과 실제 측정값 비교
3. median 차이와 range 차이가 모두 `depth_threshold` 초과 시 겹침으로 판정

**설정**:
```json
{
  "depth_check": {
    "overlap_item_check": {
      "use": true,
      "iou_threshold": 0.3,
      "depth_threshold": 20
    }
  }
}
```

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| iou_threshold | 0.3 | 0.0 ~ 0.9 | IOU/IOM 겹침 판정 임계값 |
| depth_threshold | 20mm | 0 ~ 100 | Depth 차이 임계값 |

## 처리 흐름

```
check_validation() 호출
        │
        ▼
init_depth_check()로 설정 로드
        │
        ▼
_check_inference_result() 실행
        │
        ├── [1] _detect_max_depth()
        │       └── 근접 물체 검출 시 → 에러 반환
        │
        ├── [2] _detect_standing_objects()
        │       └── 세워진 물체 검출 시 → 에러 반환
        │
        └── [3] _detect_overlapped_object()
                ├── _check_bbox_overlap_by_iou_iom()
                └── Depth 특성값 기반 검사
                        └── 겹친 물체 검출 시 → 에러 반환
        │
        ▼
make_error_code()로 에러 정보 업데이트
        │
        ▼
result_json 반환
```

## 좌표 변환

RGB 이미지와 Depth 이미지의 해상도가 다르기 때문에 좌표 변환이 필요합니다.

### 통합 함수

`_convert_bbox_coordinates(bbox_list, to_depth)` 함수가 양방향 변환을 처리합니다:

```python
def _convert_bbox_coordinates(self, bbox_list, to_depth=True):
    """
    Args:
        bbox_list: 변환할 bbox 리스트
        to_depth: True면 RGB→Depth, False면 Depth→RGB
    """
```

### 래퍼 함수

기존 호환성을 위해 래퍼 함수를 제공합니다:

```python
# RGB → Depth 변환
_convert_rgb_to_depth_bbox(bbox_list)  # 내부적으로 _convert_bbox_coordinates(bbox_list, to_depth=True) 호출

# Depth → RGB 변환
_convert_depth_to_rgb_bbox(bbox_list)  # 내부적으로 _convert_bbox_coordinates(bbox_list, to_depth=False) 호출
```

### 변환 공식

```python
# 소스 → 대상 좌표 변환
new_x = int(x * dst_w / src_w)
new_y = int(y * dst_h / src_h)
```

## 에러 코드

검출된 에러는 다음과 같은 코드로 반환됩니다:

| 검사 유형 | 에러 코드 키 | 설명 |
|-----------|--------------|------|
| 카메라 근접 | `detect_max_depth_item` | 카메라에 너무 가까운 물체 검출 |
| 세워진 물체 | `detect_stand_item` | 세워진 물체 검출 |
| 겹친 물체 | `detect_overlaped_item` | 겹쳐진 물체 검출 |

## 의존성

- `numpy`: 배열 연산 및 통계 계산
- `cv2` (OpenCV): 컨투어 검출
- `common_config`: 전역 설정 및 상수
  - `artis_depth_lookup_path`: Depth 클래스 정보 파일 경로
  - `image_resolution_rgb`: RGB 이미지 해상도
  - `image_resolution_depth`: Depth 이미지 해상도
  - `depth_reference_distance`: 기준 Depth 거리
  - `artis_ai_error_code`: 에러 코드 정의
  - `artis_ai_error_reason`: 에러 사유 정의

## 사용 예시

```python
from depth_validation import DepthValidation

# 초기화
validator = DepthValidation(config_file_path="config.json")

# 검사 실행
result = validator.check_validation(
    bbox=[[100, 100, 200, 200, 1001001]],  # [x1, y1, x2, y2, class_id]
    depth_object=depth_obj,                 # Depth 데이터 객체
    result_json=initial_result,             # 결과 JSON
    config=ai_config                        # AI 설정
)
```

## 알려진 제한사항

1. **기둥 영역 하드코딩**: 세워진 물체 검사에서 기둥 영역(중앙 70px)이 하드코딩되어 있음
2. **클래스 99로 시작하는 ID 필터링**: 좌표 변환 시 class_id가 '99'로 시작하면 무시됨
3. **단일 에러 반환**: 여러 검사에서 에러가 발생해도 첫 번째 에러만 반환됨 (우선순위: 근접 > 세워짐 > 겹침)
