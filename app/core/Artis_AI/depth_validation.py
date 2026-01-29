import json
import time
import numpy as np
import cv2

import common_config as cc
from common_config import CurrentDateTime


class DepthValidation:
    """Depth 기반 검사 클래스"""
    
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.class_summary = {}
        
        # 검사 옵션 초기화
        self.use_max_depth_check = False
        self.use_stand_item_check = False
        self.use_overlap_item_check = False
        self.iou_threshold = 0.3
        self.depth_threshold = 20  # 기본값 20mm
        
        # 클래스 정보 로드
        self.load_depth_class_info()
    
    def load_depth_class_info(self):
        """Depth 클래스 정보 로드"""
        try:
            with open(cc.artis_depth_lookup_path, 'r') as f:
                data = json.load(f)
                self.class_summary = data.get('summary', {})
                print(f"{CurrentDateTime(0)} [Artis_AI] Depth 클래스 정보 로드 완료: {len(self.class_summary)}개 클래스")
        except FileNotFoundError:
            print(f"{CurrentDateTime(0)} [Artis_AI] {cc.artis_depth_lookup_path} 파일을 찾을 수 없습니다.")
            self.class_summary = {}
        except json.JSONDecodeError:
            print(f"{CurrentDateTime(0)} [Artis_AI] {cc.artis_depth_lookup_path} 파일 형식이 잘못되었습니다.")
            self.class_summary = {}
        except Exception as e:
            print(f"{CurrentDateTime(0)} [Artis_AI] Depth 클래스 정보 로드 중 오류 발생: {e}")
            self.class_summary = {}
    
    def init_depth_check(self, json_config):
        """설정 파일에서 Depth 검사 옵션 초기화"""
        # 이전 설정값 저장 (변경 감지용)
        prev_max_depth_check = self.use_max_depth_check
        prev_stand_item_check = self.use_stand_item_check
        prev_overlap_item_check = self.use_overlap_item_check
        prev_iou_threshold = self.iou_threshold
        prev_depth_threshold = self.depth_threshold
        
        # 새로운 설정값 초기화
        self.use_max_depth_check = False
        self.use_stand_item_check = False
        self.use_overlap_item_check = False
        self.iou_threshold = 0.3
        self.depth_threshold = 20  # 기본값 20mm

        if "depth_check" in json_config:
            depth_config = json_config["depth_check"]
            
            # 1) 카메라와 너무 가까운 사물 체크 설정
            if "max_depth_check" in depth_config:
                self.use_max_depth_check = bool(depth_config["max_depth_check"])
            
            # 2) 세워진 사물 체크 설정
            if "stand_item_check" in depth_config:
                self.use_stand_item_check = bool(depth_config["stand_item_check"])
            
            # 3) 사물 겹침 체크 설정
            if "overlap_item_check" in depth_config:
                overlap_config = depth_config["overlap_item_check"]
                if isinstance(overlap_config, dict):
                    self.use_overlap_item_check = bool(overlap_config.get("use", False))
                    
                    if self.use_overlap_item_check:
                        # IOU 기반 겹침 임계값 설정
                        iou_level = float(overlap_config.get("iou_threshold", 0.3))
                        if 0.0 <= iou_level <= 0.9:
                            self.iou_threshold = iou_level
                        else:
                            print(f"{CurrentDateTime(0)} [Artis_AI] IOU 겹침 임계값 범위 초과하여 기본값으로 설정: {iou_level}")
                        
                        # Depth 임계값 설정
                        depth_threshold = int(overlap_config.get("depth_threshold", 20))
                        if 0 <= depth_threshold <= 100:  # 0 ~ 10cm
                            self.depth_threshold = depth_threshold
                        else:
                            print(f"{CurrentDateTime(0)} [Artis_AI] Depth 겹침 임계값 범위 초과하여 기본값으로 설정: {depth_threshold}")

            settings_changed = (
                prev_max_depth_check != self.use_max_depth_check or
                prev_stand_item_check != self.use_stand_item_check or
                prev_overlap_item_check != self.use_overlap_item_check or
                prev_iou_threshold != self.iou_threshold or
                prev_depth_threshold != self.depth_threshold
            )
            
            if settings_changed:
                print(f"{CurrentDateTime(0)} [Artis_AI] ===============================================")
                print(f"{CurrentDateTime(0)} [Artis_AI] - 카메라 근접 검사 : {self.use_max_depth_check}")
                print(f"{CurrentDateTime(0)} [Artis_AI] - 세워진 사물 검사 : {self.use_stand_item_check}")
                print(f"{CurrentDateTime(0)} [Artis_AI] - 겹친 사물 검사   : {self.use_overlap_item_check}")
                print(f"{CurrentDateTime(0)} [Artis_AI] - IOU 임계값       : {self.iou_threshold}")
                print(f"{CurrentDateTime(0)} [Artis_AI] - Depth 임계값     : {self.depth_threshold}mm")
                print(f"{CurrentDateTime(0)} [Artis_AI] ===============================================")
        else:
            print(f"{CurrentDateTime(0)} [Artis_AI] Depth 기반 검사 미사용")
    
    def check_validation(self, bbox, bbox_all, depth_object, result_json, config):
        """
        Depth 기반 검사 진입점

        Args:
            bbox: bbox 리스트 (RGB 좌표, OC 결과만)
            bbox_all: bbox 리스트 (RGB 좌표, OC + invalid_bbox_det_0 포함)
            depth_object: Depth 객체 (args.min_depth 포함)
            result_json: 결과 JSON 객체
            config: 설정 JSON 객체 (artis_ai_json_config)

        Returns:
            result_json: 업데이트된 결과 JSON
        """
        # 설정 초기화
        self.init_depth_check(config)

        # depth_object에서 raw_depth 가져오기
        raw_depth = getattr(depth_object, 'raw_depth', None)

        # min_depth는 depth_object에서 가져오기
        min_depth = 200  # 기본값
        if hasattr(depth_object, 'args') and hasattr(depth_object.args, 'min_depth'):
            min_depth = depth_object.args.min_depth
        
        # mode_seg_with_feature_matching 모드에서 bbox 클래스 정보 업데이트
        ai_mode = config.get("ai_model_mode", 0)
        if ai_mode == cc.artis_ai_model_mode["mode_seg_with_feature_matching"]:
            if 'artis_object_detail' in result_json and 'artis_object_bbox' in result_json:
                # bbox와 artis_object_bbox를 매칭하여 클래스 정보 업데이트
                for bbox_item in bbox:
                    if len(bbox_item) < 5:
                        continue
                    x1, y1, x2, y2 = int(bbox_item[0]), int(bbox_item[1]), int(bbox_item[2]), int(bbox_item[3])
                    
                    for obj_idx, obj_bbox in result_json['artis_object_bbox'].items():
                        if (obj_bbox[0] == x1 and obj_bbox[1] == y1 and 
                            obj_bbox[2] == x2 and obj_bbox[3] == y2):
                            if obj_idx in result_json['artis_object_detail']:
                                class_id = int(result_json['artis_object_detail'][obj_idx])
                                bbox_item[4] = class_id
                                break
        
        if not (self.use_max_depth_check or self.use_stand_item_check or self.use_overlap_item_check):
            print(f"{CurrentDateTime(0)} [Artis_AI] Depth 기반 검사 건너뜀 (조건 불만족)")
            return result_json
            
        check_start_time = time.time()
        error_code, error_reason, error_bboxes = self._check_inference_result(bbox, bbox_all, raw_depth, min_depth)
        check_time = time.time() - check_start_time
        
        print(f"{CurrentDateTime(0)} [Artis_AI] - 총 처리 시간 : {check_time*1000:.3f}ms")
        print(f"{CurrentDateTime(0)} [Artis_AI] - 검출된 사물  : {len(error_bboxes) if error_bboxes is not None else 0}개")
        print(f"{CurrentDateTime(0)} [Artis_AI] - 오류 사유    : {error_reason if error_reason is not None else '정상'}")
        
        if error_code:
            cc.artis_ai_current_error_code = error_code
            cc.artis_ai_current_error_reason = error_reason

            if 'error_object_bbox' not in result_json['depth']:
                result_json['depth']['error_object_bbox'] = {}
            for index, error_bbox in enumerate(error_bboxes):
                current_json_data = {str(index): [int(error_bbox[0]), int(error_bbox[1]), int(error_bbox[2]), int(error_bbox[3])]}
                result_json['depth']['error_object_bbox'].update(current_json_data)

            result_json = cc.make_error_code(result_json, cc.artis_ai_current_error_code, cc.artis_ai_current_error_reason)
        
        return result_json
    
    def _check_inference_result(self, bbox, bbox_all, depth, min_depth):
        """실제 검사 로직 수행
        
        Args:
            bbox: OC 결과만 포함된 bbox 리스트 (겹친 사물 검사용)
            bbox_all: OC + invalid_bbox_det_0 포함된 bbox 리스트 (세워진 사물 검사용)
            depth: raw depth 데이터
            min_depth: 최소 depth 값
        """
        # depth가 리스트인 경우 numpy 배열로 변환
        if isinstance(depth, list):
            depth = np.array(depth)
            
        if depth is None or depth.size == 0 or np.count_nonzero(np.isfinite(depth)) == 0:
            print(f"{CurrentDateTime(0)} [Artis_AI] Error: Depth 값이 없습니다.")
            return None, None, None
        
        # 1) 카메라 근접 사물 검사
        if self.use_max_depth_check:
            start_time = time.time()
            depth_result = self._detect_max_depth(depth, min_depth)
            depth_time = (time.time() - start_time) * 1000
            print(f"{CurrentDateTime(0)} [Artis_AI] ==============================================> 카메라 근접 검사 완료 ({depth_time:.3f}ms)")

            if depth_result['type'] == 'max_depth':
                error_code = str(hex(cc.artis_ai_error_code['inference']['detect_max_depth_item']))
                error_reason = cc.artis_ai_error_reason['inference'][cc.artis_ai_error_code['inference']['detect_max_depth_item']]
                max_depth_objects = self._convert_depth_to_rgb_bbox(depth_result['bboxes'])
                return error_code, error_reason, max_depth_objects

        # 2) 세워진 사물 검사 (bbox_all 사용: OC + invalid_bbox_det_0, 비어있으면 bbox 사용)
        if self.use_stand_item_check:
            # bbox_all이 비어있으면 bbox 사용 (feature_matching 모드 대응)
            bbox_for_stand_check = bbox_all if bbox_all else bbox
            
            if not bbox_for_stand_check:
                print(f"{CurrentDateTime(0)} [Artis_AI] ==============================================> [세워진 사물 검사] bbox 없음")
            else:
                start_time = time.time()
                print(f"{CurrentDateTime(0)} [Artis_AI] [세워진 사물 검사] bbox 개수: {len(bbox_for_stand_check)}")
                bbox_converted = self._convert_rgb_to_depth_bbox(bbox_for_stand_check)
                depth_result = self._detect_standing_objects(depth, bbox_converted)
                depth_time = (time.time() - start_time) * 1000
                print(f"{CurrentDateTime(0)} [Artis_AI] ==============================================> 세워진 사물 검사 완료 ({depth_time:.3f}ms)")

                if depth_result['type'] == 'standing' and depth_result['bboxes']:
                    error_code = str(hex(cc.artis_ai_error_code['inference']['detect_stand_item']))
                    error_reason = cc.artis_ai_error_reason['inference'][cc.artis_ai_error_code['inference']['detect_stand_item']]
                    depth_objects = self._convert_depth_to_rgb_bbox(depth_result['bboxes'])
                    return error_code, error_reason, depth_objects

        # 3) 겹친 사물 검사 (bbox 사용: OC 결과만)
        if self.use_overlap_item_check:
            if not bbox:
                print(f"{CurrentDateTime(0)} [Artis_AI] ==============================================> [겹친 사물 검출] bbox 없음")
            else:
                start_time = time.time()
                print(f"{CurrentDateTime(0)} [Artis_AI] [겹친 사물 검출] bbox 개수: {len(bbox)}")
                bbox_converted = self._convert_rgb_to_depth_bbox(bbox)
                overlapped_bboxes = self._detect_overlapped_object(bbox_converted, depth)
                overlap_time = (time.time() - start_time) * 1000
                print(f"{CurrentDateTime(0)} [Artis_AI] ==============================================> 겹친 사물 검사 완료 ({overlap_time:.3f}ms)")
                
                if overlapped_bboxes:
                    error_code = str(hex(cc.artis_ai_error_code['inference']['detect_overlaped_item']))
                    error_reason = cc.artis_ai_error_reason['inference'][cc.artis_ai_error_code['inference']['detect_overlaped_item']]
                    overlapped_bboxes = self._convert_depth_to_rgb_bbox(overlapped_bboxes)
                    return error_code, error_reason, overlapped_bboxes
            
        return None, None, None
    
    # ==================== 좌표 변환 함수 ====================

    def _convert_bbox_coordinates(self, bbox_list, to_depth=True):
        """
        bbox 좌표 변환 (RGB <-> Depth)

        Args:
            bbox_list: 변환할 bbox 리스트
            to_depth: True면 RGB→Depth, False면 Depth→RGB
        """
        if not bbox_list:
            return None

        rgb_h, rgb_w = cc.image_resolution_rgb
        depth_h, depth_w = cc.image_resolution_depth

        if to_depth:
            src_w, src_h = rgb_w, rgb_h
            dst_w, dst_h = depth_w, depth_h
        else:
            src_w, src_h = depth_w, depth_h
            dst_w, dst_h = rgb_w, rgb_h

        converted_list = []
        for bbox in bbox_list:
            bbox = list(map(int, bbox[:4])) + list(bbox[4:])
            x1, y1, x2, y2 = bbox[:4]
            class_id = int(bbox[4]) if len(bbox) > 4 else 0
            if str(class_id).startswith('99'):
                continue

            new_x1 = int(x1 * dst_w / src_w)
            new_y1 = int(y1 * dst_h / src_h)
            new_x2 = int(x2 * dst_w / src_w)
            new_y2 = int(y2 * dst_h / src_h)

            new_x1 = max(0, min(new_x1, dst_w-1))
            new_y1 = max(0, min(new_y1, dst_h-1))
            new_x2 = max(0, min(new_x2, dst_w-1))
            new_y2 = max(0, min(new_y2, dst_h-1))

            converted_list.append([new_x1, new_y1, new_x2, new_y2, class_id])

        return converted_list

    def _convert_rgb_to_depth_bbox(self, bbox_list):
        """RGB 좌표를 Depth 좌표로 변환"""
        return self._convert_bbox_coordinates(bbox_list, to_depth=True)

    def _convert_depth_to_rgb_bbox(self, bbox_list):
        """Depth 좌표를 RGB 좌표로 변환"""
        return self._convert_bbox_coordinates(bbox_list, to_depth=False)
    
    # ==================== 검사 함수 ====================
    
    def _detect_max_depth(self, depth, min_depth):
        """카메라 근접 물체 검출"""
        depth = np.asarray(depth)
        h, w = depth.shape
        margin = 100
        
        center = depth[margin:h-margin, margin:w-margin]
        mask = (center <= min_depth).astype(np.uint8)
        ratio = np.mean(mask)
        
        # 비율이 20% 이상이면 근접 물체 검출
        if ratio >= 0.2:
            print(f"{CurrentDateTime(0)} [Artis_AI] [카메라 근접 검사] 근접 물체 검출 → 검출 O")
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 가장 큰 contour의 bounding box 사용
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w_bbox, h_bbox = cv2.boundingRect(largest_contour)
                # 마진을 고려하여 원본 이미지 좌표로 변환
                bbox = [x + margin, y + margin, x + w_bbox + margin, y + h_bbox + margin, 1.0]
                print(f"{CurrentDateTime(0)} [Artis_AI] [카메라 근접 검사] 검출된 영역 bbox: {bbox}")
                return {'type': 'max_depth', 'bboxes': [bbox]}
            else:
                # contour가 없으면 전체 검사 영역 반환
                bbox = [margin, margin, w-margin, h-margin, 1.0]
                return {'type': 'max_depth', 'bboxes': [bbox]}
        else:
            print(f"{CurrentDateTime(0)} [Artis_AI] [카메라 근접 검사] 근접 물체 없음 → 검출 X")
            return {'type': 'none', 'bboxes': [], 'debug_info': []}
    
    def _detect_standing_objects(self, depth, bbox_list=None):
        """세워진 사물 검출"""
        if not bbox_list:
            return {'type': 'none', 'bboxes': [], 'debug_info': []}
        
        MAX_PIXEL_COUNT = 8000         # 픽셀 수 임계값 (8000픽셀 이상 시 depth 조건 적용)
        MAX_DEPTH = 100                # 최대 depth 임계값 (10cm)
        MIN_DEPTH = 30                 # 최소 depth 임계값 (3cm)
        MIN_RATIO = 0.5                # 면적 비율 임계값 (50% 이상)
        MAX_BBOX_SIZE = 200            # bbox 최대 크기
        standing_objects = []          # bbox 검출 결과 저장
        debug_info = []                # 디버깅 정보 저장

        # 기둥 영역 무시
        depth = depth.copy()
        depth_height, depth_width = depth.shape[:2]
        ignore_size = 70
        center_x = depth_width // 2
        ignore_x1 = int(center_x - (ignore_size / 2))
        ignore_x2 = int(center_x + (ignore_size / 2))
        
        # 기둥 영역의 depth 값을 베이스 Depth로 설정하여 필터링에서 제외
        depth[0:ignore_size, ignore_x1:ignore_x2] = cc.depth_reference_distance
        
        for bbox in bbox_list:
            x1, y1, x2, y2 = map(int, bbox[:4])
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Depth 영역 추출
            bbox_depth = depth[y1:y2, x1:x2]
            valid_depth = bbox_depth[bbox_depth < cc.depth_reference_distance - MIN_DEPTH]
            stand_depth = bbox_depth[bbox_depth < cc.depth_reference_distance - MAX_DEPTH]
            
            if valid_depth.size == 0 or stand_depth.size == 0:
                print(f"{CurrentDateTime(0)} [Artis_AI] bbox 크기: {bbox_width} x {bbox_height} | 유효한 Depth값 없음 → 검출 X")
                continue
            
            # 통계 계산
            stand_count = stand_depth.size
            valid_count = valid_depth.size
            area_ratio = stand_count / valid_count
            stand_std = np.std(stand_depth)
            
            # 디버깅 정보
            debug_data = {
                'bbox': bbox,
                'bbox_coords': [x1, y1, x2, y2],
                'area_ratio': area_ratio,
                'threshold_max_depth': MAX_DEPTH,
                'threshold_min_depth': MIN_DEPTH,
                'stand_depth_std': stand_std,
                'stand_count': stand_count,
                'valid_count': valid_count,
                'bbox_depth_shape': bbox_depth.shape,
            }
            
            # 검출 조건 판단
            is_detected = False

            # 기본 조건: 면적 비율 50% 이상 + 픽셀 수 8000개 이하 + 가로/세로 모두 200픽셀 이하
            if area_ratio >= MIN_RATIO and stand_count <= MAX_PIXEL_COUNT and bbox_width <= MAX_BBOX_SIZE and bbox_height <= MAX_BBOX_SIZE:
                is_detected = True
                print(f"{CurrentDateTime(0)} [Artis_AI] bbox 크기: {bbox_width} x {bbox_height} | 픽셀 수: {stand_count} | 비율: {area_ratio:.2f} → 검출 O")
            
            # 추가 조건: 픽셀 수 초과 시 depth 중간값과 bbox 크기 확인
            elif stand_count > MAX_PIXEL_COUNT:
                median_depth = np.median(stand_depth)
                depth_height_val = cc.depth_reference_distance - median_depth
                
                aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
                
                # 서있는 물체 판단 조건
                height_condition = median_depth < (cc.depth_reference_distance - (MAX_DEPTH * 2))  # 20cm 이상
                pixel_condition = stand_count < 30000  # depth 픽셀 수가 30000개 미만
                
                if height_condition and pixel_condition:
                    is_detected = True
                    print(f"{CurrentDateTime(0)} [Artis_AI] bbox 크기: {bbox_width} x {bbox_height} | 픽셀 수: {stand_count} | 사물 높이: {depth_height_val:.0f}mm | 비율: {area_ratio:.2f} | 가로세로비: {aspect_ratio:.1f} → 검출 O")
                else:
                    print(f"{CurrentDateTime(0)} [Artis_AI] bbox 크기: {bbox_width} x {bbox_height} | 픽셀 수: {stand_count} | 사물 높이: {depth_height_val:.0f}mm | 비율: {area_ratio:.2f} | 가로세로비: {aspect_ratio:.1f} → 검출 X")
            else:
                print(f"{CurrentDateTime(0)} [Artis_AI] bbox 크기: {bbox_width} x {bbox_height} | 픽셀 수: {stand_count} | 비율: {area_ratio:.2f} → 검출 X")
            
            debug_data['detected'] = is_detected
            if is_detected:
                standing_objects.append(bbox)
            
            debug_info.append(debug_data)
        
        return {'type': 'standing', 'bboxes': standing_objects, 'debug_info': debug_info}
    
    def _detect_overlapped_object(self, bbox_list, depth_array):
        """겹친 사물 검출"""
        min_depth, max_depth = 200, cc.depth_reference_distance
        overlapped_bboxes = []

        # 1) IOU & IOM 기반 검사 시작
        iou_start_time = time.time()
        iou_overlapped_bboxes, remaining_bboxes = self._check_bbox_overlap_by_iou_iom(bbox_list, depth_array)
        iou_end_time = time.time()
        iou_time = (iou_end_time - iou_start_time) * 1000
        print(f"{CurrentDateTime(0)} [Artis_AI] -> IOU & IOM 기반 처리시간: {iou_time:.3f}ms")
        
        if iou_overlapped_bboxes:
            overlapped_bboxes.extend(iou_overlapped_bboxes)

            if not remaining_bboxes:
                return overlapped_bboxes
        else:
            remaining_bboxes = bbox_list

        # 2) Depth 특성값 기반 검사 시작
        depth_start_time = time.time()
        for bbox in remaining_bboxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            class_id = int(bbox[4])
            class_id_str = str(class_id).zfill(7)

            if class_id_str not in self.class_summary:
                print(f"{CurrentDateTime(0)} [Artis_AI] 클래스 {class_id_str}에 대한 Depth 기준값 없음")
                continue

            ref_info = self.class_summary[class_id_str]
            ref_median = ref_info.get('median')
            ref_range = ref_info.get('range')

            if None in (ref_median, ref_range):
                print(f"{CurrentDateTime(0)} [Artis_AI] 클래스 {class_id_str}의 depth 정보 불완전")
                continue

            # 중심 ROI = 전체 bbox 면적의 1/3을 정사각형으로 중앙에 위치
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            center_size = int(np.sqrt(box_area / 3.0))
            half_size = center_size // 2

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center_x1 = max(x1, center_x - half_size)
            center_y1 = max(y1, center_y - half_size)
            center_x2 = min(x2, center_x + half_size)
            center_y2 = min(y2, center_y + half_size)
            center_roi = depth_array[center_y1:center_y2, center_x1:center_x2]

            if center_roi.size == 0:
                print(f"{CurrentDateTime(0)} [Artis_AI] 중심 영역 데이터 없음")
                continue

            center_flat = center_roi[(center_roi >= min_depth) & (center_roi <= max_depth)]
            if center_flat.size == 0:
                print(f"{CurrentDateTime(0)} [Artis_AI] bbox {x1,y1,x2,y2} 중심에 유효한 Depth 없음")
                continue

            bbox_roi = depth_array[y1:y2, x1:x2]
            bbox_flat = bbox_roi[(bbox_roi >= min_depth) & (bbox_roi <= max_depth)]
            if bbox_flat.size == 0:
                print(f"{CurrentDateTime(0)} [Artis_AI] bbox {x1,y1,x2,y2} 전체 영역에 유효한 Depth 없음")
                continue

            actual_median = np.median(center_flat)
            actual_range = np.max(bbox_flat) - np.min(bbox_flat)
            median_diff = abs(actual_median - ref_median)
            range_diff = abs(actual_range - ref_range)

            is_overlapped = median_diff > self.depth_threshold and range_diff > self.depth_threshold
            if is_overlapped:
                overlapped_bboxes.append([x1, y1, x2, y2, class_id]) 
                print(f"{CurrentDateTime(0)} [Artis_AI] [{class_id_str}] Bbox {x1,y1,x2,y2} → 겹침 O:")
            else:
                print(f"{CurrentDateTime(0)} [Artis_AI] [{class_id_str}] Bbox {x1,y1,x2,y2} → 겹침 X:")

            cond_median = median_diff > self.depth_threshold
            cond_range = range_diff > self.depth_threshold

            print(f"{CurrentDateTime(0)} [Artis_AI]  Median 차이    > {self.depth_threshold:>4}mm ? {median_diff:>6.1f} → {cond_median}")
            print(f"{CurrentDateTime(0)} [Artis_AI]  Range  차이    > {self.depth_threshold:>4}mm ? {range_diff:>6.1f} → {cond_range}")
            print(f"{CurrentDateTime(0)} [Artis_AI] 기준값   → Median: {ref_median}, Range: {ref_range}")
            print(f"{CurrentDateTime(0)} [Artis_AI] 측정값   → Median: {int(actual_median)}, Range: {int(actual_range)}")

        depth_end_time = time.time()
        depth_time = (depth_end_time - depth_start_time) * 1000
        print(f"{CurrentDateTime(0)} [Artis_AI] -> Depth 기반 처리시간: {depth_time:.3f}ms")

        return overlapped_bboxes
    
    # ==================== 유틸리티 함수 ====================
    
    def _calculate_iou(self, box1, box2):
        """IOU 계산"""
        x1 = max(int(box1[0]), int(box2[0]))
        y1 = max(int(box1[1]), int(box2[1]))
        x2 = min(int(box1[2]), int(box2[2]))
        y2 = min(int(box1[3]), int(box2[3]))

        # 교집합 영역의 넓이 계산
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (int(box1[2]) - int(box1[0])) * (int(box1[3]) - int(box1[1]))
        box2_area = (int(box2[2]) - int(box2[0])) * (int(box2[3]) - int(box2[1]))

        # 합집합 영역의 넓이 계산
        union = box1_area + box2_area - intersection

        iou = intersection / union if union > 0 else 0
        return iou
    
    def _calculate_iom(self, box1, box2):
        """IOM (Intersection over Minimum) 계산"""
        x1 = max(int(box1[0]), int(box2[0]))
        y1 = max(int(box1[1]), int(box2[1]))
        x2 = min(int(box1[2]), int(box2[2]))
        y2 = min(int(box1[3]), int(box2[3]))

        # 교집합 영역 넓이 계산
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (int(box1[2]) - int(box1[0])) * (int(box1[3]) - int(box1[1]))
        box2_area = (int(box2[2]) - int(box2[0])) * (int(box2[3]) - int(box2[1]))

        min_area = min(box1_area, box2_area)

        iom = intersection / min_area if min_area > 0 else 0
        return iom
    
    def _check_bbox_overlap_by_iou_iom(self, bbox_list, depth_array):
        """IOU/IOM 기반 bbox 겹침 검사"""
        if self.iou_threshold == 0.0:
            print(f"{CurrentDateTime(0)} [Artis_AI] 임계값이 0 이므로 IOU & IOM 기반 검사 스킵")
            return [], bbox_list

        overlapped, used = [], set()

        for i, base in enumerate(bbox_list):
            if i in used:
                continue

            group = [i]
            for j in range(i + 1, len(bbox_list)):
                if j in used:
                    continue

                compare = bbox_list[j]
                iou = self._calculate_iou(base, compare)
                iom = self._calculate_iom(base, compare)
                valid_depth_ratio = 0.0

                if iou > self.iou_threshold or iom > self.iou_threshold:
                    x1, y1 = max(base[0], compare[0]), max(base[1], compare[1])
                    x2, y2 = min(base[2], compare[2]), min(base[3], compare[3])

                    # 교집합 영역의 유효한 Depth 비율 계산 0.7 이상 겹침
                    region = depth_array[int(y1):int(y2), int(x1):int(x2)]
                    valid = (region > 0) & (region < cc.depth_reference_distance)
                    valid_depth_ratio = np.sum(valid) / region.size if region.size > 0 else 0

                if valid_depth_ratio > 0.7:
                    print(f"{CurrentDateTime(0)} [Artis_AI] Bbox {i}-{j} IOU: {iou:.2f} | IOM: {iom:.2f} | Depth 비율: {valid_depth_ratio:.2f} → 겹침 O")
                    group.append(j)
                else:
                    print(f"{CurrentDateTime(0)} [Artis_AI] Bbox {i}-{j} IOU: {iou:.2f} | IOM: {iom:.2f} | Depth 비율: {valid_depth_ratio:.2f} → 겹침 X")

            if len(group) > 1:
                x1 = min(bbox_list[k][0] for k in group)
                y1 = min(bbox_list[k][1] for k in group)
                x2 = max(bbox_list[k][2] for k in group)
                y2 = max(bbox_list[k][3] for k in group)
                class_id = bbox_list[group[0]][4]
                overlapped.append(np.array([x1, y1, x2, y2, class_id]))
                used.update(group)

        remaining = [bbox_list[i] for i in range(len(bbox_list)) if i not in used]
        return overlapped, remaining