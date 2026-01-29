"""
상품 관리 서비스
상품 관리 관련 비즈니스 로직
"""
from typing import Dict, List, Any, Optional
from app.core.database import execute_query, execute_one, execute_update


class ProductService:
    """상품 관리 서비스"""
    
    @staticmethod
    async def get_product_list(
        vendor_id: Optional[int] = None,
        category: Optional[str] = None,
        is_deleted: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
        search: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """상품 목록 조회"""
        try:
            # 기본 쿼리
            query = """
                SELECT 
                    i.id,
                    i.vendor_id,
                    v.vendor_name,
                    v.display_name as vendor_display_name,
                    i.item_code,
                    i.barcode,
                    i.item_name_default,
                    i.item_description_default,
                    i.category_top,
                    i.category_mid,
                    i.category_low,
                    i.currency_code,
                    i.base_amount,
                    i.vat_included,
                    i.is_pos_use,
                    i.is_deleted,
                    i.stock,
                    i.is_out_of_stock,
                    i.item_type,
                    i.order_unit,
                    i.disp_priority,
                    i.is_discounted,
                    i.discount_rate,
                    i.scan_image_count,
                    i.thumb_image_file,
                    i.similar_item_group,
                    i.option_groups,
                    i.created_at,
                    i.updated_at
                FROM items i
                LEFT JOIN vendors v ON i.vendor_id = v.id
                WHERE i.is_deleted = %s
            """
            
            params = [is_deleted]
            
            # vendor_id 필터
            if vendor_id is not None:
                query += " AND i.vendor_id = %s"
                params.append(vendor_id)
            
            # category 필터
            if category:
                query += " AND i.category_top = %s"
                params.append(category)
            
            # 검색 필터 (item_code, item_name_default, barcode에서 검색)
            if search:
                query += " AND (i.item_code LIKE %s OR i.item_name_default LIKE %s OR i.barcode LIKE %s)"
                search_pattern = f"%{search}%"
                params.extend([search_pattern, search_pattern, search_pattern])
            
            # 정렬
            query += " ORDER BY i.disp_priority DESC, i.item_code ASC"
            
            # 페이징 (limit이 None이면 전체 조회)
            if limit is not None:
                query += " LIMIT %s OFFSET %s"
                params.extend([limit, offset])
            elif offset > 0:
                # offset만 있는 경우 (limit 없이 offset만 사용하는 것은 일반적이지 않지만 지원)
                query += " LIMIT 18446744073709551615 OFFSET %s"
                params.append(offset)
            
            results = await execute_query(query, tuple(params))
            
            # Decimal 타입을 float로 변환
            for item in results:
                if 'base_amount' in item and item['base_amount'] is not None:
                    item['base_amount'] = float(item['base_amount'])
                if 'discount_rate' in item and item['discount_rate'] is not None:
                    item['discount_rate'] = float(item['discount_rate'])
                # JSON 필드 파싱
                if item.get('similar_item_group'):
                    try:
                        import json
                        if isinstance(item['similar_item_group'], str):
                            item['similar_item_group'] = json.loads(item['similar_item_group'])
                    except:
                        item['similar_item_group'] = None
                if item.get('option_groups'):
                    try:
                        import json
                        if isinstance(item['option_groups'], str):
                            item['option_groups'] = json.loads(item['option_groups'])
                    except:
                        item['option_groups'] = None
            
            return results
        except Exception as e:
            print(f"❌ 상품 목록 조회 실패: {e}")
            raise
    
    @staticmethod
    async def get_product_count(
        vendor_id: Optional[int] = None,
        category: Optional[str] = None,
        is_deleted: bool = False,
        search: Optional[str] = None
    ) -> int:
        """상품 개수 조회"""
        try:
            query = "SELECT COUNT(*) as count FROM items WHERE is_deleted = %s"
            params = [is_deleted]
            
            if vendor_id is not None:
                query += " AND vendor_id = %s"
                params.append(vendor_id)
            
            if category:
                query += " AND category_top = %s"
                params.append(category)
            
            # 검색 필터
            if search:
                query += " AND (item_code LIKE %s OR item_name_default LIKE %s OR barcode LIKE %s)"
                search_pattern = f"%{search}%"
                params.extend([search_pattern, search_pattern, search_pattern])
            
            result = await execute_one(query, tuple(params))
            return result['count'] if result else 0
        except Exception as e:
            print(f"❌ 상품 개수 조회 실패: {e}")
            raise
    
    @staticmethod
    async def get_product_detail(product_id: int) -> Optional[Dict[str, Any]]:
        """상품 상세 정보 조회"""
        try:
            query = """
                SELECT 
                    i.*,
                    v.vendor_name,
                    v.display_name as vendor_display_name
                FROM items i
                LEFT JOIN vendors v ON i.vendor_id = v.id
                WHERE i.id = %s
            """
            
            result = await execute_one(query, (product_id,))
            
            if result:
                # Decimal 타입을 float로 변환
                if 'base_amount' in result and result['base_amount'] is not None:
                    result['base_amount'] = float(result['base_amount'])
                if 'discount_rate' in result and result['discount_rate'] is not None:
                    result['discount_rate'] = float(result['discount_rate'])
                # JSON 필드 파싱
                if result.get('similar_item_group'):
                    try:
                        import json
                        if isinstance(result['similar_item_group'], str):
                            result['similar_item_group'] = json.loads(result['similar_item_group'])
                    except:
                        result['similar_item_group'] = None
                if result.get('option_groups'):
                    try:
                        import json
                        if isinstance(result['option_groups'], str):
                            result['option_groups'] = json.loads(result['option_groups'])
                    except:
                        result['option_groups'] = None
            
            return result
        except Exception as e:
            print(f"❌ 상품 상세 정보 조회 실패: {e}")
            raise
    
    @staticmethod
    async def create_product(product_data: Dict[str, Any]) -> Dict[str, Any]:
        """상품 생성"""
        # TODO: 실제 상품 생성 로직 구현
        return {"success": True, "message": "상품이 생성되었습니다"}
    
    @staticmethod
    async def update_product(product_id: str, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """상품 수정"""
        # TODO: 실제 상품 수정 로직 구현
        return {"success": True, "message": "상품이 수정되었습니다"}
    
    @staticmethod
    async def delete_product(product_id: str) -> Dict[str, Any]:
        """상품 삭제"""
        # TODO: 실제 상품 삭제 로직 구현
        return {"success": True, "message": "상품이 삭제되었습니다"}


# 전역 서비스 인스턴스
product_service = ProductService()

