"""
SDUI (Server-Driven UI) 서비스
서버에서 UI 구성 정보를 JSON 형태로 생성하여 클라이언트에 전달
"""
from typing import Dict, List, Any
from app.core.constants import TITLE, VERSION


class WebUIService:
    """웹 UI 구성 서비스"""
    
    @staticmethod
    def _get_navigation_bar(is_main_page: bool = False, page_name: str = None) -> Dict[str, Any]:
        """
        공통 네비게이션 바 구성
        
        Args:
            is_main_page: 메인 페이지 여부
                - True: 메인 페이지 (홈 버튼, X 버튼 제외)
                - False: 다른 페이지 (X 버튼만 제외)
            page_name: 페이지 이름 (검색 버튼 표시 여부 결정)
                - "product": 상품 관리 페이지 (검색 버튼 표시)
                - 기타: 검색 버튼 미표시
        """
        buttons = [
            {
                "icon": "◀",
                "label": "뒤로",
                "action": {
                    "type": "go_back"
                },
                "style": {
                    "title": "뒤로 가기"
                }
            },
            {
                "icon": "▶",
                "label": "앞으로",
                "action": {
                    "type": "go_forward"
                },
                "style": {
                    "title": "앞으로 가기"
                }
            },
            {
                "icon": "↻",
                "label": "새로고침",
                "action": {
                    "type": "refresh"
                },
                "style": {
                    "title": "새로고침"
                }
            }
        ]
        
        # 메인 페이지가 아닐 때만 홈 버튼 추가
        '''
        if not is_main_page:
            buttons.append({
                "icon": "🏠",
                "label": "홈",
                "action": {
                    "type": "go_home"
                },
                "style": {
                    "title": "홈으로"
                }
            })
        '''

        buttons.append({
            "type": "separator"
        })
        
        buttons.extend([
            {
                "icon": "⇧",
                "label": "최상단",
                "action": {
                    "type": "scroll_to_top"
                },
                "style": {
                    "title": "페이지 최상단으로 이동"
                }
            },
            {
                "icon": "↑",
                "label": "위로",
                "action": {
                    "type": "scroll_up"
                },
                "style": {
                    "title": "위로 스크롤"
                }
            },
            {
                "icon": "↓",
                "label": "아래로",
                "action": {
                    "type": "scroll_down"
                },
                "style": {
                    "title": "아래로 스크롤"
                }
            },
            {
                "icon": "⇩",
                "label": "최하단",
                "action": {
                    "type": "scroll_to_bottom"
                },
                "style": {
                    "title": "페이지 최하단으로 이동"
                }
            }
        ])
        
        # 검색 버튼은 특정 페이지에서만 표시
        # 현재는 상품 관리 페이지("product")에서만 표시
        # 나중에 다른 페이지를 추가하려면 이 리스트에 추가하면 됨
        pages_with_search = ["product"]
        if page_name in pages_with_search:
            buttons.append({
                "type": "separator"
            })
            buttons.append({
                "icon": "🔍",
                "label": "검색",
                "action": {
                    "type": "toggle_search"
                },
                "style": {
                    "title": "상품 검색"
                }
            })

        # 메인 페이지일 때만 X 버튼 추가 (브라우저 종료)
        if is_main_page:
            buttons.append({
                "type": "separator"
            })
            buttons.append({
                "icon": "✕",
                "label": "닫기",
                "action": {
                    "type": "close"
                },
                "style": {
                    "title": "브라우저 종료"
                }
            })
        
        return {
            "type": "navigation_bar",
            "buttons": buttons,
            "style": {}
        }
    
    @staticmethod
    async def generate_ui_config(page: str = "main") -> Dict[str, Any]:
        """
        UI 구성 정보를 JSON 형태로 생성
        
        Args:
            page: 페이지 이름 ("main", "product", "settings", "password")
        
        Returns:
            UI 구성 정보 딕셔너리
        """
        if page == "main":
            return await WebUIService._generate_main_page_config()
        elif page == "product":
            return await WebUIService._generate_product_page_config()
        elif page == "settings":
            return await WebUIService._generate_settings_page_config()
        elif page == "password":
            # URL 파라미터에서 리다이렉트 URL 가져오기
            from fastapi import Request
            # 이 함수는 Request 객체를 직접 받을 수 없으므로, 
            # main.py에서 쿼리 파라미터로 전달받도록 수정 필요
            redirect_url = "/settings"  # 기본값
            return await WebUIService._generate_password_page_config(redirect_url)
        else:
            return await WebUIService._generate_main_page_config()
    
    @staticmethod
    async def _generate_main_page_config() -> Dict[str, Any]:
        """메인 페이지 UI 구성 (아이콘 그리드)"""
        return {
            "title": "iScan 관리 페이지",
            "version": VERSION,
            "layout": {
                "type": "container",
                "direction": "vertical",
                "children": [
                    {
                        "type": "header",
                        "title": "iScan 관리 페이지",
                        "subtitle": f"Version {VERSION}",
                        "style": {
                            "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                            "color": "white",
                            "padding": "30px",
                            "textAlign": "center"
                        }
                    },
                    {
                        "type": "container",
                        "style": {
                            "padding": "40px 20px",
                            "maxWidth": "1200px",
                            "margin": "0 auto"
                        },
                        "children": [
                            {
                                "type": "text",
                                "content": "더 효율적인 운영의 시작, 기산전자의 iScan 과 함께하세요.",
                                "style": {
                                    "fontSize": "24px",
                                    "fontWeight": "600",
                                    "color": "#333",
                                    "textAlign": "center",
                                    "marginBottom": "50px",
                                    "lineHeight": "1.5"
                                }
                            },
                            {
                                "type": "container",
                                "direction": "horizontal",
                                "style": {
                                    "display": "flex",
                                    "flexWrap": "wrap",
                                    "justifyContent": "center",
                                    "gap": "30px"
                                },
                                "children": [
                                    {
                                        "type": "icon_card",
                                        "title": "상품 관리",
                                        "icon": "📦",
                                        "description": "상품 정보를 관리합니다",
                                        "action": {
                                            "type": "navigate",
                                            "url": "/product"
                                        },
                                        "style": {
                                            "width": "280px",
                                            "height": "200px",
                                            "backgroundColor": "#ffffff",
                                            "borderRadius": "12px",
                                            "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                                            "cursor": "pointer",
                                            "transition": "all 0.3s ease"
                                        }
                                    },
                                    {
                                        "type": "icon_card",
                                        "title": "환경 설정",
                                        "icon": "⚙️",
                                        "description": "시스템 설정을 관리합니다",
                                        "action": {
                                            "type": "navigate",
                                            "url": "/password?redirect=/settings"
                                        },
                                        "style": {
                                            "width": "280px",
                                            "height": "200px",
                                            "backgroundColor": "#ffffff",
                                            "borderRadius": "12px",
                                            "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                                            "cursor": "pointer",
                                            "transition": "all 0.3s ease"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    WebUIService._get_navigation_bar(is_main_page=True, page_name="main")
                ]
            },
            "theme": {
                "primaryColor": "#667eea",
                "secondaryColor": "#764ba2",
                "backgroundColor": "#ffffff",
                "textColor": "#333333",
                "borderRadius": "8px",
                "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
            }
        }
    
    @staticmethod
    async def _generate_product_page_config() -> Dict[str, Any]:
        """상품 관리 페이지 UI 구성"""
        # 데이터베이스 조회는 클라이언트 측에서 비동기로 처리
        # 페이지는 먼저 렌더링하고, 데이터는 나중에 로드
        
        return {
            "title": "상품 관리",
            "version": VERSION,
            "layout": {
                "type": "container",
                "direction": "vertical",
                "children": [
                    {
                        "type": "header",
                        "title": "상품 관리",
                        "subtitle": "",
                        "style": {
                            "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                            "color": "white",
                            "padding": "30px",
                            "textAlign": "center"
                        }
                    },
                    {
                        "type": "container",
                        "style": {
                            "padding": "40px 20px",
                            "maxWidth": "1200px",
                            "margin": "0 auto"
                        },
                        "children": [
                            {
                                "type": "card",
                                "title": "상품 목록",
                                "style": {
                                    "marginBottom": "20px"
                                },
                                "children": [
                                    {
                                        "type": "table",
                                        "id": "product-table",
                                        "headers": ["상품코드", "상품명", "카테고리", "가격", "POS사용", "스캔 이미지 개수", "대표 이미지"],
                                        "rows": [],
                                        "style": {
                                            "width": "100%",
                                            "marginTop": "20px"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    WebUIService._get_navigation_bar(is_main_page=False, page_name="product")
                ]
            },
            "theme": {
                "primaryColor": "#667eea",
                "secondaryColor": "#764ba2",
                "backgroundColor": "#ffffff",
                "textColor": "#333333",
                "borderRadius": "8px",
                "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
            }
        }
    
    @staticmethod
    async def _generate_settings_page_config() -> Dict[str, Any]:
        """환경설정 페이지 UI 구성"""
        from app.services.settings_service import settings_service
        
        return {
            "title": "환경설정",
            "version": VERSION,
            "layout": {
                "type": "container",
                "direction": "vertical",
                "children": [
                    {
                        "type": "header",
                        "title": "환경설정",
                        "subtitle": "시스템 설정을 관리합니다",
                        "style": {
                            "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                            "color": "white",
                            "padding": "30px",
                            "textAlign": "center"
                        }
                    },
                    {
                        "type": "container",
                        "style": {
                            "padding": "40px 20px",
                            "maxWidth": "1200px",
                            "margin": "0 auto"
                        },
                        "children": [
                            {
                                "type": "card",
                                "title": "시스템 설정",
                                "style": {
                                    "marginBottom": "20px"
                                },
                                "children": [
                                    {
                                        "type": "text",
                                        "content": "환경설정 기능이 곧 제공될 예정입니다.",
                                        "style": {
                                            "fontSize": "16px",
                                            "color": "#666",
                                            "textAlign": "center",
                                            "padding": "40px"
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    WebUIService._get_navigation_bar(is_main_page=False, page_name="settings")
                ]
            },
            "theme": {
                "primaryColor": "#667eea",
                "secondaryColor": "#764ba2",
                "backgroundColor": "#ffffff",
                "textColor": "#333333",
                "borderRadius": "8px",
                "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
            }
        }
    
    @staticmethod
    async def _generate_password_page_config(redirect_url: str = "/settings") -> Dict[str, Any]:
        """비밀번호 입력 페이지 UI 구성"""
        return {
            "title": "비밀번호 입력",
            "version": VERSION,
            "layout": {
                "type": "container",
                "direction": "vertical",
                "children": [
                    {
                        "type": "header",
                        "title": "관리자 인증",
                        "subtitle": "비밀번호를 입력하세요",
                        "style": {
                            "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                            "color": "white",
                            "padding": "12px",
                            "textAlign": "center",
                            "marginBottom": "5px"
                        }
                    },
                    {
                        "type": "container",
                        "style": {
                            "padding": "5px",
                            "maxWidth": "800px",
                            "margin": "0 auto",
                            "maxHeight": "calc(100vh - 68px)",
                            "display": "flex",
                            "flexDirection": "column",
                            "justifyContent": "flex-start",
                            "alignItems": "center",
                            "overflowY": "auto",
                            "overflowX": "hidden",
                            "gap": "5px"
                        },
                        "className": "sdui-password-container",
                        "children": [
                            {
                                "type": "password_input",
                                "id": "password-input",
                                "placeholder": "관리자 비밀번호",
                                "showError": True,
                                "style": {
                                    "marginBottom": "0px",
                                    "width": "100%",
                                    "maxWidth": "100%"
                                }
                            },
                            {
                                "type": "numpad",
                                "passwordInputId": "password-input",
                                "maxLength": 10,
                                "style": {
                                    "marginBottom": "0px"
                                }
                            },
                            {
                                "type": "button",
                                "label": "입력완료",
                                "action": {
                                    "type": "verify_password",
                                    "passwordInputId": "password-input",
                                    "redirectUrl": redirect_url
                                },
                                "style": {
                                    "variant": "primary",
                                    "fullWidth": True
                                }
                            }
                        ]
                    },
                    WebUIService._get_navigation_bar(is_main_page=False, page_name="password")
                ]
            },
            "theme": {
                "primaryColor": "#667eea",
                "secondaryColor": "#764ba2",
                "backgroundColor": "#ffffff",
                "textColor": "#333333",
                "borderRadius": "8px",
                "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
            }
        }
    
    @staticmethod
    async def handle_ui_action(action_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        UI 액션 처리
        
        Args:
            action_type: 액션 타입
            payload: 액션 페이로드
            
        Returns:
            처리 결과
        """
        # 향후 확장 가능한 액션 처리 로직
        return {
            "success": True,
            "message": f"Action {action_type} processed",
            "data": payload
        }


# 전역 서비스 인스턴스
web_ui_service = WebUIService()

