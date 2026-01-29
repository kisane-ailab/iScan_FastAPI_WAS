/**
 * SDUI 로더
 * SDUI 스크립트 로드 및 UI 구성 로드 기능
 */

(function() {
    'use strict';
    
    /**
     * sdui.js 스크립트 로드
     * @param {Function} onLoad - 로드 완료 콜백
     * @param {Function} onError - 로드 실패 콜백
     */
    function loadSDUI(onLoad, onError) {
        try {
            const script = document.createElement('script');
            // 캐시 무효화를 위한 버전 파라미터 추가
            script.src = '/static/sdui.js?v=' + Date.now();
            script.onload = function() {
                if (typeof SDUIRenderer === 'undefined') {
                    const error = new Error('SDUIRenderer 클래스를 찾을 수 없습니다');
                    if (onError) onError(error);
                    return;
                }
                if (onLoad) onLoad();
            };
            script.onerror = function() {
                const error = new Error('sdui.js 파일을 로드할 수 없습니다');
                if (onError) onError(error);
            };
            document.head.appendChild(script);
        } catch (e) {
            if (onError) onError(e);
        }
    }
    
    /**
     * UI 구성 로드 및 렌더링
     * @param {string} page - 페이지 이름 (예: 'main', 'password', 'product', 'settings')
     * @param {Object} options - 추가 옵션
     * @param {string} options.redirect - 리다이렉트 URL (password 페이지용)
     */
    async function loadUIConfig(page = 'main', options = {}) {
        const app = document.getElementById('app');
        if (!app) {
            console.error('App container not found');
            return;
        }
        
        try {
            // URL 파라미터 구성
            const params = new URLSearchParams({ page });
            if (options.redirect) {
                params.append('redirect', options.redirect);
            }
            
            const response = await fetch(`/api/ui/config?${params.toString()}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const config = await response.json();
            
            if (typeof SDUIRenderer === 'undefined') {
                throw new Error('SDUIRenderer 클래스를 찾을 수 없습니다');
            }
            
            app.innerHTML = '';
            
            const renderer = new SDUIRenderer(config);
            renderer.render(app);
            
            // 페이지별 렌더링 후 콜백 실행
            if (page === 'product') {
                console.log('[loadUIConfig] 상품 페이지 감지됨, 데이터 로드 준비...');
                
                // 커스텀 이벤트 발생 (product.html에서도 감지 가능)
                const event = new CustomEvent('sdui-render-complete', { 
                    detail: { page: 'product' } 
                });
                document.dispatchEvent(event);
                
                // DOM이 완전히 렌더링된 후 데이터 로드
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        setTimeout(() => {
                            console.log('[loadUIConfig] 상품 페이지 렌더링 완료, 데이터 로드 시작...');
                            
                            const table = document.querySelector('#product-table');
                            console.log('[loadUIConfig] 테이블 요소 찾기 시도, 결과:', table ? '찾음' : '없음');
                            
                            // 전역 함수 사용
                            if (typeof window.ProductService !== 'undefined' && typeof window.ProductService.loadProductData === 'function') {
                                console.log('[loadUIConfig] ProductService.loadProductData 호출');
                                window.ProductService.loadProductData();
                            } else if (typeof window.SDUICommon !== 'undefined' && typeof window.SDUICommon.loadProductData === 'function') {
                                console.log('[loadUIConfig] SDUICommon.loadProductData 호출 (하위 호환)');
                                window.SDUICommon.loadProductData();
                            } else {
                                console.error('[loadUIConfig] loadProductData 함수를 찾을 수 없습니다');
                            }
                        }, 500);
                    });
                });
            } else {
                console.log('[loadUIConfig] 다른 페이지:', page);
            }
        } catch (error) {
            app.innerHTML = `
                <div class="error">
                    <h2>오류 발생</h2>
                    <p><strong>에러:</strong> ${error.message}</p>
                    <p>브라우저 콘솔(F12)을 확인하세요.</p>
                </div>
            `;
        }
    }
    
    /**
     * 앱 시작 함수
     * @param {string} page - 페이지 이름
     * @param {Object} options - 추가 옵션
     */
    function startApp(page = 'main', options = {}) {
        // DOM이 준비될 때까지 대기
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                loadSDUI(
                    () => loadUIConfig(page, options),
                    (error) => {
                        const app = document.getElementById('app');
                        if (app) {
                            app.innerHTML = `<div class="error"><h2>오류</h2><p>${error.message}</p></div>`;
                        }
                    }
                );
            });
        } else {
            // 이미 로드된 경우
            setTimeout(() => {
                loadSDUI(
                    () => loadUIConfig(page, options),
                    (error) => {
                        const app = document.getElementById('app');
                        if (app) {
                            app.innerHTML = `<div class="error"><h2>오류</h2><p>${error.message}</p></div>`;
                        }
                    }
                );
            }, 100);
        }
    }
    
    // 전역으로 노출
    window.SDUILoader = {
        loadSDUI,
        loadUIConfig,
        startApp
    };
})();

