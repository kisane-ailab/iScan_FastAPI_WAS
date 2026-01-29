/**
 * 공통 초기화 스크립트
 * 모든 HTML 페이지에서 공통으로 사용하는 초기화 로직
 * 
 * 이 파일은 다른 모듈들을 통합하는 진입점 역할을 합니다.
 * 필요한 모듈을 자동으로 로드합니다.
 */

(function() {
    'use strict';
    
    // 모듈 로드 함수
    function loadScript(src) {
        return new Promise((resolve, reject) => {
            // 이미 로드되어 있는지 확인 (정확한 매칭을 위해 src 속성 직접 확인)
            const scripts = document.querySelectorAll('script[src]');
            let alreadyLoaded = false;
            for (let script of scripts) {
                if (script.src && (script.src.includes(src) || script.getAttribute('src') === src)) {
                    alreadyLoaded = true;
                    break;
                }
            }
            
            if (alreadyLoaded) {
                // 이미 로드된 스크립트라도 완전히 실행될 때까지 대기
                setTimeout(resolve, 50);
                return;
            }
            
            const script = document.createElement('script');
            script.src = src;
            script.onload = () => {
                // 스크립트가 로드된 후 약간의 지연을 두어 실행 완료 보장
                setTimeout(resolve, 50);
            };
            script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
            document.head.appendChild(script);
        });
    }
    
    // 필요한 모듈 로드
    async function loadModules() {
        const modules = [];
        
        // utils.js 로드
        if (!window.Utils) {
            modules.push(loadScript('/static/utils.js'));
        }
        
        // sdui-loader.js 로드
        if (!window.SDUILoader) {
            modules.push(loadScript('/static/sdui-loader.js'));
        }
        
        // product-service.js는 선택적 (product 페이지에서만 필요)
        // 필요할 때만 로드하도록 함
        
        await Promise.all(modules);
    }
    
    // SDUICommon 통합 객체 생성
    function createSDUICommon() {
        // 필수 모듈 확인
        if (!window.SDUILoader) {
            return false;
        }
        
        // SDUICommon 통합 객체 생성 (하위 호환성 유지)
        window.SDUICommon = {
            // SDUILoader 기능
            loadSDUI: window.SDUILoader.loadSDUI,
            loadUIConfig: window.SDUILoader.loadUIConfig,
            startApp: window.SDUILoader.startApp
        };
        
        // ProductService가 있으면 추가
        if (window.ProductService) {
            window.SDUICommon.loadProductData = window.ProductService.loadProductData;
            window.loadProductData = window.ProductService.loadProductData;
        }
        
        console.log('[common.js] SDUICommon 초기화 완료');
        console.log('[common.js] 사용 가능한 모듈:', {
            Utils: !!window.Utils,
            ProductService: !!window.ProductService,
            SDUILoader: !!window.SDUILoader,
            SDUICommon: !!window.SDUICommon
        });
        
        return true;
    }
    
    // 모듈 로드 및 초기화
    loadModules().then(() => {
        // 모듈이 이미 로드되어 있으면 즉시 생성, 아니면 대기
        if (!createSDUICommon()) {
            let attempts = 0;
            const maxAttempts = 50;
            
            const check = () => {
                attempts++;
                if (createSDUICommon() || attempts >= maxAttempts) {
                    if (attempts >= maxAttempts) {
                        console.error('[common.js] 모듈 초기화 실패:', {
                            Utils: !!window.Utils,
                            ProductService: !!window.ProductService,
                            SDUILoader: !!window.SDUILoader
                        });
                    }
                } else {
                    setTimeout(check, 100);
                }
            };
            
            check();
        }
    }).catch(error => {
        console.error('[common.js] 모듈 로드 실패:', error);
    });
})();
