/**
 * 공통 유틸리티 함수
 * 모든 페이지에서 공통으로 사용하는 유틸리티 함수들
 */

(function() {
    'use strict';
    
    // console 폴백 (구형 브라우저 대응)
    if (typeof console === 'undefined') {
        window.console = {
            log: function() {},
            error: function() {},
            warn: function() {}
        };
    }
    
    /**
     * 타임아웃이 있는 fetch
     * @param {string} url - 요청 URL
     * @param {Object} options - fetch 옵션
     * @param {number} timeout - 타임아웃 시간 (ms, 기본값: 10000)
     * @returns {Promise<Response>}
     */
    async function fetchWithTimeout(url, options = {}, timeout = 10000) {
        // AbortController 지원 여부 확인
        if (typeof AbortController === 'undefined') {
            // AbortController를 지원하지 않는 브라우저의 경우 기본 fetch 사용
            console.warn('[Utils] AbortController를 지원하지 않습니다. 타임아웃 없이 fetch를 사용합니다.');
            return fetch(url, options);
        }
        
        const controller = new AbortController();
        const id = setTimeout(() => {
            try {
                controller.abort();
            } catch (e) {
                // abort 중 에러 무시
            }
        }, timeout);
        
        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            clearTimeout(id);
            return response;
        } catch (error) {
            clearTimeout(id);
            if (error.name === 'AbortError' || error.name === 'TimeoutError') {
                throw new Error('요청 시간이 초과되었습니다. 데이터베이스 서버가 응답하지 않습니다.');
            }
            throw error;
        }
    }
    
    // 전역으로 노출
    window.Utils = {
        fetchWithTimeout
    };
})();

