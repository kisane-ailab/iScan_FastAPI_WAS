/**
 * 상품 서비스
 * 상품 관련 비즈니스 로직
 */

(function() {
    'use strict';
    
    // 로딩 중 플래그 (중복 호출 방지)
    let isLoadingProductData = false;
    
    // 검색어 저장
    let currentSearchTerm = '';
    
    // 현재 선택된 상품 정보 저장 (ServMan 실행 요청용)
    let currentSelectedProduct = null;
    
    // SearchKeyboard 인스턴스 참조
    let searchKeyboardInstance = null;
    
    /**
     * 상품 데이터 로드
     */
    async function loadProductData(searchTerm = '') {
        console.log('[ProductService] loadProductData 함수 호출됨');
        
        // 이미 로딩 중이면 무시
        if (isLoadingProductData) {
            console.log('[ProductService] 이미 로딩 중입니다. 중복 호출 무시');
            return;
        }
        
        // 먼저 테이블 자체를 찾기
        const tableElement = document.querySelector('#product-table');
        if (!tableElement) {
            console.error('[ProductService] #product-table을 찾을 수 없습니다');
            return;
        }
        
        console.log('[ProductService] 테이블 요소 찾음:', tableElement);
        
        // 로딩 시작
        isLoadingProductData = true;
        
        // tbody를 찾거나 생성
        let tbody = tableElement.querySelector('tbody');
        if (!tbody) {
            console.log('[ProductService] tbody가 없어서 생성합니다');
            tbody = document.createElement('tbody');
            tableElement.appendChild(tbody);
        }
        
        const headerSubtitle = document.querySelector('.sdui-header p');
        
        console.log('[ProductService] tbody 요소 준비 완료, 데이터 로드 시작');
        
        // 로딩 메시지 표시
        if (headerSubtitle) {
            headerSubtitle.textContent = '데이터베이스에서 상품 정보를 불러오는 중...';
        }
        
        // 로딩 인디케이터 표시 (공통 스타일 사용)
        tbody.innerHTML = `
            <tr>
                <td colspan="7">
                    <div class="loading-inline">
                        <div class="spinner"></div>
                        <div>데이터베이스에서 상품 정보를 불러오는 중...</div>
                    </div>
                </td>
            </tr>
        `;
        
        try {
            console.log('[ProductService] API 호출 시작: /api/products');
            
            // Utils.fetchWithTimeout 사용
            const fetchFn = (window.Utils && window.Utils.fetchWithTimeout) || fetch;
            // 검색어가 있으면 쿼리 파라미터에 추가
            let apiUrl = '/api/products?offset=0';
            if (searchTerm) {
                apiUrl += `&search=${encodeURIComponent(searchTerm)}`;
            }
            const response = await fetchFn(apiUrl, {}, 10000);
            console.log('[ProductService] API 응답 받음:', response.status);
            
            const result = await response.json();
            
            // 응답이 성공이 아니거나 result.success가 false인 경우
            if (!response.ok || (result && result.success === false)) {
                const errorMsg = result?.error || result?.detail || `HTTP ${response.status}: ${response.statusText}`;
                throw new Error(errorMsg);
            }
            
            if (result.success && result.data && result.data.length > 0) {
                console.log('[ProductService] 데이터 수신 성공:', result.data.length, '개');
                // 헤더 업데이트
                if (headerSubtitle) {
                    if (searchTerm) {
                        headerSubtitle.textContent = `검색 결과: "${searchTerm}" - 총 ${result.total}개의 상품`;
                    } else {
                        headerSubtitle.textContent = `총 ${result.total}개의 상품`;
                    }
                }
                
                // 테이블 데이터 업데이트
                tbody.innerHTML = '';
                result.data.forEach((product, index) => {
                    const row = document.createElement('tr');
                    row.style.cursor = 'pointer';
                    row.dataset.productId = product.id;
                    
                    // 스캔 이미지 개수 확인
                    const scanImageCount = product.scan_image_count !== undefined && product.scan_image_count !== null 
                        ? parseInt(product.scan_image_count) : 0;
                    const isScanAvailable = scanImageCount > 0;
                    
                    // iScan 사용 가능한 아이템 강조
                    if (isScanAvailable) {
                        // 연한 초록색 배경과 왼쪽 테두리로 강조
                        row.style.backgroundColor = index % 2 === 0 ? '#f0fdf4' : '#ecfdf5';
                        row.style.borderLeft = '4px solid #10b981';
                        row.style.fontWeight = '500';
                    } else {
                        row.style.backgroundColor = index % 2 === 0 ? '#ffffff' : '#f8f9fa';
                        row.style.borderLeft = '4px solid transparent';
                    }
                    
                    // 행 클릭 이벤트 추가
                    row.addEventListener('click', () => {
                        showProductDetailModal(product.id);
                    });
                    
                    // 호버 효과
                    const originalBgColor = row.style.backgroundColor;
                    row.addEventListener('mouseenter', () => {
                        if (isScanAvailable) {
                            row.style.backgroundColor = '#dcfce7';
                        } else {
                            row.style.backgroundColor = '#e8f0fe';
                        }
                    });
                    row.addEventListener('mouseleave', () => {
                        row.style.backgroundColor = originalBgColor;
                    });
                    
                    // 대표 이미지 표시 여부 확인 (값이 없거나 "default_"로 시작하면 빈 셀)
                    const thumbImage = product.thumb_image_file;
                    const showImageIcon = thumbImage && !thumbImage.startsWith('default_') ? '🖼️' : '';
                    
                    const cells = [
                        product.item_code || '',
                        product.item_name_default || '',
                        product.category_top || '',
                        product.base_amount ? `${parseFloat(product.base_amount).toLocaleString()}원` : '0원',
                        product.is_pos_use ? '사용' : '미사용',
                        product.scan_image_count !== undefined && product.scan_image_count !== null ? product.scan_image_count.toString() : '0',
                        showImageIcon
                    ];
                    
                    cells.forEach((cellContent, cellIndex) => {
                        const cell = document.createElement('td');
                        cell.style.padding = '12px';
                        cell.style.borderBottom = '1px solid #eee';
                        // 가격 컬럼은 우측 정렬
                        if (cellIndex === 3) {
                            cell.style.textAlign = 'right';
                        }
                        // POS사용, 스캔이미지 개수, 대표 이미지 컬럼은 중앙 정렬
                        else if (cellIndex === 4 || cellIndex === 5 || cellIndex === 6) {
                            cell.style.textAlign = 'center';
                        }
                        // POS사용이 "미사용"인 경우 빨간색으로 강조
                        if (cellIndex === 4 && cellContent === '미사용') {
                            cell.style.color = '#dc3545';
                            cell.style.fontWeight = '600';
                        }
                        cell.textContent = cellContent;
                        row.appendChild(cell);
                    });
                    
                    tbody.appendChild(row);
                });
                console.log('[ProductService] 테이블 업데이트 완료');
            } else {
                console.log('[ProductService] 데이터가 없습니다');
                // 데이터가 없는 경우
                if (headerSubtitle) {
                    if (searchTerm) {
                        headerSubtitle.textContent = `검색 결과: "${searchTerm}" - 상품을 찾을 수 없습니다`;
                    } else {
                        headerSubtitle.textContent = '상품이 없습니다';
                    }
                }
                const noDataMessage = searchTerm 
                    ? `"${searchTerm}"에 대한 검색 결과가 없습니다`
                    : '등록된 상품이 없습니다';
                tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 40px; color: #666;">${noDataMessage}</td></tr>`;
            }
        } catch (error) {
            console.error('[ProductService] 상품 데이터 로드 실패:', error);
            console.error('[ProductService] 에러 상세:', error.stack);
            
            // 에러 메시지 표시
            let errorMessage = '알 수 없는 오류가 발생했습니다';
            
            if (error.message) {
                if (error.message.includes('시간이 초과')) {
                    errorMessage = '데이터베이스 서버가 응답하지 않습니다. 서버 상태를 확인해주세요.';
                } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                    errorMessage = '네트워크 연결에 실패했습니다. 서버에 연결할 수 없습니다.';
                } else {
                    errorMessage = error.message;
                }
            }
            
            if (headerSubtitle) {
                headerSubtitle.textContent = '데이터베이스 연결 실패';
            }
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" style="text-align: center; padding: 40px; color: #c33;">
                        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
                            <strong style="font-size: 16px;">데이터베이스 연결에 실패했습니다</strong>
                            <span style="font-size: 14px; color: #666; margin-top: 5px;">
                                ${errorMessage}
                            </span>
                            <button onclick="ProductService.loadProductData()" style="margin-top: 15px; padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 14px;">
                                다시 시도
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        } finally {
            // 로딩 완료
            isLoadingProductData = false;
        }
    }
    
    /**
     * 상품 상세 정보 모달 표시
     */
    async function showProductDetailModal(productId) {
        console.log('[ProductService] 상품 상세 정보 로드 시작:', productId);
        
        // 기존 모달이 있으면 제거
        const existingModal = document.getElementById('product-detail-modal');
        if (existingModal) {
            existingModal.remove();
        }
        
        // 모달 오버레이 생성
        const overlay = document.createElement('div');
        overlay.id = 'product-detail-modal';
        overlay.style.position = 'fixed';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = '100%';
        overlay.style.height = '100%';
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        overlay.style.display = 'flex';
        overlay.style.justifyContent = 'center';
        overlay.style.alignItems = 'center';
        overlay.style.zIndex = '10000';
        overlay.style.animation = 'fadeIn 0.2s ease-in';
        
        // 오버레이 클릭 시 닫기
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                closeProductDetailModal();
            }
        });
        
        // 모달 박스 생성
        const modal = document.createElement('div');
        modal.style.backgroundColor = '#ffffff';
        modal.style.borderRadius = '12px';
        modal.style.padding = '30px';
        modal.style.maxWidth = '800px';
        modal.style.width = '90%';
        modal.style.maxHeight = '90vh';
        modal.style.overflowY = 'auto';
        modal.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
        modal.style.animation = 'slideUp 0.3s ease-out';
        modal.style.position = 'relative';
        
        // 모달 클릭 시 이벤트 전파 방지
        modal.addEventListener('click', (e) => {
            e.stopPropagation();
        });
        
        // 로딩 메시지 표시
        modal.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h2 style="margin: 0; color: #333;">상품 상세 정보</h2>
                <button id="close-modal-btn" style="background: #f0f0f0; border: 2px solid #ccc; border-radius: 50%; font-size: 36px; font-weight: bold; cursor: pointer; color: #333; padding: 0; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; transition: all 0.2s; line-height: 1;">&times;</button>
            </div>
            <div class="loading-inline" style="display: flex; align-items: center; gap: 10px; padding: 40px; justify-content: center;">
                <div class="spinner"></div>
                <div>상품 정보를 불러오는 중...</div>
            </div>
        `;
        
        overlay.appendChild(modal);
        document.body.appendChild(overlay);
        
        // 닫기 버튼 이벤트
        const loadingCloseBtn = document.getElementById('close-modal-btn');
        loadingCloseBtn.addEventListener('click', closeProductDetailModal);
        loadingCloseBtn.addEventListener('mouseenter', function() {
            this.style.background = '#e0e0e0';
            this.style.borderColor = '#999';
            this.style.transform = 'scale(1.1)';
        });
        loadingCloseBtn.addEventListener('mouseleave', function() {
            this.style.background = '#f0f0f0';
            this.style.borderColor = '#ccc';
            this.style.transform = 'scale(1)';
        });
        
        try {
            // 상품 상세 정보 가져오기
            const fetchFn = (window.Utils && window.Utils.fetchWithTimeout) || fetch;
            const response = await fetchFn(`/api/products/${productId}`, {}, 10000);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            if (!result.success || !result.data) {
                throw new Error('상품 정보를 가져올 수 없습니다');
            }
            
            const product = result.data;
            
            // 현재 선택된 상품 정보 저장 (ServMan 실행 요청용)
            currentSelectedProduct = product;
            
            // 상품 정보 포맷팅 함수
            const formatValue = (value, type = 'text') => {
                if (value === null || value === undefined) return '-';
                
                switch (type) {
                    case 'boolean':
                        return value ? '예' : '아니오';
                    case 'currency':
                        return `${parseFloat(value).toLocaleString()}원`;
                    case 'percent':
                        return `${parseFloat(value)}%`;
                    case 'date':
                        return new Date(value).toLocaleString('ko-KR');
                    case 'item_type':
                        const types = { 0: '자체제작', 1: '유통상품', 2: '선택상품', 3: 'Tray' };
                        return types[value] || value;
                    case 'order_unit':
                        return value === 0 ? '낱개' : '세트';
                    default:
                        return value.toString();
                }
            };
            
            // 대표 이미지 요청 함수
            async function requestThumbnailImage() {
                const thumbImageFile = product.thumb_image_file;
                // thumb_image_file이 있으면 요청
                if (thumbImageFile) {
                    const imageContainer = document.getElementById('product-thumbnail-container');
                    if (!imageContainer) return;
                    
                    try {
                        console.log('[ProductService] 대표 이미지 요청 시작:', thumbImageFile);
                        const response = await fetch('/api/edgeman/request-thumbnail', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                product: product
                            }),
                        });

                        if (response.ok) {
                            const result = await response.json();
                            if (result.success && result.image_data) {
                                console.log('[ProductService] 대표 이미지 수신 성공');
                                // 이미지 표시 영역에 이미지 추가
                                imageContainer.innerHTML = `
                                    <img src="${result.image_data}" 
                                         alt="대표 이미지" 
                                         style="max-width: 100%; max-height: 400px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); object-fit: contain;">
                                `;
                                imageContainer.style.display = 'block';
                            } else {
                                // 에러 메시지 표시
                                console.warn('[ProductService] 대표 이미지 응답에 이미지 데이터가 없습니다:', result);
                                const errorMessage = result.message || '대표 이미지를 불러올 수 없습니다.';
                                const errorDetail = result.error ? `\n${result.error}` : '';
                                imageContainer.innerHTML = `
                                    <div style="padding: 20px; text-align: center; color: #dc3545;">
                                        <div style="font-size: 48px; margin-bottom: 10px;">⚠️</div>
                                        <div style="font-weight: 600; margin-bottom: 8px; font-size: 16px;">${errorMessage}</div>
                                        ${errorDetail ? `<div style="font-size: 12px; color: #666; margin-top: 8px; word-break: break-all;">${errorDetail}</div>` : ''}
                                    </div>
                                `;
                                imageContainer.style.display = 'block';
                            }
                        } else {
                            // HTTP 오류 응답 처리
                            console.warn('[ProductService] 대표 이미지 요청 실패:', response.status);
                            let errorMessage = '대표 이미지를 불러올 수 없습니다.';
                            try {
                                const errorResult = await response.json();
                                errorMessage = errorResult.message || errorMessage;
                                const errorDetail = errorResult.error ? `\n${errorResult.error}` : '';
                                imageContainer.innerHTML = `
                                    <div style="padding: 20px; text-align: center; color: #dc3545;">
                                        <div style="font-size: 48px; margin-bottom: 10px;">⚠️</div>
                                        <div style="font-weight: 600; margin-bottom: 8px; font-size: 16px;">${errorMessage}</div>
                                        ${errorDetail ? `<div style="font-size: 12px; color: #666; margin-top: 8px; word-break: break-all;">${errorDetail}</div>` : ''}
                                    </div>
                                `;
                            } catch (e) {
                                imageContainer.innerHTML = `
                                    <div style="padding: 20px; text-align: center; color: #dc3545;">
                                        <div style="font-size: 48px; margin-bottom: 10px;">⚠️</div>
                                        <div style="font-weight: 600; margin-bottom: 8px; font-size: 16px;">${errorMessage}</div>
                                        <div style="font-size: 12px; color: #666; margin-top: 8px;">HTTP ${response.status}</div>
                                    </div>
                                `;
                            }
                            imageContainer.style.display = 'block';
                        }
                    } catch (error) {
                        // 네트워크 오류 등 예외 처리
                        console.warn('[ProductService] 대표 이미지 요청 중 오류:', error);
                        imageContainer.innerHTML = `
                            <div style="padding: 20px; text-align: center; color: #dc3545;">
                                <div style="font-size: 48px; margin-bottom: 10px;">⚠️</div>
                                <div style="font-weight: 600; margin-bottom: 8px; font-size: 16px;">대표 이미지를 불러올 수 없습니다.</div>
                                <div style="font-size: 12px; color: #666; margin-top: 8px; word-break: break-all;">${error.message || '알 수 없는 오류가 발생했습니다.'}</div>
                            </div>
                        `;
                        imageContainer.style.display = 'block';
                    }
                }
            }
            
            // 모달 내용 생성
            modal.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 2px solid #667eea; padding-bottom: 15px;">
                    <h2 style="margin: 0; color: #333;">상품 상세 정보</h2>
                    <button id="close-modal-btn" style="background: #f0f0f0; border: 2px solid #ccc; border-radius: 50%; font-size: 36px; font-weight: bold; cursor: pointer; color: #333; padding: 0; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; transition: all 0.2s; line-height: 1;">&times;</button>
                </div>
                <div id="product-thumbnail-container" style="display: none; margin-bottom: 20px; text-align: center; padding: 10px; background: #f8f9fa; border-radius: 8px;"></div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 8px; font-size: 16px;">상품 코드</label>
                        <div id="product-code-display" style="padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff; border-radius: 8px; font-size: 20px; font-weight: 700; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3); border: 2px solid #5568d3; cursor: pointer; transition: all 0.2s;" title="클릭하여 ServMan 실행">${formatValue(product.item_code)}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 8px; font-size: 16px;">상품명</label>
                        <div id="product-name-display" style="padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff; border-radius: 8px; font-size: 20px; font-weight: 700; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3); border: 2px solid #5568d3; cursor: pointer; transition: all 0.2s;" title="클릭하여 ServMan 실행">${formatValue(product.item_name_default)}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">스캔 이미지 개수</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px; ${(product.scan_image_count !== undefined && product.scan_image_count !== null && parseInt(product.scan_image_count) > 0) ? 'font-weight: 700;' : ''}">${formatValue(product.scan_image_count)}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">대표 이미지</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.thumb_image_file)}</div>
                    </div>

                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">벤더명</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.vendor_name)}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">바코드</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.barcode)}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">대분류</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.category_top)}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">중분류</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.category_mid)}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">소분류</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.category_low)}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">POS 사용</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px; ${product.is_pos_use ? '' : 'color: #dc3545; font-weight: 600;'}">${product.is_pos_use ? '사용' : '미사용'}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">가격</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.base_amount, 'currency')}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">통화 코드</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.currency_code)}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">부가세 포함</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.vat_included, 'boolean')}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">재고 수량</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.stock)}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">품절 여부</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.is_out_of_stock, 'boolean')}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">상품 유형</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.item_type, 'item_type')}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">주문 단위</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.order_unit, 'order_unit')}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">노출 우선순위</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.disp_priority)}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">할인 여부</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.is_discounted, 'boolean')}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">할인율</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.discount_rate, 'percent')}</div>
                    </div>
                    <div style="grid-column: 1 / -1;">
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">상품 설명</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px; min-height: 60px;">${formatValue(product.item_description_default)}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">생성일</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.created_at, 'date')}</div>
                    </div>
                    <div>
                        <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 5px;">수정일</label>
                        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">${formatValue(product.updated_at, 'date')}</div>
                    </div>
                </div>
            `;
            
            // 닫기 버튼 이벤트 재등록
            const closeBtn = document.getElementById('close-modal-btn');
            closeBtn.addEventListener('click', closeProductDetailModal);
            closeBtn.addEventListener('mouseenter', function() {
                this.style.background = '#e0e0e0';
                this.style.borderColor = '#999';
                this.style.transform = 'scale(1.1)';
            });
            closeBtn.addEventListener('mouseleave', function() {
                this.style.background = '#f0f0f0';
                this.style.borderColor = '#ccc';
                this.style.transform = 'scale(1)';
            });
            
            // 상품 코드와 상품명 클릭 이벤트 추가 (ServMan 실행)
            const productCodeDisplay = document.getElementById('product-code-display');
            const productNameDisplay = document.getElementById('product-name-display');
            
            if (productCodeDisplay) {
                productCodeDisplay.addEventListener('click', requestRunServMan);
                productCodeDisplay.addEventListener('mouseenter', function() {
                    this.style.transform = 'scale(1.02)';
                    this.style.boxShadow = '0 6px 16px rgba(102, 126, 234, 0.4)';
                });
                productCodeDisplay.addEventListener('mouseleave', function() {
                    this.style.transform = 'scale(1)';
                    this.style.boxShadow = '0 4px 12px rgba(102, 126, 234, 0.3)';
                });
            }
            
            if (productNameDisplay) {
                productNameDisplay.addEventListener('click', requestRunServMan);
                productNameDisplay.addEventListener('mouseenter', function() {
                    this.style.transform = 'scale(1.02)';
                    this.style.boxShadow = '0 6px 16px rgba(102, 126, 234, 0.4)';
                });
                productNameDisplay.addEventListener('mouseleave', function() {
                    this.style.transform = 'scale(1)';
                    this.style.boxShadow = '0 4px 12px rgba(102, 126, 234, 0.3)';
                });
            }
            
            // 대표 이미지 요청 (비동기로 실행)
            requestThumbnailImage();
            
        } catch (error) {
            console.error('[ProductService] 상품 상세 정보 로드 실패:', error);
            modal.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h2 style="margin: 0; color: #333;">상품 상세 정보</h2>
                    <button id="close-modal-btn" style="background: #f0f0f0; border: 2px solid #ccc; border-radius: 50%; font-size: 36px; font-weight: bold; cursor: pointer; color: #333; padding: 0; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; transition: all 0.2s; line-height: 1;">&times;</button>
                </div>
                <div style="text-align: center; padding: 40px; color: #c33;">
                    <div style="font-size: 48px; margin-bottom: 15px;">⚠️</div>
                    <div style="font-size: 16px; margin-bottom: 10px;"><strong>상품 정보를 불러올 수 없습니다</strong></div>
                    <div style="font-size: 14px; color: #666;">${error.message || '알 수 없는 오류가 발생했습니다'}</div>
                </div>
            `;
            const errorCloseBtn = document.getElementById('close-modal-btn');
            errorCloseBtn.addEventListener('click', closeProductDetailModal);
            errorCloseBtn.addEventListener('mouseenter', function() {
                this.style.background = '#e0e0e0';
                this.style.borderColor = '#999';
                this.style.transform = 'scale(1.1)';
            });
            errorCloseBtn.addEventListener('mouseleave', function() {
                this.style.background = '#f0f0f0';
                this.style.borderColor = '#ccc';
                this.style.transform = 'scale(1)';
            });
        }
    }
    
    /**
     * 에러 팝업 표시 (sdui.js의 showErrorPopup과 동일한 방식)
     */
    function showErrorPopup(message) {
        // 이미 제거 중이면 무시
        if (window.productServiceErrorPopupRemoving) {
            return;
        }
        
        // 기존 팝업이 있으면 즉시 제거 (애니메이션 없이)
        const existingPopup = document.getElementById('product-service-error-popup');
        if (existingPopup) {
            try {
                existingPopup.style.display = 'none';
                if (existingPopup.parentNode) {
                    existingPopup.parentNode.removeChild(existingPopup);
                }
            } catch (e) {
                // 무시
            }
        }

        // 팝업 오버레이 생성
        const overlay = document.createElement('div');
        overlay.id = 'product-service-error-popup';
        overlay.style.position = 'fixed';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = '100%';
        overlay.style.height = '100%';
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        overlay.style.display = 'flex';
        overlay.style.justifyContent = 'center';
        overlay.style.alignItems = 'center';
        overlay.style.zIndex = '10000';
        overlay.style.animation = 'fadeIn 0.2s ease-in';

        // 팝업 박스 생성
        const popup = document.createElement('div');
        popup.style.backgroundColor = '#ffffff';
        popup.style.borderRadius = '12px';
        popup.style.padding = '30px 40px';
        popup.style.maxWidth = '400px';
        popup.style.width = '90%';
        popup.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
        popup.style.textAlign = 'center';
        popup.style.animation = 'slideUp 0.3s ease-out';

        // 에러 아이콘
        const icon = document.createElement('div');
        icon.textContent = '⚠️';
        icon.style.fontSize = '48px';
        icon.style.marginBottom = '15px';
        popup.appendChild(icon);

        // 에러 메시지
        const messageDiv = document.createElement('div');
        messageDiv.textContent = message;
        messageDiv.style.fontSize = '18px';
        messageDiv.style.color = '#dc3545';
        messageDiv.style.marginBottom = '25px';
        messageDiv.style.lineHeight = '1.5';
        messageDiv.style.whiteSpace = 'pre-line';  // \n을 줄바꿈으로 처리
        popup.appendChild(messageDiv);

        // 확인 버튼
        const button = document.createElement('button');
        button.textContent = '확인';
        button.style.backgroundColor = '#667eea';
        button.style.color = '#ffffff';
        button.style.border = 'none';
        button.style.borderRadius = '8px';
        button.style.padding = '12px 40px';
        button.style.fontSize = '16px';
        button.style.fontWeight = '600';
        button.style.cursor = 'pointer';
        button.style.transition = 'all 0.2s ease';
        button.style.width = '100%';
        button.style.maxWidth = '200px';

        // 버튼 호버 효과
        button.addEventListener('mouseenter', function() {
            this.style.backgroundColor = '#5568d3';
            this.style.transform = 'scale(1.05)';
        });
        button.addEventListener('mouseleave', function() {
            this.style.backgroundColor = '#667eea';
            this.style.transform = 'scale(1)';
        });

        // 팝업 제거 함수 (중복 실행 방지)
        const removePopup = (e) => {
            if (e) {
                e.stopPropagation();
                e.preventDefault();
            }
            
            if (window.productServiceErrorPopupRemoving || !overlay || !overlay.parentNode) {
                return;
            }
            
            window.productServiceErrorPopupRemoving = true;
            overlay.style.pointerEvents = 'none';
            overlay.style.animation = 'fadeOut 0.2s ease-out';
            overlay.style.opacity = '0';
            
            setTimeout(() => {
                try {
                    if (overlay && overlay.parentNode) {
                        overlay.parentNode.removeChild(overlay);
                    }
                } catch (e) {
                    // 이미 제거된 경우 무시
                }
                setTimeout(() => {
                    window.productServiceErrorPopupRemoving = false;
                }, 100);
            }, 200);
        };

        // 버튼 클릭 시 팝업 제거
        button.addEventListener('click', removePopup);
        
        // 오버레이 클릭 시 팝업 제거
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                removePopup(e);
            }
        });

        popup.appendChild(button);
        overlay.appendChild(popup);
        document.body.appendChild(overlay);

        // 애니메이션 스타일 추가 (이미 있으면 추가하지 않음)
        if (!document.getElementById('product-service-error-popup-styles')) {
            const style = document.createElement('style');
            style.id = 'product-service-error-popup-styles';
            style.textContent = `
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                @keyframes fadeOut {
                    from { opacity: 1; }
                    to { opacity: 0; }
                }
                @keyframes slideUp {
                    from {
                        transform: translateY(20px);
                        opacity: 0;
                    }
                    to {
                        transform: translateY(0);
                        opacity: 1;
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    /**
     * ServMan 프로그램 실행 요청
     * 
     * 전송되는 JSON 예시:
     * {
     *   "product": {
     *     "id": 123,
     *     "vendor_id": 1,
     *     "vendor_name": "cheonsang_seongsu",
     *     "vendor_display_name": "천상성수",
     *     "item_code": "ITEM001",
     *     "barcode": "8801234567890",
     *     "item_name_default": "테스트 상품명",
     *     "item_description_default": "상품 설명입니다.",
     *     "category_top": "식품",
     *     "category_mid": "과자",
     *     "category_low": "스낵",
     *     "currency_code": "KRW",
     *     "base_amount": 5000.0,
     *     "vat_included": true,
     *     "is_pos_use": true,
     *     "is_deleted": false,
     *     "stock": 100,
     *     "is_out_of_stock": false,
     *     "item_type": 0,
     *     "order_unit": 0,
     *     "disp_priority": 1,
     *     "is_discounted": false,
     *     "discount_rate": 0.0,
     *     "scan_image_count": 5,
     *     "thumb_image_file": "thumb_item001.jpg",
     *     "similar_item_group": null,
     *     "option_groups": null,
     *     "created_at": "2025-12-03T10:30:00",
     *     "updated_at": "2025-12-03T15:45:00"
     *   }
     * }
     * 
     * 주요 필드:
     * - id: 상품 고유 ID
     * - vendor_id: 벤더 ID
     * - vendor_name: 벤더명
     * - vendor_display_name: 벤더 표시명
     * - item_code: 상품 코드
     * - barcode: 바코드
     * - item_name_default: 상품명
     * - item_description_default: 상품 설명
     * - category_top/mid/low: 대/중/소분류
     * - currency_code: 통화 코드 (예: "KRW")
     * - base_amount: 기본 가격 (float)
     * - vat_included: 부가세 포함 여부 (boolean)
     * - is_pos_use: POS 사용 여부 (boolean)
     * - stock: 재고 수량
     * - is_out_of_stock: 품절 여부 (boolean)
     * - item_type: 상품 유형 (0: 자체제작, 1: 유통상품, 2: 선택상품, 3: Tray)
     * - order_unit: 주문 단위 (0: 낱개, 1: 세트)
     * - scan_image_count: 스캔 이미지 개수
     * - thumb_image_file: 대표 이미지 파일명
     * - similar_item_group: 유사 상품 그룹 (JSON 또는 null)
     * - option_groups: 옵션 그룹 (JSON 또는 null)
     * - created_at: 생성일시
     * - updated_at: 수정일시
     */
    async function requestRunServMan() {
        console.log('[ProductService] ServMan 실행 요청 시작');
        
        // 현재 선택된 상품 정보 확인
        if (!currentSelectedProduct) {
            console.warn('[ProductService] 선택된 상품 정보가 없습니다.');
            showErrorPopup('상품 정보를 찾을 수 없습니다.');
            return;
        }
        
        try {
            const response = await fetch('/api/edgeman/run-servman', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    product: currentSelectedProduct
                }),
            });

            if (response.ok) {
                const result = await response.json();
                // EdgeMan 서버 통신 실패 경고가 있는 경우 에러 팝업 표시
                if (result.warning) {
                    showErrorPopup('EdgeMan 과 통신할 수 없습니다.\nServMan 프로그램을 실행할 수 없습니다.');
                    return;
                }
                console.log('[ProductService] ServMan 실행 요청 성공:', result);
            } else {
                const errorResult = await response.json().catch(() => ({}));
                const errorMessage = errorResult.message || errorResult.detail || 'EdgeMan 과 통신할 수 없습니다.\nServMan 프로그램을 실행할 수 없습니다.';
                showErrorPopup(errorMessage);
            }
        } catch (error) {
            console.warn('[ProductService] ServMan 실행 요청 중 오류 발생:', error);
            // 네트워크 오류 등으로 서버 요청이 실패한 경우 에러 팝업 표시
            showErrorPopup('EdgeMan 과 통신할 수 없습니다.\nServMan 프로그램을 실행할 수 없습니다.');
        }
    }
    
    /**
     * 상품 상세 정보 모달 닫기
     */
    function closeProductDetailModal() {
        const overlay = document.getElementById('product-detail-modal');
        if (overlay) {
            overlay.style.animation = 'fadeOut 0.2s ease-out';
            setTimeout(() => {
                overlay.remove();
            }, 200);
        }
    }
    
    /**
     * 검색 토글
     */
    function toggleSearch() {
        const searchModal = document.getElementById('product-search-modal');
        if (searchModal) {
            // 이미 열려있으면 닫기
            closeSearchModal();
        } else {
            // 검색 모달 열기
            showSearchModal();
        }
    }
    
    /**
     * 검색 모달 표시
     */
    function showSearchModal() {
        // 기존 모달이 있으면 제거
        const existingModal = document.getElementById('product-search-modal');
        if (existingModal) {
            existingModal.remove();
        }
        
        // 모달 오버레이 생성
        const overlay = document.createElement('div');
        overlay.id = 'product-search-modal';
        overlay.style.position = 'fixed';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = '100%';
        overlay.style.height = '100%';
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        overlay.style.display = 'flex';
        overlay.style.justifyContent = 'center';
        overlay.style.alignItems = 'center';
        overlay.style.zIndex = '10000';
        overlay.style.animation = 'fadeIn 0.2s ease-in';
        
        // 오버레이 클릭 시 닫기
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                closeSearchModal();
            }
        });
        
        // 모달 박스 생성 (키보드를 포함하도록 높이 조정)
        const modal = document.createElement('div');
        modal.style.backgroundColor = '#ffffff';
        modal.style.borderRadius = '12px';
        modal.style.padding = '30px';
        modal.style.maxWidth = '900px';
        modal.style.width = '95%';
        modal.style.maxHeight = '95vh';
        modal.style.display = 'flex';
        modal.style.flexDirection = 'column';
        modal.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
        modal.style.animation = 'slideUp 0.3s ease-out';
        modal.style.position = 'relative';
        modal.style.overflowY = 'auto';
        modal.style.overflowX = 'hidden';
        
        // 모달 클릭 시 이벤트 전파 방지
        modal.addEventListener('click', (e) => {
            e.stopPropagation();
        });
        
        // 검색 모달 상단 내용
        const modalContent = document.createElement('div');
        modalContent.style.flexShrink = '0';
        modalContent.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 2px solid #667eea; padding-bottom: 15px;">
                <h2 style="margin: 0; color: #333;">상품 검색</h2>
                <button id="close-search-btn" style="background: #f0f0f0; border: 2px solid #ccc; border-radius: 50%; font-size: 36px; font-weight: bold; cursor: pointer; color: #333; padding: 0; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; transition: all 0.2s; line-height: 1;">&times;</button>
            </div>
            <div style="margin-bottom: 20px;">
                <label style="display: block; font-weight: bold; color: #667eea; margin-bottom: 10px;">검색어 입력</label>
                <input type="text" id="search-input" placeholder="상품코드, 상품명, 바코드로 검색" 
                    style="width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 5px; font-size: 16px; box-sizing: border-box; caret-color: #667eea; outline: none;"
                    value="${currentSearchTerm}" autocomplete="off" spellcheck="false">
                <div style="margin-top: 10px; font-size: 14px; color: #666;">
                    상품코드, 상품명, 바코드에서 검색합니다.
                </div>
            </div>
            <div style="display: flex; gap: 10px; justify-content: flex-end; margin-bottom: 20px;">
                <button id="clear-search-btn" style="padding: 12px 24px; background: #f0f0f0; border: 2px solid #ccc; border-radius: 5px; cursor: pointer; font-size: 16px; color: #333; transition: all 0.2s;">
                    초기화
                </button>
                <button id="search-btn" style="padding: 12px 24px; background: #667eea; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; color: white; transition: all 0.2s;">
                    검색
                </button>
            </div>
        `;
        
        // 키보드 컨테이너 생성
        const keyboardContainer = document.createElement('div');
        keyboardContainer.id = 'search-keyboard-container';
        keyboardContainer.style.flexShrink = '0';
        keyboardContainer.style.borderTop = '2px solid #dee2e6';
        keyboardContainer.style.paddingTop = '10px';
        keyboardContainer.style.paddingBottom = '10px';
        keyboardContainer.style.width = '100%';
        keyboardContainer.style.maxWidth = '100%';
        keyboardContainer.style.overflowX = 'hidden';
        keyboardContainer.style.overflowY = 'visible';
        keyboardContainer.style.boxSizing = 'border-box';
        
        modal.appendChild(modalContent);
        modal.appendChild(keyboardContainer);
        
        overlay.appendChild(modal);
        document.body.appendChild(overlay);
        
        // 이벤트 리스너 등록 (먼저 요소들을 가져와야 함)
        const closeBtn = document.getElementById('close-search-btn');
        const searchBtn = document.getElementById('search-btn');
        const clearBtn = document.getElementById('clear-search-btn');
        const searchInput = document.getElementById('search-input');
        
        // 키보드 렌더링 (searchInput이 선언된 후에 호출)
        // SearchKeyboard 모듈 사용
        if (window.SearchKeyboard) {
            // 기존 인스턴스가 있으면 정리
            if (searchKeyboardInstance) {
                searchKeyboardInstance.destroy();
                searchKeyboardInstance = null;
            }
            
            // 새 인스턴스 생성
            try {
                searchKeyboardInstance = new window.SearchKeyboard(keyboardContainer, searchInput);
            } catch (error) {
                console.error('[ProductService] SearchKeyboard 생성 실패:', error);
                searchKeyboardInstance = null;
            }
        } else {
            console.error('[ProductService] SearchKeyboard 모듈을 찾을 수 없습니다.');
        }
        
        closeBtn.addEventListener('click', closeSearchModal);
        closeBtn.addEventListener('mouseenter', function() {
            this.style.background = '#e0e0e0';
            this.style.borderColor = '#999';
            this.style.transform = 'scale(1.1)';
        });
        closeBtn.addEventListener('mouseleave', function() {
            this.style.background = '#f0f0f0';
            this.style.borderColor = '#ccc';
            this.style.transform = 'scale(1)';
        });
        
        searchBtn.addEventListener('click', () => {
            const searchTerm = searchInput.value.trim();
            currentSearchTerm = searchTerm;
            closeSearchModal();
            loadProductData(searchTerm);
        });
        
        searchBtn.addEventListener('mouseenter', function() {
            this.style.background = '#5568d3';
        });
        searchBtn.addEventListener('mouseleave', function() {
            this.style.background = '#667eea';
        });
        
        clearBtn.addEventListener('click', () => {
            searchInput.value = '';
            currentSearchTerm = '';
            closeSearchModal();
            loadProductData('');
        });
        
        clearBtn.addEventListener('mouseenter', function() {
            this.style.background = '#e0e0e0';
        });
        clearBtn.addEventListener('mouseleave', function() {
            this.style.background = '#f0f0f0';
        });
        
        // 입력 필드 클릭 시 포커스 및 커서 표시
        searchInput.addEventListener('click', () => {
            searchInput.focus();
            // 커서를 끝으로 이동
            const length = searchInput.value.length;
            searchInput.setSelectionRange(length, length);
        });
        
        // 입력 필드 포커스 시 스타일 강조 및 커서 표시
        searchInput.addEventListener('focus', () => {
            searchInput.style.borderColor = '#667eea';
            searchInput.style.boxShadow = '0 0 0 2px rgba(102, 126, 234, 0.2)';
            // 커서를 끝으로 이동하여 표시
            setTimeout(() => {
                const length = searchInput.value.length;
                searchInput.setSelectionRange(length, length);
            }, 10);
        });
        
        searchInput.addEventListener('blur', () => {
            searchInput.style.borderColor = '#ddd';
            searchInput.style.boxShadow = 'none';
        });
        
        // 입력 필드 클릭 시 커서 위치 표시
        searchInput.addEventListener('click', (e) => {
            // 클릭한 위치로 커서 이동
            const clickPosition = e.target.selectionStart || 0;
            setTimeout(() => {
                searchInput.setSelectionRange(clickPosition, clickPosition);
            }, 10);
        });
        
        // 초기 포커스 설정
        setTimeout(() => {
            searchInput.focus();
            const length = searchInput.value.length;
            searchInput.setSelectionRange(length, length);
        }, 100);
    }
    
    /**
     * 검색 모달 닫기
     */
    function closeSearchModal() {
        const modal = document.getElementById('product-search-modal');
        if (modal) {
            modal.style.animation = 'fadeOut 0.2s ease-out';
            setTimeout(() => {
                // 모달이 완전히 제거된 후에 SearchKeyboard 인스턴스 정리
                if (searchKeyboardInstance) {
                    searchKeyboardInstance.destroy();
                    searchKeyboardInstance = null;
                }
                modal.remove();
            }, 200);
        } else {
            // 모달이 없는 경우에도 인스턴스 정리
            if (searchKeyboardInstance) {
                searchKeyboardInstance.destroy();
                searchKeyboardInstance = null;
            }
        }
    }
    
    // 전역으로 노출
    window.ProductService = {
        loadProductData,
        showProductDetailModal,
        closeProductDetailModal,
        toggleSearch
    };
    
    console.log('[ProductService] 초기화 완료');
})();

