/**
 * SDUI (Server-Driven UI) 렌더러
 * 서버에서 받은 JSON 구성 정보를 해석하여 동적으로 UI를 렌더링
 */

// 상수 정의
const RESOLUTION_CONSTANTS = {
    // 해상도 범위
    KIOSK: {
        MIN_HEIGHT: 1800,
        MIN_WIDTH: 1000,
        MAX_WIDTH: 1200
    },
    POS: {
        MIN_HEIGHT: 700,
        MAX_HEIGHT: 900,
        MIN_WIDTH: 1000,
        MAX_WIDTH: 1200
    },
    // 표준 해상도
    STANDARD: {
        POS_WIDTH: 1024,
        POS_HEIGHT: 768,
        KIOSK_WIDTH: 1024,
        KIOSK_HEIGHT: 1920
    },
    // 타이밍 상수 (ms)
    TIMING: {
        DEBOUNCE_DELAY: 100,
        RERENDER_DELAY_1: 100,
        RERENDER_DELAY_2: 300,
        STYLE_CHECK_DELAY: 50,
        POPUP_REMOVE_DELAY: 100
    }
};

// 디버깅 모드 플래그 (개발 환경에서는 true로 설정)
const DEBUG_MODE = window.location.hostname === 'localhost' || 
                   window.location.hostname === '127.0.0.1' ||
                   window.location.search.includes('debug=true');

// 디버깅 로그 헬퍼
const debugLog = (...args) => {
    if (DEBUG_MODE) {
        console.log(...args);
    }
};

const debugWarn = (...args) => {
    if (DEBUG_MODE) {
        console.warn(...args);
    }
};

// 디바운스 헬퍼 함수
const debounce = (func, delay) => {
    let timeoutId;
    return (...args) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(null, args), delay);
    };
};

// 스타일 재적용 헬퍼 (여러 시점에 적용)
const applyStylesWithRetry = (applyFn, delays = [0, RESOLUTION_CONSTANTS.TIMING.RERENDER_DELAY_1, RESOLUTION_CONSTANTS.TIMING.RERENDER_DELAY_2]) => {
    delays.forEach((delay, index) => {
        if (delay === 0) {
            applyFn();
        } else {
            setTimeout(() => {
                debugLog(`[applyStylesWithRetry] ${delay}ms 후 스타일 재적용 (${index + 1}/${delays.length})`);
                applyFn();
            }, delay);
        }
    });
};

class SDUIRenderer {
    constructor(config) {
        this.config = config;
        this.theme = config.theme || {};
        this.errorPopupRemoving = false; // 에러 팝업 제거 중 플래그
        this.keyboardVisible = false; // 키보드 표시 여부
        this.keyboardMode = 'en'; // 'en' 또는 'ko'
        this.keyboardLayout = 'letters'; // 'letters' 또는 'numbers'
        this.currentInputElement = null; // 현재 포커스된 입력 필드
    }
    
    /**
     * 해상도 타입 판별 헬퍼 함수
     * @returns {Object} {isKIOSK: boolean, isPOS: boolean, screenWidth: number, screenHeight: number}
     */
    static getResolutionType() {
        const screenWidth = window.innerWidth;
        const screenHeight = window.innerHeight;
        const { KIOSK, POS } = RESOLUTION_CONSTANTS;
        
        // KIOSK: 세로가 1800 이상이고 가로가 1000~1200 범위
        const isKIOSK = screenHeight >= KIOSK.MIN_HEIGHT && 
                       screenWidth >= KIOSK.MIN_WIDTH && 
                       screenWidth <= KIOSK.MAX_WIDTH;
        
        // POS: 세로가 700~900 범위이고 가로가 1000~1200 범위
        const isPOS = screenHeight >= POS.MIN_HEIGHT && 
                     screenHeight <= POS.MAX_HEIGHT && 
                     screenWidth >= POS.MIN_WIDTH && 
                     screenWidth <= POS.MAX_WIDTH;
        
        return { isKIOSK, isPOS, screenWidth, screenHeight };
    }

    /**
     * UI 렌더링 시작
     */
    render(container) {
        if (this.config.layout) {
            this.renderElement(this.config.layout, container);
        }
    }

    /**
     * 요소 렌더링 (재귀적)
     */
    renderElement(element, parent) {
        if (!element || !element.type) return;

        const elementMap = {
            'container': () => this.renderContainer(element, parent),
            'header': () => this.renderHeader(element, parent),
            'card': () => this.renderCard(element, parent),
            'icon_card': () => this.renderIconCard(element, parent),
            'button': () => this.renderButton(element, parent),
            'text': () => this.renderText(element, parent),
            'table': () => this.renderTable(element, parent),
            'navigation_bar': () => this.renderNavigationBar(element, parent),
            'password_input': () => this.renderPasswordInput(element, parent),
            'numpad': () => this.renderNumpad(element, parent),
            'touch_keyboard': () => this.renderTouchKeyboard(element, parent),
        };

        const renderer = elementMap[element.type];
        if (renderer) {
            renderer();
        } else {
            console.warn(`Unknown element type: ${element.type}`);
        }
    }

    /**
     * 컨테이너 렌더링
     */
    renderContainer(element, parent) {
        const container = document.createElement('div');
        
        // className이 있으면 적용
        if (element.className) {
            container.className = element.className;
        }
        
        this.applyStyles(container, element.style || {});
        
        if (element.direction === 'horizontal') {
            container.style.display = 'flex';
            container.style.flexDirection = 'row';
        } else {
            container.style.display = 'flex';
            container.style.flexDirection = 'column';
        }

        if (element.gap) {
            container.style.gap = element.gap;
        }

        if (element.children) {
            element.children.forEach(child => {
                this.renderElement(child, container);
            });
        }

        parent.appendChild(container);
    }

    /**
     * 헤더 렌더링
     */
    renderHeader(element, parent) {
        const header = document.createElement('header');
        
        // 해상도별 최적화
        const screenWidth = window.innerWidth;
        const screenHeight = window.innerHeight;
        
        let headerPadding, headerMargin, titleFontSize, subtitleFontSize, titleMargin, subtitleMargin;
            if (screenWidth === 1024) {
                if (screenHeight === 768) {
                    // POS 해상도 (1024x768) - 적절한 크기로 조정
                    headerPadding = '12px';
                    headerMargin = '0 0 5px 0';
                    titleFontSize = '24px';
                    subtitleFontSize = '16px';
                    titleMargin = '0';
                    subtitleMargin = '2px 0 0 0';
                } 
                else if (screenHeight === 1920) {
                    // KIOSK 해상도 (1024x1920)
                    headerPadding = '30px';
                    headerMargin = '0 0 10px 0';
                    titleFontSize = '28px';
                    subtitleFontSize = '18px';
                    titleMargin = '0';
                    subtitleMargin = '5px 0 0 0';
                } else {
                    const isSmall = screenHeight < 1000;
                    headerPadding = isSmall ? '12px' : '30px';
                    headerMargin = isSmall ? '0 0 5px 0' : '0 0 10px 0';
                    titleFontSize = isSmall ? '24px' : '28px';
                    subtitleFontSize = isSmall ? '16px' : '18px';
                    titleMargin = '0';
                    subtitleMargin = isSmall ? '3px 0 0 0' : '5px 0 0 0';
                }
            } else {
                const scale = Math.min(screenWidth / 1024, screenHeight / 768);
                headerPadding = Math.max(12, Math.min(30, 20 * scale)) + 'px';
                headerMargin = '0 0 ' + Math.max(5, Math.min(10, 8 * scale)) + 'px 0';
                titleFontSize = Math.max(24, Math.min(28, 26 * scale)) + 'px';
                subtitleFontSize = Math.max(16, Math.min(18, 17 * scale)) + 'px';
                titleMargin = '0';
                subtitleMargin = Math.max(3, Math.min(5, 4 * scale)) + 'px 0 0 0';
            }
        
        // 기본 스타일 적용
        const defaultStyle = {
            padding: headerPadding,
            margin: headerMargin,
            ...(element.style || {})
        };
        this.applyStyles(header, defaultStyle);

        if (element.title) {
            const title = document.createElement('h1');
            title.textContent = element.title;
            title.style.margin = titleMargin;
            title.style.fontSize = titleFontSize;
            header.appendChild(title);
        }

        if (element.subtitle) {
            const subtitle = document.createElement('p');
            subtitle.textContent = element.subtitle;
            subtitle.style.margin = subtitleMargin;
            subtitle.style.opacity = '0.9';
            subtitle.style.fontSize = subtitleFontSize;
            header.appendChild(subtitle);
        }

        parent.appendChild(header);
    }

    /**
     * 카드 렌더링
     */
    renderCard(element, parent) {
        const card = document.createElement('div');
        card.className = 'sdui-card';
        
        const defaultStyle = {
            backgroundColor: '#ffffff',
            borderRadius: this.theme.borderRadius || '8px',
            padding: '20px',
            margin: '10px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        };
        
        this.applyStyles(card, { ...defaultStyle, ...(element.style || {}) });

        if (element.title) {
            const title = document.createElement('h2');
            title.textContent = element.title;
            title.style.margin = '0 0 15px 0';
            title.style.fontSize = '18px';
            title.style.color = this.theme.textColor || '#333';
            card.appendChild(title);
        }

        if (element.children) {
            element.children.forEach(child => {
                this.renderElement(child, card);
            });
        }

        parent.appendChild(card);
    }

    /**
     * 아이콘 카드 렌더링
     */
    renderIconCard(element, parent) {
        const card = document.createElement('div');
        card.className = 'sdui-icon-card';
        
        // 해상도 타입 확인
        const { isKIOSK, isPOS } = SDUIRenderer.getResolutionType();
        
        const defaultStyle = {
            backgroundColor: '#ffffff',
            borderRadius: this.theme.borderRadius || '12px',
            padding: '30px',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            textAlign: 'center',
            transition: 'all 0.3s ease',
        };
        
        // 서버에서 받은 스타일 적용
        let finalStyle = { ...defaultStyle, ...(element.style || {}) };
        
        // 해상도별 크기 조정
        if (isKIOSK) {
            // 키오스크 해상도: 더 큰 크기로 조정
            finalStyle.width = '450px';
            finalStyle.height = '350px';
            finalStyle.padding = '50px';
        } else if (isPOS) {
            // POS 해상도: 키오스크보다는 작지만 기존보다 크게 조정
            finalStyle.width = '350px';
            finalStyle.height = '250px';
            finalStyle.padding = '40px';
        }
        // 기타 해상도는 서버에서 받은 원래 크기 유지
        
        this.applyStyles(card, finalStyle);

        // 아이콘
        if (element.icon) {
            const icon = document.createElement('div');
            icon.textContent = element.icon;
            // 해상도별 아이콘 크기 조정
            if (isKIOSK) {
                icon.style.fontSize = '100px';
                icon.style.marginBottom = '30px';
            } else if (isPOS) {
                icon.style.fontSize = '80px';
                icon.style.marginBottom = '25px';
            } else {
                icon.style.fontSize = '64px';
                icon.style.marginBottom = '20px';
            }
            card.appendChild(icon);
        }

        // 제목
        if (element.title) {
            const title = document.createElement('h3');
            title.textContent = element.title;
            title.style.margin = '0 0 10px 0';
            // 해상도별 제목 폰트 크기 조정
            if (isKIOSK) {
                title.style.fontSize = '32px';
            } else if (isPOS) {
                title.style.fontSize = '26px';
            } else {
                title.style.fontSize = '20px';
            }
            title.style.fontWeight = '600';
            title.style.color = this.theme.textColor || '#333';
            card.appendChild(title);
        }

        // 설명
        if (element.description) {
            const description = document.createElement('p');
            description.textContent = element.description;
            description.style.margin = '0';
            // 해상도별 설명 폰트 크기 조정
            if (isKIOSK) {
                description.style.fontSize = '22px';
            } else if (isPOS) {
                description.style.fontSize = '18px';
            } else {
                description.style.fontSize = '14px';
            }
            description.style.color = '#666';
            card.appendChild(description);
        }

        // 호버 효과
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 8px 12px rgba(0,0,0,0.15)';
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
        });

        // 액션 처리
        if (element.action) {
            card.addEventListener('click', () => {
                this.handleAction(element.action);
            });
        }

        parent.appendChild(card);
    }

    /**
     * 버튼 렌더링
     */
    renderButton(element, parent) {
        const button = document.createElement('button');
        button.textContent = element.label || 'Button';
        button.className = 'sdui-button';

        // 해상도별 크기 계산 함수
        const getButtonSize = () => {
            const screenWidth = window.innerWidth;
            const screenHeight = window.innerHeight;
            
            let buttonPadding, buttonFontSize, buttonMinHeight;
            if (screenWidth === 1024) {
                if (screenHeight === 768) {
                    // POS 해상도 (1024x768) - 더 크게 조정
                    buttonPadding = '10px 20px';
                    buttonFontSize = '18px';
                    buttonMinHeight = '42px';
                } else if (screenHeight === 1920) {
                    // KIOSK 해상도 (1024x1920) - 더 크게
                    buttonPadding = '30px 50px';
                    buttonFontSize = '32px';
                    buttonMinHeight = '70px';
                } else {
                    // 기타 1024 너비
                    const isSmall = screenHeight < 1000;
                    buttonPadding = isSmall ? '10px 20px' : '20px 30px';
                    buttonFontSize = isSmall ? '18px' : '24px';
                    buttonMinHeight = isSmall ? '42px' : '50px';
                }
            } else {
                // 다른 해상도 - 화면 크기에 비례
                const scale = Math.min(screenWidth / 1024, screenHeight / 768);
                buttonPadding = Math.max(10, Math.min(20, 18 * scale)) + 'px ' + Math.max(20, Math.min(30, 25 * scale)) + 'px';
                buttonFontSize = Math.max(18, Math.min(24, 22 * scale)) + 'px';
                buttonMinHeight = Math.max(42, Math.min(50, 46 * scale)) + 'px';
            }
            return { buttonPadding, buttonFontSize, buttonMinHeight };
        };

        // 버튼 크기 동적 조정 함수 (해상도 변경 시) - 먼저 정의
        const updateButtonSize = () => {
            const { buttonPadding, buttonFontSize, buttonMinHeight } = getButtonSize();
            
            // 기존 스타일에서 padding, font-size, min-height 제거
            const currentStyle = button.style.cssText;
            const cleanedStyle = currentStyle
                .replace(/padding\s*:[^;]+!important\s*;?/gi, '')
                .replace(/font-size\s*:[^;]+!important\s*;?/gi, '')
                .replace(/min-height\s*:[^;]+!important\s*;?/gi, '')
                .replace(/padding\s*:[^;]+;?/gi, '')
                .replace(/font-size\s*:[^;]+;?/gi, '')
                .replace(/min-height\s*:[^;]+;?/gi, '');
            
            // 새로운 스타일 추가
            button.style.cssText = cleanedStyle + 
                ` padding: ${buttonPadding} !important; ` +
                ` font-size: ${buttonFontSize} !important; ` +
                ` min-height: ${buttonMinHeight} !important;`;
            
            // 디버깅 로그 제거 (브라우저 반올림으로 인한 불필요한 경고 방지)
        };

        const variant = element.style?.variant || 'primary';
        // baseStyle에는 해상도별 값을 넣지 않고, 나중에 updateButtonSize에서 설정
        const baseStyle = {
            borderRadius: this.theme.borderRadius || '8px',
            border: 'none',
            cursor: 'pointer',
            fontWeight: element.style?.fontWeight || '500',
            transition: 'all 0.3s ease',
            boxSizing: 'border-box',
        };

        const variantStyles = {
            primary: {
                backgroundColor: this.theme.primaryColor || '#667eea',
                color: '#ffffff',
            },
            secondary: {
                backgroundColor: '#e0e0e0',
                color: '#333333',
            },
            link: {
                backgroundColor: 'transparent',
                color: this.theme.primaryColor || '#667eea',
                textDecoration: 'underline',
                padding: '5px 10px',
            },
        };

        // 먼저 기본 스타일 적용 (variant 스타일 제외)
        this.applyStyles(button, {
            ...baseStyle,
            ...variantStyles[variant],
        });

        // element.style가 있으면 적용 (하지만 padding, fontSize, minHeight는 제외)
        if (element.style) {
            const { padding, fontSize, minHeight, ...otherStyles } = element.style;
            this.applyStyles(button, otherStyles);
        }

        if (element.style?.fullWidth) {
            button.style.width = '100%';
            button.style.minWidth = '200px'; // 최소 너비 보장
            button.style.display = 'block'; // 명시적으로 블록 요소로 설정
            button.style.boxSizing = 'border-box';
        }

        if (element.style?.marginTop) {
            button.style.marginTop = element.style.marginTop;
        }
        
        // 초기 크기 설정 (모든 스타일 적용 후 강제로 덮어쓰기)
        updateButtonSize();
        
        // DOM에 추가
        parent.appendChild(button);
        
        // DOM에 추가된 후 여러 시점에서 크기 조정 (렌더링 완료 후)
        requestAnimationFrame(() => {
            updateButtonSize();
        });
        
        setTimeout(() => {
            updateButtonSize();
        }, 50);
        
        setTimeout(() => {
            updateButtonSize();
        }, 200);
        
        // 브라우저 크기 변경 시 버튼 크기 조정 (디바운스 적용)
        const handleButtonResize = debounce(updateButtonSize, RESOLUTION_CONSTANTS.TIMING.DEBOUNCE_DELAY);
        
        window.addEventListener('resize', handleButtonResize);

        // 액션 처리
        if (element.action) {
            button.addEventListener('click', () => {
                this.handleAction(element.action);
            });
        }
    }

    /**
     * 텍스트 렌더링
     */
    renderText(element, parent) {
        const text = document.createElement(element.style?.tag || 'p');
        text.textContent = element.content || '';
        
        // 해상도 타입 확인
        const { isKIOSK } = SDUIRenderer.getResolutionType();
        
        // 스타일 적용
        let finalStyle = { ...(element.style || {}) };
        
        // 키오스크 해상도이고 fontSize가 설정되어 있을 경우 조금 키우기
        if (isKIOSK && finalStyle.fontSize) {
            // fontSize를 숫자로 변환하여 4px 정도 증가
            const currentSize = parseInt(finalStyle.fontSize);
            if (!isNaN(currentSize)) {
                finalStyle.fontSize = (currentSize + 4) + 'px';
            }
        }
        
        this.applyStyles(text, finalStyle);
        parent.appendChild(text);
    }

    /**
     * 테이블 렌더링
     */
    renderTable(element, parent) {
        const table = document.createElement('table');
        this.applyStyles(table, element.style || {});
        table.style.borderCollapse = 'collapse';
        table.style.width = '100%';
        
        // id 속성 추가
        if (element.id) {
            table.id = element.id;
        }

        // 헤더
        if (element.headers) {
            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            element.headers.forEach((header, headerIndex) => {
                const th = document.createElement('th');
                th.textContent = header;
                th.style.padding = '12px';
                th.style.borderBottom = '2px solid #ddd';
                // 가격 컬럼(인덱스 3)은 우측 정렬
                if (headerIndex === 3) {
                    th.style.textAlign = 'right';
                }
                // POS사용(인덱스 4), 스캔 이미지 개수(인덱스 5), 대표 이미지(인덱스 6) 컬럼은 중앙 정렬
                else if (headerIndex === 4 || headerIndex === 5 || headerIndex === 6) {
                    th.style.textAlign = 'center';
                } else {
                    th.style.textAlign = 'left';
                }
                th.style.backgroundColor = '#f8f9fa';
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);
            table.appendChild(thead);
        }

        // 바디
        if (element.rows) {
            const tbody = document.createElement('tbody');
            element.rows.forEach((row, index) => {
                const tr = document.createElement('tr');
                tr.style.backgroundColor = index % 2 === 0 ? '#ffffff' : '#f8f9fa';
                
                row.cells.forEach(cell => {
                    const td = document.createElement('td');
                    td.style.padding = '12px';
                    td.style.borderBottom = '1px solid #eee';
                    
                    // 셀 내용이 객체인 경우 (버튼 등) 렌더링
                    if (typeof cell === 'object' && cell !== null) {
                        this.renderElement(cell, td);
                    } else {
                        td.textContent = cell || '';
                    }
                    
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
            table.appendChild(tbody);
        }

        parent.appendChild(table);
    }

    /**
     * 네비게이션 바 렌더링
     */
    renderNavigationBar(element, parent) {
        const navBar = document.createElement('div');
        navBar.className = 'sdui-navigation-bar';
        
        // 해상도별 네비게이션 바 크기 계산
        const { isKIOSK, isPOS, screenWidth, screenHeight } = SDUIRenderer.getResolutionType();
        
        let navPadding, navGap, buttonSize, buttonFontSize, separatorHeight;
        if (isPOS) {
            // POS 해상도 - 작게
            navPadding = '8px 12px';
            navGap = '8px';
            buttonSize = '50px';
            buttonFontSize = '20px';
            separatorHeight = '35px';
        } else if (isKIOSK) {
            // KIOSK 해상도 - 크게
            navPadding = '15px 20px';
            navGap = '15px';
            buttonSize = '70px';
            buttonFontSize = '28px';
            separatorHeight = '50px';
        } else {
            // 기타 해상도
            const scale = Math.min(screenWidth / 1024, screenHeight / 768);
            const isSmall = screenHeight < 1000;
            navPadding = isSmall ? '8px 12px' : '15px 20px';
            navGap = isSmall ? '8px' : '15px';
            buttonSize = isSmall ? '50px' : '70px';
            buttonFontSize = isSmall ? '20px' : '28px';
            separatorHeight = isSmall ? '35px' : '50px';
        }
        
        const defaultStyle = {
            position: 'fixed',
            bottom: '0',
            left: '0',
            right: '0',
            backgroundColor: '#f8f9fa',
            borderTop: '2px solid #dee2e6',
            padding: navPadding,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            gap: navGap,
            zIndex: '1000',
            boxShadow: '0 -2px 10px rgba(0,0,0,0.1)',
        };
        
        this.applyStyles(navBar, { ...defaultStyle, ...(element.style || {}) });

        // 네비게이션 버튼들
        const buttons = element.buttons || [];
        
        buttons.forEach((button, index) => {
            // 구분선
            if (button.type === 'separator') {
                const separator = document.createElement('div');
                separator.style.width = '2px';
                separator.style.height = separatorHeight;
                separator.style.backgroundColor = '#dee2e6';
                separator.style.margin = '0 5px';
                navBar.appendChild(separator);
                return;
            }

            const btn = document.createElement('button');
            btn.className = 'sdui-nav-button';
            
            const btnStyle = {
                width: buttonSize,
                height: buttonSize,
                backgroundColor: button.active ? '#667eea' : '#ffffff',
                color: button.active ? '#ffffff' : '#333333',
                border: '2px solid #dee2e6',
                borderRadius: '12px',
                fontSize: buttonFontSize,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s ease',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                userSelect: 'none',
                WebkitUserSelect: 'none',
                msUserSelect: 'none',
            };
            
            this.applyStyles(btn, { ...btnStyle, ...(button.style || {}) });

            // 아이콘 또는 텍스트
            if (button.icon) {
                btn.textContent = button.icon;
            } else if (button.label) {
                btn.textContent = button.label;
                // 레이블이 있는 경우 폰트 크기 조정
                const { isPOS } = SDUIRenderer.getResolutionType();
                const labelFontSize = isPOS ? '12px' : '14px';
                btn.style.fontSize = labelFontSize;
            }

            // 툴팁 (title 속성)
            if (button.style && button.style.title) {
                btn.title = button.style.title;
            }

            // 호버 효과
            btn.addEventListener('mouseenter', function() {
                if (!button.active) {
                    this.style.backgroundColor = '#e9ecef';
                    this.style.transform = 'scale(1.05)';
                }
            });
            btn.addEventListener('mouseleave', function() {
                if (!button.active) {
                    this.style.backgroundColor = '#ffffff';
                    this.style.transform = 'scale(1)';
                }
            });

            // 터치 이벤트 (키오스크/POS 환경)
            btn.addEventListener('touchstart', function() {
                this.style.transform = 'scale(0.95)';
            });
            btn.addEventListener('touchend', function() {
                this.style.transform = 'scale(1)';
            });

            // 액션 처리
            if (button.action) {
                btn.addEventListener('click', () => {
                    this.handleAction(button.action);
                });
            }

            navBar.appendChild(btn);
        });

        parent.appendChild(navBar);
        
        // 네비게이션 바 높이 계산 및 컨테이너 높이 조정
        const navBarHeight = navBar.offsetHeight;
        const isPasswordPage = parent.querySelector('.sdui-password-input') !== null;
        
        if (isPasswordPage) {
            // 비밀번호 페이지의 경우 컨테이너 높이를 네비게이션 바 높이에 맞춰 조정
            const passwordContainer = parent.querySelector('.sdui-password-container') ||
                                    parent.querySelector('.sdui-container') || 
                                    parent.querySelector('[style*="maxHeight"]') ||
                                    document.querySelector('[style*="maxHeight"]');
            
            debugLog('[renderNavigationBar] 비밀번호 페이지 감지:', {
                foundContainer: !!passwordContainer,
                containerClass: passwordContainer ? passwordContainer.className : 'none',
                containerId: passwordContainer ? passwordContainer.id : 'none'
            });
            
            if (passwordContainer) {
                // 초기 렌더링 시 레이아웃 시프트 방지를 위해 컨테이너를 숨김
                const originalVisibility = passwordContainer.style.visibility;
                const originalOpacity = passwordContainer.style.opacity;
                passwordContainer.style.visibility = 'hidden';
                passwordContainer.style.opacity = '0';
                
                // display와 flexDirection은 먼저 설정
                const currentDisplay = window.getComputedStyle(passwordContainer).display;
                if (currentDisplay === 'none') {
                    passwordContainer.style.display = 'flex';
                    passwordContainer.style.flexDirection = 'column';
                }
                
                const { isKIOSK, isPOS } = SDUIRenderer.getResolutionType();
                
                debugLog('[renderNavigationBar] 해상도 정보:', {
                    screenWidth: window.innerWidth,
                    screenHeight: window.innerHeight,
                    isKIOSK,
                    isPOS,
                    navBarHeight
                });
                
                // 헤더 높이 계산 헬퍼
                const getHeaderHeight = () => {
                    const header = parent.querySelector('header');
                    if (header) {
                        const height = header.offsetHeight || header.getBoundingClientRect().height;
                        return height > 0 ? height : 0;
                    }
                    return 0;
                };
                
                // 네비게이션 바 높이 계산 헬퍼
                const getNavBarHeight = () => {
                    if (navBar && navBar.offsetHeight > 0) {
                        return navBar.offsetHeight;
                    }
                    // 예상 높이 계산 (해상도별)
                    if (isPOS) {
                        return 68; // POS 해상도 예상 네비게이션 바 높이
                    } else if (isKIOSK) {
                        return 102; // KIOSK 해상도 예상 네비게이션 바 높이
                    } else {
                        return navBarHeight > 0 ? navBarHeight : 85; // 기본값
                    }
                };
                
                // 해상도별 스타일 조정 (초기 렌더링 최적화)
                const applyPasswordContainerStyles = (isInitial = false) => {
                    const { isKIOSK: currentIsKIOSK, isPOS: currentIsPOS } = SDUIRenderer.getResolutionType();
                    
                    debugLog('[applyPasswordContainerStyles] 호출:', {
                        isInitial,
                        isKIOSK: currentIsKIOSK,
                        isPOS: currentIsPOS
                    });
                    
                    // display와 flexDirection은 항상 설정
                    passwordContainer.style.display = 'flex';
                    passwordContainer.style.flexDirection = 'column';
                    
                    // 헤더 높이 계산
                    let headerHeight = 0;
                    const currentHeader = parent.querySelector('header');
                    if (currentHeader && currentHeader.offsetHeight > 0) {
                        headerHeight = currentHeader.offsetHeight;
                    } else {
                        // 예상 높이 계산 (해상도별)
                        if (currentIsPOS) {
                            headerHeight = 80; // POS 해상도 예상 헤더 높이
                        } else if (currentIsKIOSK) {
                            headerHeight = 120; // KIOSK 해상도 예상 헤더 높이
                        } else {
                            headerHeight = 100; // 기본 예상 높이
                        }
                    }
                    
                    // 네비게이션 바 높이 계산
                    let currentNavBarHeight = getNavBarHeight();
                    if (navBar && navBar.offsetHeight > 0) {
                        currentNavBarHeight = navBar.offsetHeight;
                    }
                    
                    // 네비게이션 바 높이 + 여유 공간
                    const containerMaxHeight = `calc(100vh - ${currentNavBarHeight + 5}px)`;
                    passwordContainer.style.maxHeight = containerMaxHeight;
                    
                    if (currentIsKIOSK) {
                        // KIOSK 해상도: 중앙 정렬, 큰 패딩과 간격
                        const availableHeight = window.innerHeight - headerHeight - currentNavBarHeight;
                        const minHeightValue = Math.max(availableHeight * 0.95, 800); // 사용 가능한 높이의 95% 또는 최소 800px
                        
                        debugLog('[applyPasswordContainerStyles] KIOSK 스타일 적용:', {
                            headerHeight,
                            currentNavBarHeight,
                            windowInnerHeight: window.innerHeight,
                            availableHeight,
                            minHeightValue
                        });
                        
                        passwordContainer.style.padding = '20px';
                        passwordContainer.style.maxWidth = '900px';
                        passwordContainer.style.justifyContent = 'center'; // 중앙 정렬
                        passwordContainer.style.gap = '30px';
                        passwordContainer.style.minHeight = minHeightValue + 'px'; // 최소 높이 설정으로 중앙 정렬 보장
                        
                        // 초기 렌더링이면 정확한 높이로 재계산 후 표시
                        if (isInitial) {
                            requestAnimationFrame(() => {
                                requestAnimationFrame(() => {
                                    // 정확한 높이 재계산
                                    const finalHeader = parent.querySelector('header');
                                    const finalHeaderHeight = finalHeader ? (finalHeader.offsetHeight || finalHeader.getBoundingClientRect().height) : headerHeight;
                                    const finalNavBarHeight = navBar.offsetHeight || navBar.getBoundingClientRect().height || currentNavBarHeight;
                                    const finalAvailableHeight = window.innerHeight - finalHeaderHeight - finalNavBarHeight;
                                    const finalMinHeightValue = Math.max(finalAvailableHeight * 0.95, 800);
                                    
                                    // 높이만 업데이트
                                    passwordContainer.style.minHeight = finalMinHeightValue + 'px';
                                    passwordContainer.style.maxHeight = `calc(100vh - ${finalNavBarHeight + 5}px)`;
                                    
                                    // 컨테이너 표시
                                    passwordContainer.style.visibility = originalVisibility || 'visible';
                                    passwordContainer.style.opacity = originalOpacity || '1';
                                    
                                    debugLog('[applyPasswordContainerStyles] KIOSK 초기 렌더링 완료:', {
                                        finalHeaderHeight,
                                        finalNavBarHeight,
                                        finalAvailableHeight,
                                        finalMinHeightValue
                                    });
                                });
                            });
                        }
                    } else if (currentIsPOS) {
                        // POS 해상도: 상단 정렬, 작은 패딩과 간격
                        debugLog('[applyPasswordContainerStyles] POS 스타일 적용');
                        passwordContainer.style.padding = '5px';
                        passwordContainer.style.maxWidth = '800px';
                        passwordContainer.style.justifyContent = 'flex-start';
                        passwordContainer.style.gap = '5px';
                        passwordContainer.style.minHeight = 'auto'; // POS는 minHeight 제거
                        
                        // 초기 렌더링이면 정확한 높이로 재계산 후 표시
                        if (isInitial) {
                            requestAnimationFrame(() => {
                                requestAnimationFrame(() => {
                                    // 정확한 높이 재계산
                                    const finalNavBarHeight = navBar.offsetHeight || navBar.getBoundingClientRect().height || currentNavBarHeight;
                                    passwordContainer.style.maxHeight = `calc(100vh - ${finalNavBarHeight + 5}px)`;
                                    
                                    // 컨테이너 표시
                                    passwordContainer.style.visibility = originalVisibility || 'visible';
                                    passwordContainer.style.opacity = originalOpacity || '1';
                                    
                                    debugLog('[applyPasswordContainerStyles] POS 초기 렌더링 완료:', {
                                        finalNavBarHeight
                                    });
                                });
                            });
                        }
                    } else {
                        // 기본 스타일 (KIOSK도 POS도 아닌 경우)
                        debugLog('[applyPasswordContainerStyles] 기본 스타일 적용');
                        passwordContainer.style.padding = '10px';
                        passwordContainer.style.maxWidth = '900px';
                        passwordContainer.style.justifyContent = 'flex-start';
                        passwordContainer.style.gap = '10px';
                        passwordContainer.style.minHeight = 'auto';
                        
                        // 초기 렌더링이면 정확한 높이로 재계산 후 표시
                        if (isInitial) {
                            requestAnimationFrame(() => {
                                requestAnimationFrame(() => {
                                    // 정확한 높이 재계산
                                    const finalNavBarHeight = navBar.offsetHeight || navBar.getBoundingClientRect().height || currentNavBarHeight;
                                    passwordContainer.style.maxHeight = `calc(100vh - ${finalNavBarHeight + 5}px)`;
                                    
                                    // 컨테이너 표시
                                    passwordContainer.style.visibility = originalVisibility || 'visible';
                                    passwordContainer.style.opacity = originalOpacity || '1';
                                    
                                    debugLog('[applyPasswordContainerStyles] 기본 스타일 초기 렌더링 완료');
                                });
                            });
                        }
                    }
                };
                
                // 초기 스타일 적용 (예상 높이 사용)
                applyPasswordContainerStyles(true);
                
                // 리사이즈 이벤트 리스너 추가 (초기 렌더링이 아닐 때만)
                const handlePasswordPageResize = debounce(() => {
                    if (passwordContainer && parent.querySelector('.sdui-password-input')) {
                        debugLog('[renderNavigationBar] 비밀번호 페이지 리사이즈 이벤트 발생');
                        applyPasswordContainerStyles(false);
                    }
                }, RESOLUTION_CONSTANTS.TIMING.DEBOUNCE_DELAY);
                
                window.addEventListener('resize', handlePasswordPageResize);
            } else {
                console.warn('[renderNavigationBar] ⚠️ 비밀번호 컨테이너를 찾을 수 없습니다!');
            }
        }
        
        // 메인 페이지의 경우 컨테이너를 중앙 정렬
        const isMainPage = window.location.pathname === '/' || 
                          window.location.pathname === '/web' || 
                          window.location.search.includes('page=main') ||
                          parent.querySelector('.sdui-icon-card') !== null; // 아이콘 카드가 있으면 메인 페이지
        
        if (isMainPage && !isPasswordPage) {
            debugLog('[renderNavigationBar] 메인 페이지 감지, 컨테이너 중앙 정렬 적용');
            
            // 메인 컨테이너 찾기 (헤더 다음의 첫 번째 컨테이너)
            const header = parent.querySelector('header');
            let mainContainer = null;
            
            // 헤더 다음의 컨테이너 찾기
            if (header && header.nextElementSibling) {
                mainContainer = header.nextElementSibling;
            } else {
                // 헤더가 없거나 다음 요소가 없으면 첫 번째 컨테이너 찾기
                const containers = parent.querySelectorAll('.sdui-container, div[style*="maxWidth"]');
                if (containers.length > 0) {
                    mainContainer = containers[0];
                } else {
                    // 마지막으로 첫 번째 자식 요소 사용
                    mainContainer = parent.firstElementChild;
                }
            }
            
            // 네비게이션 바는 제외
            if (mainContainer === navBar) {
                mainContainer = null;
            }
            
            if (mainContainer) {
                // 초기 렌더링 시 레이아웃 시프트 방지를 위해 컨테이너를 숨김
                const originalVisibility = mainContainer.style.visibility;
                const originalOpacity = mainContainer.style.opacity;
                mainContainer.style.visibility = 'hidden';
                mainContainer.style.opacity = '0';
                
                const applyMainPageStyles = (isInitial = false) => {
                    const { isKIOSK, isPOS } = SDUIRenderer.getResolutionType();
                    
                    // 헤더 높이 계산 (더 정확한 방법 사용)
                    const currentHeader = parent.querySelector('header');
                    let headerHeight = 0;
                    if (currentHeader) {
                        // 렌더링이 완료된 경우 offsetHeight 사용, 아니면 예상 높이 사용
                        if (currentHeader.offsetHeight > 0) {
                            headerHeight = currentHeader.offsetHeight;
                        } else {
                            // 예상 높이 계산 (해상도별)
                            if (isPOS) {
                                headerHeight = 80; // POS 해상도 예상 헤더 높이
                            } else if (isKIOSK) {
                                headerHeight = 120; // KIOSK 해상도 예상 헤더 높이
                            } else {
                                headerHeight = 100; // 기본 예상 높이
                            }
                        }
                    }
                    
                    // 네비게이션 바 높이 계산
                    let currentNavBarHeight = 0;
                    if (navBar.offsetHeight > 0) {
                        currentNavBarHeight = navBar.offsetHeight;
                    } else {
                        // 예상 높이 계산 (해상도별)
                        if (isPOS) {
                            currentNavBarHeight = 68; // POS 해상도 예상 네비게이션 바 높이
                        } else if (isKIOSK) {
                            currentNavBarHeight = 102; // KIOSK 해상도 예상 네비게이션 바 높이
                        } else {
                            currentNavBarHeight = 85; // 기본 예상 높이
                        }
                    }
                    
                    // 사용 가능한 높이 계산
                    const availableHeight = window.innerHeight - headerHeight - currentNavBarHeight;
                    
                    debugLog('[applyMainPageStyles] 메인 페이지 스타일 적용:', {
                        isInitial,
                        isKIOSK,
                        isPOS,
                        headerHeight,
                        currentNavBarHeight,
                        windowInnerHeight: window.innerHeight,
                        availableHeight,
                        containerElement: mainContainer
                    });
                    
                    // 컨테이너를 flex로 설정하고 중앙 정렬
                    mainContainer.style.display = 'flex';
                    mainContainer.style.flexDirection = 'column';
                    mainContainer.style.justifyContent = 'center'; // 중앙 정렬
                    mainContainer.style.minHeight = availableHeight + 'px';
                    mainContainer.style.maxHeight = availableHeight + 'px';
                    mainContainer.style.overflowY = 'auto';
                    mainContainer.style.overflowX = 'hidden';
                    
                    // 해상도별 패딩 조정
                    if (isPOS) {
                        mainContainer.style.padding = '20px';
                    } else if (isKIOSK) {
                        mainContainer.style.padding = '40px 20px';
                    } else {
                        mainContainer.style.padding = '30px 20px';
                    }
                    
                    // 초기 렌더링이면 한 번만 정확한 높이로 재계산 후 표시
                    if (isInitial) {
                        // requestAnimationFrame을 사용하여 브라우저가 레이아웃을 계산한 후 정확한 높이 적용
                        requestAnimationFrame(() => {
                            requestAnimationFrame(() => {
                                // 정확한 높이 재계산
                                const finalHeader = parent.querySelector('header');
                                const finalHeaderHeight = finalHeader ? (finalHeader.offsetHeight || finalHeader.getBoundingClientRect().height) : headerHeight;
                                const finalNavBarHeight = navBar.offsetHeight || navBar.getBoundingClientRect().height || currentNavBarHeight;
                                const finalAvailableHeight = window.innerHeight - finalHeaderHeight - finalNavBarHeight;
                                
                                // 높이만 업데이트 (다른 스타일은 유지)
                                mainContainer.style.minHeight = finalAvailableHeight + 'px';
                                mainContainer.style.maxHeight = finalAvailableHeight + 'px';
                                
                                // 이제 컨테이너를 표시 (부드러운 전환)
                                mainContainer.style.visibility = originalVisibility || 'visible';
                                mainContainer.style.opacity = originalOpacity || '1';
                                
                                debugLog('[applyMainPageStyles] 초기 렌더링 완료, 컨테이너 표시:', {
                                    finalHeaderHeight,
                                    finalNavBarHeight,
                                    finalAvailableHeight
                                });
                            });
                        });
                    }
                };
                
                // 초기 스타일 적용 (예상 높이 사용)
                applyMainPageStyles(true);
                
                // 리사이즈 이벤트 리스너 추가 (초기 렌더링이 아닐 때만)
                const handleMainPageResize = debounce(() => {
                    if (mainContainer && parent.querySelector('.sdui-icon-card')) {
                        debugLog('[renderNavigationBar] 메인 페이지 리사이즈 이벤트 발생');
                        applyMainPageStyles(false);
                    }
                }, RESOLUTION_CONSTANTS.TIMING.DEBOUNCE_DELAY);
                
                window.addEventListener('resize', handleMainPageResize);
            } else {
                console.warn('[renderNavigationBar] ⚠️ 메인 컨테이너를 찾을 수 없습니다!');
            }
        }
        
        // 다른 페이지들(상품 관리, 환경설정 등)의 경우에도 레이아웃 시프트 방지
        if (!isPasswordPage && !isMainPage) {
            // 헤더 찾기
            const header = parent.querySelector('header');
            
            const mainContent = parent.querySelector('.sdui-container') || 
                               parent.querySelector('div[style*="maxWidth"]') ||
                               (header && header.nextElementSibling) ||
                               parent.firstElementChild;
            
            if (mainContent && mainContent !== navBar && mainContent !== header) {
                // 초기 렌더링 시 레이아웃 시프트 방지를 위해 컨테이너를 숨김
                const originalVisibility = mainContent.style.visibility;
                const originalOpacity = mainContent.style.opacity;
                mainContent.style.visibility = 'hidden';
                mainContent.style.opacity = '0';
                
                // 예상 네비게이션 바 높이 계산
                const { isKIOSK, isPOS } = SDUIRenderer.getResolutionType();
                let expectedNavBarHeight = navBarHeight;
                if (expectedNavBarHeight === 0) {
                    if (isPOS) {
                        expectedNavBarHeight = 68;
                    } else if (isKIOSK) {
                        expectedNavBarHeight = 102;
                    } else {
                        expectedNavBarHeight = 85;
                    }
                }
                
                // 초기 패딩 적용 (예상 높이 사용)
                if (mainContent !== parent) {
                    mainContent.style.paddingBottom = (expectedNavBarHeight + 10) + 'px';
                } else {
                    document.body.style.paddingBottom = (expectedNavBarHeight + 10) + 'px';
                }
                
                // 정확한 높이로 재계산 후 표시
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        const finalNavBarHeight = navBar.offsetHeight || navBar.getBoundingClientRect().height || expectedNavBarHeight;
                        
                        if (mainContent !== parent) {
                            mainContent.style.paddingBottom = (finalNavBarHeight + 10) + 'px';
                        } else {
                            document.body.style.paddingBottom = (finalNavBarHeight + 10) + 'px';
                        }
                        
                        // 컨테이너 표시
                        mainContent.style.visibility = originalVisibility || 'visible';
                        mainContent.style.opacity = originalOpacity || '1';
                        
                        debugLog('[renderNavigationBar] 다른 페이지 초기 렌더링 완료:', {
                            finalNavBarHeight,
                            page: window.location.pathname
                        });
                    });
                });
            }
        }
    }

    /**
     * 비밀번호 입력 필드 렌더링
     */
    renderPasswordInput(element, parent) {
        const container = document.createElement('div');
        container.className = 'sdui-password-input';
        
        // 해상도별 최적화를 위한 크기 계산
        const { isKIOSK, isPOS, screenWidth, screenHeight } = SDUIRenderer.getResolutionType();
        
        // 1024x768 (POS) 또는 1024x1920 (KIOSK) 최적화
        let containerMargin, containerPadding, inputPadding, inputFontSize, inputLetterSpacing;
        
        if (isPOS) {
            // POS 해상도 - 더 작게
            containerMargin = '0 auto 3px';
            containerPadding = '3px';
            inputPadding = '8px';
            inputFontSize = '14px';
            inputLetterSpacing = '4px';
        } else if (isKIOSK) {
            // KIOSK 해상도 - 더 크게
            containerMargin = '0 auto 30px';
            containerPadding = '20px';
            inputPadding = '30px';
            inputFontSize = '32px';
            inputLetterSpacing = '10px';
        } else {
            // 기타 해상도 - 화면 크기에 비례
            const isSmall = screenHeight < 1000;
            containerMargin = isSmall ? '0 auto 3px' : '0 auto 20px';
            containerPadding = isSmall ? '3px' : '15px';
            inputPadding = isSmall ? '8px' : '20px';
            inputFontSize = isSmall ? '14px' : '24px';
            inputLetterSpacing = isSmall ? '4px' : '8px';
        }
        
        const defaultStyle = {
            width: '100%',
            maxWidth: '100%', // 입력완료 버튼과 같은 너비
            margin: containerMargin,
            padding: containerPadding,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            minWidth: '200px', // 최소 너비 보장
        };
        
        this.applyStyles(container, { ...defaultStyle, ...(element.style || {}) });

        const inputWrapper = document.createElement('div');
        inputWrapper.style.position = 'relative';
        inputWrapper.style.width = '100%';
        inputWrapper.style.minWidth = '200px';
        
        const input = document.createElement('input');
        input.type = 'password';
        input.id = element.id || 'password-input';
        input.placeholder = element.placeholder || '비밀번호를 입력하세요';
        input.readOnly = true; // 키보드 입력 방지 (키오스크 환경)
        
        // 최소 높이 계산 (해상도별) - 더 크게 설정
        let minHeight;
        if (isPOS) {
            minHeight = '40px'; // POS 해상도 - 더 크게
        } else if (isKIOSK) {
            minHeight = '55px'; // KIOSK 해상도
        } else {
            const scale = Math.min(screenWidth / 1024, screenHeight / 768);
            minHeight = Math.max(40, Math.min(55, 47 * scale)) + 'px';
        }
        
        const inputStyle = {
            width: '100%',
            minHeight: minHeight,
            height: 'auto',
            padding: inputPadding,
            fontSize: inputFontSize,
            fontWeight: '700',
            textAlign: 'center',
            border: '2px solid #dee2e6',
            borderRadius: '12px',
            backgroundColor: '#ffffff',
            outline: 'none',
            letterSpacing: inputLetterSpacing,
            fontFamily: 'monospace',
            boxSizing: 'border-box',
            display: 'block', // 명시적으로 블록 요소로 설정
        };
        
        this.applyStyles(input, { ...inputStyle, ...(element.inputStyle || {}) });
        
        // 비밀번호 값 저장을 위한 데이터 속성
        input.dataset.passwordValue = '';
        
        inputWrapper.appendChild(input);
        container.appendChild(inputWrapper);
        
        // 에러 메시지 영역
        if (element.showError) {
            const errorMsg = document.createElement('div');
            errorMsg.id = element.id ? `${element.id}-error` : 'password-error';
            errorMsg.style.display = 'none';
            errorMsg.style.color = '#dc3545';
            errorMsg.style.fontSize = '14px';
            errorMsg.style.marginTop = '10px';
            errorMsg.style.textAlign = 'center';
            container.appendChild(errorMsg);
        }
        
        // 비밀번호 입력 필드 크기 조정 함수 (해상도별 최적화)
        const updateInputSize = () => {
            const { isKIOSK, isPOS, screenWidth, screenHeight } = SDUIRenderer.getResolutionType();
            
            let inputPadding, inputFontSize, inputLetterSpacing, containerMargin, containerPadding;
            
            if (isPOS) {
                // POS 해상도 - 더 작게
                containerMargin = '0 auto 3px';
                containerPadding = '3px';
                inputPadding = '8px';
                inputFontSize = '14px';
                inputLetterSpacing = '4px';
            } else if (isKIOSK) {
                // KIOSK 해상도 - 더 크게
                containerMargin = '0 auto 30px';
                containerPadding = '20px';
                inputPadding = '30px';
                inputFontSize = '32px';
                inputLetterSpacing = '10px';
            } else {
                // 기타 해상도 - 화면 크기에 비례
                const isSmall = screenHeight < 1000;
                containerMargin = isSmall ? '0 auto 3px' : '0 auto 30px';
                containerPadding = isSmall ? '3px' : '20px';
                inputPadding = isSmall ? '8px' : '30px';
                inputFontSize = isSmall ? '14px' : '32px';
                inputLetterSpacing = isSmall ? '4px' : '10px';
            }
            
            // 최소 높이 계산 - 더 크게 설정
            let inputMinHeight;
            if (isPOS) {
                inputMinHeight = '40px';
            } else if (isKIOSK) {
                inputMinHeight = '70px';
            } else {
                const scale = Math.min(screenWidth / 1024, screenHeight / 768);
                inputMinHeight = Math.max(40, Math.min(70, 55 * scale)) + 'px';
            }
            
            // 컨테이너는 부모의 너비를 따라가도록 (입력완료 버튼과 동일)
            container.style.maxWidth = '100%';
            container.style.margin = containerMargin;
            container.style.padding = containerPadding;
            input.style.padding = inputPadding;
            input.style.fontSize = inputFontSize;
            input.style.letterSpacing = inputLetterSpacing;
            input.style.fontWeight = '700';
            input.style.minHeight = inputMinHeight;
        };
        
        // 초기 크기 설정
        updateInputSize();
        
        // 브라우저 크기 변경 시 동적으로 조정 (디바운스 적용)
        const handleResize = debounce(updateInputSize, RESOLUTION_CONSTANTS.TIMING.DEBOUNCE_DELAY);
        window.addEventListener('resize', handleResize);
        
        parent.appendChild(container);
    }

    /**
     * 숫자 키패드 렌더링
     */
    renderNumpad(element, parent) {
        const container = document.createElement('div');
        container.className = 'sdui-numpad';
        
        // 키패드 크기 계산 함수 (해상도별 최적화)
        const updateNumpadSize = () => {
            const screenWidth = window.innerWidth;
            const screenHeight = window.innerHeight;
            
            let numpadWidth, gap, padding, buttonFontSize;
            if (screenWidth === 1024) {
                if (screenHeight === 768) {
                    // POS 해상도 (1024x768) - 더 크게 조정
                    numpadWidth = 360;
                    gap = 8;
                    padding = 10;
                    buttonFontSize = 26;
                } else if (screenHeight === 1920) {
                    // KIOSK 해상도 (1024x1920) - 더 크게
                    numpadWidth = 650;
                    gap = 25;
                    padding = 25;
                    buttonFontSize = 50;
                } else {
                    // 기타 1024 너비
                    const isSmall = screenHeight < 1000;
                    numpadWidth = isSmall ? 360 : 650;
                    gap = isSmall ? 8 : 25;
                    padding = isSmall ? 10 : 25;
                    buttonFontSize = isSmall ? 26 : 50;
                }
            } else {
                // 다른 해상도 - 화면 크기에 비례
                const scale = Math.min(screenWidth / 1024, screenHeight / 768);
                numpadWidth = Math.max(360, Math.min(650, 505 * scale));
                gap = Math.max(8, Math.min(25, 16 * scale));
                padding = Math.max(10, Math.min(25, 17 * scale));
                buttonFontSize = Math.max(26, Math.min(50, 38 * scale));
            }
            
            // 버튼 크기 계산: (키패드 너비 - 좌우 패딩 - gap * 2) / 3
            const buttonSize = (numpadWidth - padding * 2 - gap * 2) / 3;
            
            // 컨테이너 스타일 업데이트
            container.style.width = numpadWidth + 'px';
            container.style.maxWidth = numpadWidth + 'px';
            container.style.gap = gap + 'px';
            container.style.padding = padding + 'px';
            
            // 버튼 크기 업데이트
            const buttons = container.querySelectorAll('.sdui-numpad-button');
            
            buttons.forEach(button => {
                button.style.minHeight = buttonSize + 'px';
                button.style.fontSize = buttonFontSize + 'px';
            });
        };
        
        // 초기 크기 설정 (해상도별)
        const initialScreenWidth = window.innerWidth;
        const initialScreenHeight = window.innerHeight;
        
        let initialGap, initialPadding, initialWidth;
        if (initialScreenWidth === 1024) {
            if (initialScreenHeight === 768) {
                initialGap = 8;
                initialPadding = 10;
                initialWidth = 360;
            } else if (initialScreenHeight === 1920) {
                initialGap = 20;
                initialPadding = 20;
                initialWidth = 500;
            } else {
                const isSmall = initialScreenHeight < 1000;
                initialGap = isSmall ? 8 : 25;
                initialPadding = isSmall ? 10 : 25;
                initialWidth = isSmall ? 360 : 650;
            }
        } else {
            const scale = Math.min(initialScreenWidth / 1024, initialScreenHeight / 768);
            initialGap = Math.max(8, Math.min(25, 16 * scale));
            initialPadding = Math.max(10, Math.min(25, 17 * scale));
            initialWidth = Math.max(360, Math.min(650, 505 * scale));
        }
        
        const defaultStyle = {
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: initialGap + 'px',
            margin: '0 auto',
            padding: initialPadding + 'px',
            width: initialWidth + 'px',
            transition: 'all 0.3s ease', // 크기 변경 시 부드러운 전환
            alignSelf: 'center', // 세로 중앙 정렬을 위한 추가
        };
        
        this.applyStyles(container, { ...defaultStyle, ...(element.style || {}) });
        
        // 초기 크기 계산
        updateNumpadSize();
        
        // 브라우저 크기 변경 시 동적으로 조정 (디바운스 적용)
        const handleResize = debounce(updateNumpadSize, RESOLUTION_CONSTANTS.TIMING.DEBOUNCE_DELAY);
        window.addEventListener('resize', handleResize);
        
        // 컴포넌트가 제거될 때 이벤트 리스너 정리 (선택사항)
        // container.dataset.resizeHandler = 'true';

        const buttons = [
            { value: '1', label: '1' },
            { value: '2', label: '2' },
            { value: '3', label: '3' },
            { value: '4', label: '4' },
            { value: '5', label: '5' },
            { value: '6', label: '6' },
            { value: '7', label: '7' },
            { value: '8', label: '8' },
            { value: '9', label: '9' },
            { value: 'clear', label: 'X', icon: '✕' },
            { value: '0', label: '0' },
            { value: 'backspace', label: '⌫', icon: '⌫' },
        ];

        // 초기 버튼 크기 계산 (해상도별)
        const btnInitScreenWidth = window.innerWidth;
        const btnInitScreenHeight = window.innerHeight;
        
        let btnInitNumpadWidth, btnInitGap, btnInitPadding, btnInitFontSize;
        if (btnInitScreenWidth === 1024) {
            if (btnInitScreenHeight === 768) {
                btnInitNumpadWidth = 360;
                btnInitGap = 8;
                btnInitPadding = 10;
                btnInitFontSize = 26;
            } else if (btnInitScreenHeight === 1920) {
                btnInitNumpadWidth = 650;
                btnInitGap = 25;
                btnInitPadding = 25;
                btnInitFontSize = 50;
            } else {
                const isSmall = btnInitScreenHeight < 1000;
                btnInitNumpadWidth = isSmall ? 360 : 650;
                btnInitGap = isSmall ? 8 : 25;
                btnInitPadding = isSmall ? 10 : 25;
                btnInitFontSize = isSmall ? 26 : 50;
            }
        } else {
            const scale = Math.min(btnInitScreenWidth / 1024, btnInitScreenHeight / 768);
            btnInitNumpadWidth = Math.max(360, Math.min(650, 505 * scale));
            btnInitGap = Math.max(8, Math.min(25, 16 * scale));
            btnInitPadding = Math.max(10, Math.min(25, 17 * scale));
            btnInitFontSize = Math.max(26, Math.min(50, 38 * scale));
        }
        
        const initialButtonSize = (btnInitNumpadWidth - btnInitPadding * 2 - btnInitGap * 2) / 3;

        buttons.forEach(btn => {
            const button = document.createElement('button');
            button.className = 'sdui-numpad-button';
            button.dataset.value = btn.value;
            
            const btnStyle = {
                width: '100%',
                aspectRatio: '1',
                minHeight: initialButtonSize + 'px',
                fontSize: btnInitFontSize + 'px',
                fontWeight: '600',
                border: '2px solid #dee2e6',
                borderRadius: '12px',
                backgroundColor: '#ffffff',
                color: '#333333',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                userSelect: 'none',
                WebkitUserSelect: 'none',
                msUserSelect: 'none',
            };
            
            // 특수 버튼 스타일
            if (btn.value === 'clear' || btn.value === 'backspace') {
                btnStyle.backgroundColor = '#f8f9fa';
                btnStyle.color = '#dc3545';
            }
            
            this.applyStyles(button, { ...btnStyle, ...(element.buttonStyle || {}) });

            if (btn.icon) {
                button.textContent = btn.icon;
            } else {
                button.textContent = btn.label;
            }

            // 호버 효과
            button.addEventListener('mouseenter', function() {
                this.style.backgroundColor = '#e9ecef';
                this.style.transform = 'scale(1.05)';
            });
            button.addEventListener('mouseleave', function() {
                const isSpecial = btn.value === 'clear' || btn.value === 'backspace';
                this.style.backgroundColor = isSpecial ? '#f8f9fa' : '#ffffff';
                this.style.transform = 'scale(1)';
            });

            // 터치 이벤트
            button.addEventListener('touchstart', function() {
                this.style.transform = 'scale(0.95)';
            });
            button.addEventListener('touchend', function() {
                this.style.transform = 'scale(1)';
            });

            // 클릭 이벤트
            button.addEventListener('click', () => {
                this.handleNumpadInput(btn.value, element);
            });

            container.appendChild(button);
        });

        parent.appendChild(container);
    }

    /**
     * 키패드 입력 처리
     */
    handleNumpadInput(value, numpadElement) {
        const passwordInputId = numpadElement.passwordInputId || 'password-input';
        const input = document.getElementById(passwordInputId);
        if (!input) return;

        let currentPassword = input.dataset.passwordValue || '';

        if (value === 'clear') {
            // 전체 삭제
            currentPassword = '';
        } else if (value === 'backspace') {
            // 한 글자 삭제
            currentPassword = currentPassword.slice(0, -1);
        } else if (value >= '0' && value <= '9') {
            // 숫자 입력
            const maxLength = numpadElement.maxLength || 10;
            if (currentPassword.length < maxLength) {
                currentPassword += value;
            }
        }

        // 비밀번호 값 업데이트
        input.dataset.passwordValue = currentPassword;
        input.value = '*'.repeat(currentPassword.length);

        // 에러 메시지 숨기기
        const errorMsg = document.getElementById(`${passwordInputId}-error`);
        if (errorMsg) {
            errorMsg.style.display = 'none';
        }
    }

    /**
     * 터치 키보드 렌더링
     */
    renderTouchKeyboard(element, parent) {
        // 키보드 컨테이너가 이미 있으면 재사용
        let keyboardContainer = document.getElementById('sdui-touch-keyboard');
        if (!keyboardContainer) {
            keyboardContainer = document.createElement('div');
            keyboardContainer.id = 'sdui-touch-keyboard';
            keyboardContainer.className = 'sdui-touch-keyboard';
            document.body.appendChild(keyboardContainer);
        }

        // 네비게이션 바 높이 계산
        const navBar = document.querySelector('.sdui-navigation-bar');
        const navBarHeight = navBar ? navBar.offsetHeight : 85;

        // 키보드 기본 스타일 (높이 최소화)
        const defaultStyle = {
            position: 'fixed',
            bottom: navBarHeight + 'px',
            left: '0',
            right: '0',
            backgroundColor: '#f8f9fa',
            borderTop: '2px solid #dee2e6',
            padding: '8px',
            zIndex: '999',
            transform: 'translateY(100%)',
            transition: 'transform 0.3s ease-in-out',
            boxShadow: '0 -2px 10px rgba(0,0,0,0.1)',
            maxHeight: '40vh',
            overflowY: 'auto',
        };

        this.applyStyles(keyboardContainer, { ...defaultStyle, ...(element.style || {}) });

        // 키보드 내용 렌더링
        this._renderKeyboardContent(keyboardContainer);

        // 입력 필드 포커스 이벤트 리스너 추가
        this._setupInputFocusListeners();

        // 네비게이션 바 높이 변경 감지 및 키보드 높이 업데이트 (리사이즈 이벤트)
        const updateKeyboardPosition = debounce(() => {
            const currentNavBar = document.querySelector('.sdui-navigation-bar');
            const currentNavBarHeight = currentNavBar ? currentNavBar.offsetHeight : 85;
            keyboardContainer.style.bottom = currentNavBarHeight + 'px';
            
            // 키보드가 표시되어 있으면 body와 html padding-bottom 업데이트
            if (this.keyboardVisible) {
                requestAnimationFrame(() => {
                    const keyboardHeight = keyboardContainer.offsetHeight || keyboardContainer.getBoundingClientRect().height;
                    const totalHeight = keyboardHeight + currentNavBarHeight;
                    if (keyboardHeight > 0) {
                        document.body.style.paddingBottom = totalHeight + 'px';
                        document.documentElement.style.paddingBottom = totalHeight + 'px';
                    }
                });
            }
        }, RESOLUTION_CONSTANTS.TIMING.DEBOUNCE_DELAY);

        window.addEventListener('resize', updateKeyboardPosition);
    }

    /**
     * 키보드 내용 렌더링
     */
    _renderKeyboardContent(container) {
        container.innerHTML = '';

        // 키보드 레이아웃 렌더링
        const keyboardLayout = document.createElement('div');
        keyboardLayout.className = 'sdui-keyboard-layout';
        keyboardLayout.style.display = 'grid';
        keyboardLayout.style.gap = '4px';

        if (this.keyboardLayout === 'letters') {
            this._renderLetterKeys(keyboardLayout);
        } else {
            this._renderNumberKeys(keyboardLayout);
        }

        container.appendChild(keyboardLayout);
    }

    /**
     * 문자 키 렌더링
     */
    _renderLetterKeys(container) {
        const { isKIOSK, isPOS } = SDUIRenderer.getResolutionType();
        // 버튼 크기와 폰트 크기 줄이기
        const buttonSize = isPOS ? '35px' : isKIOSK ? '50px' : '42px';
        const buttonFontSize = isPOS ? '14px' : isKIOSK ? '20px' : '16px';

        if (this.keyboardMode === 'en') {
            // 영문 QWERTY 레이아웃
            const rows = [
                ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
                ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
                ['z', 'x', 'c', 'v', 'b', 'n', 'm']
            ];

            rows.forEach((row, rowIndex) => {
                const rowDiv = document.createElement('div');
                rowDiv.style.display = 'flex';
                rowDiv.style.justifyContent = 'center';
                rowDiv.style.gap = '4px';
                rowDiv.style.marginBottom = '4px';

                row.forEach(key => {
                    const keyButton = this._createKeyButton(key.toUpperCase(), key, buttonSize, buttonFontSize);
                    rowDiv.appendChild(keyButton);
                });

                // 마지막 행에 백스페이스 추가
                if (rowIndex === 2) {
                    const backspaceBtn = this._createKeyButton('⌫', 'backspace', buttonSize, buttonFontSize, true);
                    rowDiv.appendChild(backspaceBtn);
                }

                container.appendChild(rowDiv);
            });
        } else {
            // 한글 두벌식 레이아웃
            const rows = [
                ['ㅂ', 'ㅈ', 'ㄷ', 'ㄱ', 'ㅅ', 'ㅛ', 'ㅕ', 'ㅑ', 'ㅐ', 'ㅔ'],
                ['ㅁ', 'ㄴ', 'ㅇ', 'ㄹ', 'ㅎ', 'ㅗ', 'ㅓ', 'ㅏ', 'ㅣ'],
                ['ㅋ', 'ㅌ', 'ㅊ', 'ㅍ', 'ㅠ', 'ㅜ', 'ㅡ']
            ];

            rows.forEach((row, rowIndex) => {
                const rowDiv = document.createElement('div');
                rowDiv.style.display = 'flex';
                rowDiv.style.justifyContent = 'center';
                rowDiv.style.gap = '4px';
                rowDiv.style.marginBottom = '4px';

                row.forEach(key => {
                    const keyButton = this._createKeyButton(key, key, buttonSize, buttonFontSize);
                    rowDiv.appendChild(keyButton);
                });

                // 마지막 행에 백스페이스 추가
                if (rowIndex === 2) {
                    const backspaceBtn = this._createKeyButton('⌫', 'backspace', buttonSize, buttonFontSize, true);
                    rowDiv.appendChild(backspaceBtn);
                }

                container.appendChild(rowDiv);
            });
        }

        // 스페이스바 행 (한/영, 123 버튼 포함)
        const spaceRow = document.createElement('div');
        spaceRow.style.display = 'flex';
        spaceRow.style.justifyContent = 'center';
        spaceRow.style.alignItems = 'center';
        spaceRow.style.gap = '4px';
        spaceRow.style.marginTop = '4px';

        // 한/영 전환 버튼
        const langButton = document.createElement('button');
        langButton.textContent = this.keyboardMode === 'en' ? '한글' : '영문';
        langButton.className = 'sdui-keyboard-control-btn';
        langButton.style.padding = '6px 12px';
        langButton.style.borderRadius = '6px';
        langButton.style.border = '1px solid #dee2e6';
        langButton.style.backgroundColor = '#ffffff';
        langButton.style.cursor = 'pointer';
        langButton.style.fontSize = buttonFontSize;
        langButton.style.minWidth = buttonSize;
        langButton.style.minHeight = buttonSize;
        langButton.addEventListener('click', () => {
            this.keyboardMode = this.keyboardMode === 'en' ? 'ko' : 'en';
            const keyboardContainer = document.getElementById('sdui-touch-keyboard');
            if (keyboardContainer) {
                this._renderKeyboardContent(keyboardContainer);
            }
        });

        // 스페이스바
        const spaceBtn = this._createKeyButton('Space', ' ', 'auto', buttonFontSize, false, true);
        spaceBtn.style.flex = '1';
        spaceBtn.style.maxWidth = '400px';
        spaceBtn.style.minHeight = buttonSize;

        // 레이아웃 전환 버튼 (문자/숫자)
        const layoutButton = document.createElement('button');
        if (this.keyboardLayout === 'letters') {
            layoutButton.textContent = this.keyboardMode === 'ko' ? '123' : '123';
        } else {
            layoutButton.textContent = this.keyboardMode === 'ko' ? 'ㄱㄴㄷ' : 'ABC';
        }
        layoutButton.className = 'sdui-keyboard-control-btn';
        layoutButton.style.padding = '6px 12px';
        layoutButton.style.borderRadius = '6px';
        layoutButton.style.border = '1px solid #dee2e6';
        layoutButton.style.backgroundColor = '#ffffff';
        layoutButton.style.cursor = 'pointer';
        layoutButton.style.fontSize = buttonFontSize;
        layoutButton.style.minWidth = buttonSize;
        layoutButton.style.minHeight = buttonSize;
        layoutButton.addEventListener('click', () => {
            this.keyboardLayout = this.keyboardLayout === 'letters' ? 'numbers' : 'letters';
            const keyboardContainer = document.getElementById('sdui-touch-keyboard');
            if (keyboardContainer) {
                this._renderKeyboardContent(keyboardContainer);
            }
        });

        spaceRow.appendChild(langButton);
        spaceRow.appendChild(spaceBtn);
        spaceRow.appendChild(layoutButton);
        container.appendChild(spaceRow);
    }

    /**
     * 숫자 키 렌더링
     */
    _renderNumberKeys(container) {
        const { isKIOSK, isPOS } = SDUIRenderer.getResolutionType();
        // 버튼 크기와 폰트 크기 줄이기
        const buttonSize = isPOS ? '35px' : isKIOSK ? '50px' : '42px';
        const buttonFontSize = isPOS ? '14px' : isKIOSK ? '20px' : '16px';

        const rows = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['-', '/', ':', ';', '(', ')', '$', '&', '@', '"'],
            ['.', ',', '?', '!', "'", '~', '#', '%', '^', '*']
        ];

        rows.forEach((row, rowIndex) => {
            const rowDiv = document.createElement('div');
            rowDiv.style.display = 'flex';
            rowDiv.style.justifyContent = 'center';
            rowDiv.style.gap = '4px';
            rowDiv.style.marginBottom = '4px';

            row.forEach(key => {
                const keyButton = this._createKeyButton(key, key, buttonSize, buttonFontSize);
                rowDiv.appendChild(keyButton);
            });

            // 마지막 행에 백스페이스 추가
            if (rowIndex === 2) {
                const backspaceBtn = this._createKeyButton('⌫', 'backspace', buttonSize, buttonFontSize, true);
                rowDiv.appendChild(backspaceBtn);
            }

            container.appendChild(rowDiv);
        });

        // 스페이스바 행 (한/영, 123 버튼 포함)
        const spaceRow = document.createElement('div');
        spaceRow.style.display = 'flex';
        spaceRow.style.justifyContent = 'center';
        spaceRow.style.alignItems = 'center';
        spaceRow.style.gap = '4px';
        spaceRow.style.marginTop = '4px';

        // 한/영 전환 버튼
        const langButton = document.createElement('button');
        langButton.textContent = this.keyboardMode === 'en' ? '한글' : '영문';
        langButton.className = 'sdui-keyboard-control-btn';
        langButton.style.padding = '6px 12px';
        langButton.style.borderRadius = '6px';
        langButton.style.border = '1px solid #dee2e6';
        langButton.style.backgroundColor = '#ffffff';
        langButton.style.cursor = 'pointer';
        langButton.style.fontSize = buttonFontSize;
        langButton.style.minWidth = buttonSize;
        langButton.style.minHeight = buttonSize;
        langButton.addEventListener('click', () => {
            this.keyboardMode = this.keyboardMode === 'en' ? 'ko' : 'en';
            const keyboardContainer = document.getElementById('sdui-touch-keyboard');
            if (keyboardContainer) {
                this._renderKeyboardContent(keyboardContainer);
            }
        });

        // 스페이스바
        const spaceBtn = this._createKeyButton('Space', ' ', 'auto', buttonFontSize, false, true);
        spaceBtn.style.flex = '1';
        spaceBtn.style.maxWidth = '400px';
        spaceBtn.style.minHeight = buttonSize;

        // 레이아웃 전환 버튼 (문자/숫자)
        const layoutButton = document.createElement('button');
        if (this.keyboardLayout === 'letters') {
            layoutButton.textContent = this.keyboardMode === 'ko' ? '123' : '123';
        } else {
            layoutButton.textContent = this.keyboardMode === 'ko' ? 'ㄱㄴㄷ' : 'ABC';
        }
        layoutButton.className = 'sdui-keyboard-control-btn';
        layoutButton.style.padding = '6px 12px';
        layoutButton.style.borderRadius = '6px';
        layoutButton.style.border = '1px solid #dee2e6';
        layoutButton.style.backgroundColor = '#ffffff';
        layoutButton.style.cursor = 'pointer';
        layoutButton.style.fontSize = buttonFontSize;
        layoutButton.style.minWidth = buttonSize;
        layoutButton.style.minHeight = buttonSize;
        layoutButton.addEventListener('click', () => {
            this.keyboardLayout = this.keyboardLayout === 'letters' ? 'numbers' : 'letters';
            const keyboardContainer = document.getElementById('sdui-touch-keyboard');
            if (keyboardContainer) {
                this._renderKeyboardContent(keyboardContainer);
            }
        });

        spaceRow.appendChild(langButton);
        spaceRow.appendChild(spaceBtn);
        spaceRow.appendChild(layoutButton);
        container.appendChild(spaceRow);
    }

    /**
     * 키 버튼 생성
     */
    _createKeyButton(label, value, size, fontSize, isSpecial = false, isSpace = false) {
        const button = document.createElement('button');
        button.textContent = label;
        button.className = 'sdui-keyboard-key';
        button.dataset.keyValue = value;

        const buttonStyle = {
            minWidth: isSpace ? 'auto' : size,
            minHeight: isSpace ? size : size,
            fontSize: fontSize,
            fontWeight: '500',
            border: '1px solid #dee2e6',
            borderRadius: '6px',
            backgroundColor: isSpecial ? '#e9ecef' : '#ffffff',
            color: isSpecial ? '#dc3545' : '#333333',
            cursor: 'pointer',
            transition: 'all 0.1s ease',
            boxShadow: '0 1px 2px rgba(0,0,0,0.1)',
            userSelect: 'none',
            WebkitUserSelect: 'none',
            msUserSelect: 'none',
            padding: isSpace ? '4px 8px' : '0',
        };

        this.applyStyles(button, buttonStyle);

        // 호버 효과
        button.addEventListener('mouseenter', function() {
            this.style.backgroundColor = '#e9ecef';
            this.style.transform = 'scale(1.05)';
        });
        button.addEventListener('mouseleave', function() {
            this.style.backgroundColor = isSpecial ? '#e9ecef' : '#ffffff';
            this.style.transform = 'scale(1)';
        });

        // 터치 이벤트
        button.addEventListener('touchstart', function() {
            this.style.transform = 'scale(0.95)';
        });
        button.addEventListener('touchend', function() {
            this.style.transform = 'scale(1)';
        });

        // 클릭 이벤트
        button.addEventListener('click', () => {
            this.handleKeyboardInput(value);
        });

        return button;
    }

    /**
     * 키보드 입력 처리
     */
    handleKeyboardInput(value) {
        if (!this.currentInputElement) {
            // 현재 포커스된 입력 필드 찾기
            const activeElement = document.activeElement;
            if (activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA')) {
                this.currentInputElement = activeElement;
            } else {
                // 포커스된 요소가 없으면 첫 번째 입력 필드 찾기
                const firstInput = document.querySelector('input[type="text"], textarea');
                if (firstInput) {
                    this.currentInputElement = firstInput;
                    firstInput.focus();
                } else {
                    return; // 입력 필드가 없으면 무시
                }
            }
        }

        const input = this.currentInputElement;

        if (value === 'backspace') {
            // 백스페이스
            if (input.selectionStart > 0) {
                const start = input.selectionStart;
                const end = input.selectionEnd;
                const text = input.value;
                input.value = text.substring(0, start - 1) + text.substring(end);
                input.selectionStart = input.selectionEnd = start - 1;
            }
        } else if (value === ' ') {
            // 스페이스
            const start = input.selectionStart;
            const end = input.selectionEnd;
            const text = input.value;
            input.value = text.substring(0, start) + ' ' + text.substring(end);
            input.selectionStart = input.selectionEnd = start + 1;
        } else {
            // 일반 문자 입력
            const start = input.selectionStart;
            const end = input.selectionEnd;
            const text = input.value;
            input.value = text.substring(0, start) + value + text.substring(end);
            input.selectionStart = input.selectionEnd = start + value.length;
        }

        // 입력 이벤트 발생
        const inputEvent = new Event('input', { bubbles: true });
        input.dispatchEvent(inputEvent);
    }

    /**
     * 입력 필드 포커스 리스너 설정
     */
    _setupInputFocusListeners() {
        // 기존 리스너 제거 (중복 방지)
        document.removeEventListener('focusin', this._inputFocusHandler);
        
        // 새로운 리스너 추가
        this._inputFocusHandler = (e) => {
            if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) {
                // 비밀번호 입력 필드는 제외
                if (e.target.type !== 'password') {
                    this.currentInputElement = e.target;
                    
                    // 키보드가 표시되어 있으면 입력 필드로 스크롤 조정
                    if (this.keyboardVisible) {
                        const keyboardContainer = document.getElementById('sdui-touch-keyboard');
                        if (keyboardContainer) {
                            const keyboardHeight = keyboardContainer.offsetHeight;
                            if (keyboardHeight > 0) {
                                requestAnimationFrame(() => {
                                    this._scrollToInput(e.target, keyboardHeight);
                                });
                            }
                        }
                    }
                }
            }
        };
        
        document.addEventListener('focusin', this._inputFocusHandler);
    }

    /**
     * 키보드 토글
     */
    handleToggleKeyboard() {
        if (this.keyboardVisible) {
            this.hideKeyboard();
        } else {
            this.showKeyboard();
        }
    }

    /**
     * 키보드 표시
     */
    showKeyboard() {
        let keyboardContainer = document.getElementById('sdui-touch-keyboard');
        if (!keyboardContainer) {
            // 키보드가 없으면 생성
            this.renderTouchKeyboard({}, document.body);
            keyboardContainer = document.getElementById('sdui-touch-keyboard');
        }

        if (keyboardContainer) {
            // 네비게이션 바 높이 재계산 (정확한 높이를 위해 requestAnimationFrame 사용)
            requestAnimationFrame(() => {
                const navBar = document.querySelector('.sdui-navigation-bar');
                const navBarHeight = navBar ? navBar.offsetHeight : 85;
                keyboardContainer.style.bottom = navBarHeight + 'px';
                
                this.keyboardVisible = true;
                keyboardContainer.style.transform = 'translateY(0)';
                
                // 키보드 높이 계산 후 body와 html에 padding-bottom 추가하여 스크롤 가능하도록
                // 애니메이션 완료를 위해 여러 번 시도
                const updatePadding = () => {
                    const keyboardHeight = keyboardContainer.offsetHeight || keyboardContainer.getBoundingClientRect().height;
                    const navBar = document.querySelector('.sdui-navigation-bar');
                    const navBarHeight = navBar ? navBar.offsetHeight : 85;
                    const totalHeight = keyboardHeight + navBarHeight;
                    
                    if (keyboardHeight > 0) {
                        // body와 html 모두에 padding-bottom 추가
                        document.body.style.paddingBottom = totalHeight + 'px';
                        document.documentElement.style.paddingBottom = totalHeight + 'px';
                        
                        // 현재 포커스된 입력 필드가 있으면 스크롤 조정
                        if (this.currentInputElement) {
                            this._scrollToInput(this.currentInputElement, keyboardHeight);
                        }
                    }
                };
                
                // 즉시 시도
                requestAnimationFrame(() => {
                    updatePadding();
                    // 애니메이션 완료 후 다시 시도
                    setTimeout(() => {
                        updatePadding();
                    }, 350); // transition 시간 (0.3s) + 여유
                });
            });
        }
    }

    /**
     * 키보드 숨김
     */
    hideKeyboard() {
        const keyboardContainer = document.getElementById('sdui-touch-keyboard');
        if (keyboardContainer) {
            this.keyboardVisible = false;
            keyboardContainer.style.transform = 'translateY(100%)';
            
            // body와 html의 padding-bottom 제거
            document.body.style.paddingBottom = '';
            document.documentElement.style.paddingBottom = '';
        }
    }

    /**
     * 검색 토글
     */
    handleToggleSearch() {
        // ProductService가 있으면 검색 기능 호출
        if (window.ProductService && window.ProductService.toggleSearch) {
            window.ProductService.toggleSearch();
        } else {
            console.warn('ProductService.toggleSearch is not available');
        }
    }
    
    /**
     * 입력 필드로 스크롤 조정
     */
    _scrollToInput(inputElement, keyboardHeight) {
        if (!inputElement) return;
        
        // 입력 필드의 위치 계산
        const inputRect = inputElement.getBoundingClientRect();
        const inputBottom = inputRect.bottom;
        const viewportHeight = window.innerHeight;
        const navBar = document.querySelector('.sdui-navigation-bar');
        const navBarHeight = navBar ? navBar.offsetHeight : 85;
        
        // 키보드가 가리는 영역 계산 (네비게이션 바 높이 + 키보드 높이)
        const coveredArea = navBarHeight + keyboardHeight;
        
        // 입력 필드가 키보드에 가려지는지 확인
        const visibleArea = viewportHeight - coveredArea;
        
        if (inputBottom > visibleArea) {
            // 입력 필드가 가려지면 스크롤 조정
            const scrollAmount = inputBottom - visibleArea + 20; // 20px 여유 공간
            window.scrollBy({
                top: scrollAmount,
                behavior: 'smooth'
            });
        }
    }

    /**
     * 스타일 적용
     */
    applyStyles(element, styles) {
        // 버튼 요소이고 padding, fontSize, minHeight가 있으면 로그 출력 및 건너뛰기
        const isButton = element.classList && element.classList.contains('sdui-button');
        if (isButton && (styles.padding || styles.fontSize || styles.minHeight)) {
            console.log('[applyStyles] 버튼에 크기 관련 스타일 적용 시도 (건너뛰기):', {
                padding: styles.padding,
                fontSize: styles.fontSize,
                minHeight: styles.minHeight
            });
        }
        
        Object.keys(styles).forEach(key => {
            // 버튼의 경우 padding, fontSize, minHeight는 건너뛰기 (updateButtonSize에서 처리)
            if (isButton && (key === 'padding' || key === 'fontSize' || key === 'minHeight')) {
                debugLog('[applyStyles] 버튼 크기 스타일 건너뛰기:', key, styles[key]);
                return; // 건너뛰기
            }
            
            // CSS 속성명 변환 (camelCase to kebab-case)
            const cssKey = key.replace(/([A-Z])/g, '-$1').toLowerCase();
            
            // 특수 처리
            if (key === 'textAlign') {
                element.style.textAlign = styles[key];
            } else if (key === 'fontSize') {
                element.style.fontSize = styles[key];
            } else if (key === 'fontFamily') {
                element.style.fontFamily = styles[key];
            } else if (key === 'backgroundColor') {
                element.style.backgroundColor = styles[key];
            } else if (key === 'color') {
                element.style.color = styles[key];
            } else if (key === 'padding') {
                element.style.padding = styles[key];
            } else if (key === 'margin') {
                element.style.margin = styles[key];
            } else if (key === 'marginTop') {
                element.style.marginTop = styles[key];
            } else if (key === 'borderRadius') {
                element.style.borderRadius = styles[key];
            } else if (key === 'width') {
                element.style.width = styles[key];
            } else if (key === 'minHeight') {
                element.style.minHeight = styles[key];
            } else if (key === 'overflow') {
                element.style.overflow = styles[key];
            } else if (key === 'height') {
                element.style.height = styles[key];
            } else if (key === 'boxShadow') {
                element.style.boxShadow = styles[key];
            } else if (key === 'cursor') {
                element.style.cursor = styles[key];
            } else if (key === 'transition') {
                element.style.transition = styles[key];
            } else if (key === 'maxWidth') {
                element.style.maxWidth = styles[key];
            } else if (key === 'marginBottom') {
                element.style.marginBottom = styles[key];
            } else if (key === 'lineHeight') {
                element.style.lineHeight = styles[key];
            } else if (key === 'fontWeight') {
                element.style.fontWeight = styles[key];
            } else if (key === 'display') {
                element.style.display = styles[key];
            } else if (key === 'flexWrap') {
                element.style.flexWrap = styles[key];
            } else if (key === 'justifyContent') {
                element.style.justifyContent = styles[key];
            } else if (key === 'alignItems') {
                element.style.alignItems = styles[key];
            } else if (key === 'flexDirection') {
                element.style.flexDirection = styles[key];
            } else {
                element.style[cssKey] = styles[key];
            }
        });
    }

    /**
     * 액션 처리
     */
    async handleAction(action) {
        if (!action) return;

        switch (action.type) {
            case 'api_call':
                await this.handleApiCall(action);
                break;
            case 'navigate':
                this.handleNavigate(action);
                break;
            case 'go_back':
                this.handleGoBack();
                break;
            case 'go_forward':
                this.handleGoForward();
                break;
            case 'refresh':
                this.handleRefresh();
                break;
            case 'go_home':
                this.handleGoHome();
                break;
            case 'scroll_to_top':
                this.handleScrollToTop();
                break;
            case 'scroll_up':
                this.handleScrollUp();
                break;
            case 'scroll_down':
                this.handleScrollDown();
                break;
            case 'scroll_to_bottom':
                this.handleScrollToBottom();
                break;
            case 'close':
                this.handleClose();
                break;
            case 'verify_password':
                await this.handleVerifyPassword(action);
                break;
            case 'toggle_keyboard':
                this.handleToggleKeyboard();
                break;
            case 'toggle_search':
                this.handleToggleSearch();
                break;
            default:
                console.warn(`Unknown action type: ${action.type}`);
        }
    }

    /**
     * 네비게이션 처리
     */
    handleNavigate(action) {
        if (action.url) {
            window.location.href = action.url;
        }
    }

    /**
     * 뒤로 가기
     */
    handleGoBack() {
        if (window.history.length > 1) {
            window.history.back();
        } else {
            window.location.href = '/';
        }
    }

    /**
     * 앞으로 가기
     */
    handleGoForward() {
        window.history.forward();
    }

    /**
     * 새로고침
     */
    handleRefresh() {
        window.location.reload();
    }

    /**
     * 홈으로 이동
     */
    handleGoHome() {
        window.location.href = '/';
    }

    /**
     * 위로 스크롤
     */
    handleScrollUp() {
        // 현재 스크롤 위치에서 한 화면(viewport height)만큼 위로 스크롤
        const currentScroll = window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0;
        const viewportHeight = window.innerHeight;
        const newScroll = Math.max(0, currentScroll - viewportHeight);
        
        const scrollOptions = {
            top: newScroll,
            behavior: 'smooth'
        };
        
        // document.documentElement 스크롤
        if (document.documentElement.scrollHeight > window.innerHeight) {
            document.documentElement.scrollTo(scrollOptions);
        }
        
        // document.body 스크롤
        if (document.body.scrollHeight > window.innerHeight) {
            document.body.scrollTo(scrollOptions);
        }
        
        // window 스크롤 (가장 확실한 방법)
        window.scrollTo(scrollOptions);
    }

    /**
     * 아래로 스크롤
     */
    handleScrollDown() {
        // 현재 스크롤 위치에서 한 화면(viewport height)만큼 아래로 스크롤
        const currentScroll = window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0;
        const viewportHeight = window.innerHeight;
        
        // 최대 스크롤 위치 계산
        const maxScroll = Math.max(
            document.documentElement.scrollHeight - window.innerHeight,
            document.body.scrollHeight - window.innerHeight,
            0
        );
        
        // 한 화면만큼 아래로 이동 (최대 스크롤 위치를 넘지 않도록)
        const newScroll = Math.min(maxScroll, currentScroll + viewportHeight);
        
        const scrollOptions = {
            top: newScroll,
            behavior: 'smooth'
        };
        
        // document.documentElement 스크롤
        if (document.documentElement.scrollHeight > window.innerHeight) {
            document.documentElement.scrollTo(scrollOptions);
        }
        
        // document.body 스크롤
        if (document.body.scrollHeight > window.innerHeight) {
            document.body.scrollTo(scrollOptions);
        }
        
        // window 스크롤 (가장 확실한 방법)
        window.scrollTo(scrollOptions);
    }

    /**
     * 페이지 최상단으로 이동
     */
    handleScrollToTop() {
        const scrollOptions = {
            top: 0,
            behavior: 'smooth'
        };
        
        // document.documentElement 스크롤
        if (document.documentElement.scrollHeight > window.innerHeight) {
            document.documentElement.scrollTo(scrollOptions);
        }
        
        // document.body 스크롤
        if (document.body.scrollHeight > window.innerHeight) {
            document.body.scrollTo(scrollOptions);
        }
        
        // window 스크롤 (가장 확실한 방법)
        window.scrollTo(scrollOptions);
    }

    /**
     * 페이지 최하단으로 이동
     */
    handleScrollToBottom() {
        // 최대 스크롤 위치 계산
        const maxScroll = Math.max(
            document.documentElement.scrollHeight - window.innerHeight,
            document.body.scrollHeight - window.innerHeight,
            0
        );
        
        const scrollOptions = {
            top: maxScroll,
            behavior: 'smooth'
        };
        
        // document.documentElement 스크롤
        if (document.documentElement.scrollHeight > window.innerHeight) {
            document.documentElement.scrollTo(scrollOptions);
        }
        
        // document.body 스크롤
        if (document.body.scrollHeight > window.innerHeight) {
            document.body.scrollTo(scrollOptions);
        }
        
        // window 스크롤 (가장 확실한 방법)
        window.scrollTo(scrollOptions);
    }

    /**
     * 닫기 (브라우저 종료 - 메인 페이지에서만 사용)
     */
    async handleClose() {
        // 웹 브라우저가 실행되는 PC의 HTTP 서버 엔드포인트로 종료 요청
        try {
            const response = await fetch('/api/edgeman/terminate-browser', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({}),
            });

            if (response.ok) {
                const result = await response.json();
                // EdgeMan 서버 통신 실패 경고가 있는 경우 에러 팝업 표시
                if (result.warning) {
                    this.showErrorPopup('EdgeMan 과 통신할 수 없습니다.\n브라우저를 닫을 수 없습니다.');
                    return;
                }
                // 서버에서 종료 요청이 성공적으로 처리됨
                debugLog('브라우저 종료 요청이 서버로 전송되었습니다.');
            } else {
                const errorResult = await response.json().catch(() => ({}));
                const errorMessage = errorResult.message || errorResult.detail || 'EdgeMan 과 통신할 수 없습니다.\n브라우저를 닫을 수 없습니다.';
                this.showErrorPopup(errorMessage);
            }
        } catch (error) {
            debugWarn('브라우저 종료 요청 중 오류 발생:', error);
            // 네트워크 오류 등으로 서버 요청이 실패한 경우 에러 팝업 표시
            this.showErrorPopup('EdgeMan 과 통신할 수 없습니다.\n브라우저를 닫을 수 없습니다.');
        }
    }

    /**
     * 브라우저 종료 대체 방법 (서버 요청 실패 시)
     */
    _tryCloseBrowser() {
        // 키오스크/POS 환경에서 브라우저 종료
        // window.close()는 사용자가 직접 연 창에서만 작동하므로
        // 키오스크 환경에서는 실제로는 작동하지 않을 수 있음
        if (window.navigator && window.navigator.app) {
            // Electron 환경
            if (window.navigator.app.exitApp) {
                window.navigator.app.exitApp();
            } else {
                window.close();
            }
        } else {
            // 일반 브라우저 환경
            window.close();
            // 창을 닫을 수 없는 경우 홈으로 이동
            setTimeout(() => {
                window.location.href = '/';
            }, 100);
        }
    }

    /**
     * 비밀번호 검증 처리
     */
    async handleVerifyPassword(action) {
        const passwordInputId = action.passwordInputId || 'password-input';
        const input = document.getElementById(passwordInputId);
        if (!input) {
            console.error('Password input not found');
            return;
        }

        const password = input.dataset.passwordValue || '';
        const redirectUrl = action.redirectUrl || '/settings';

        try {
            const response = await fetch('/api/auth/verify-password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    password: password,
                    redirect_url: redirectUrl
                }),
            });

            if (!response.ok) {
                // HTTP 에러 응답 처리
                const errorResult = await response.json();
                const errorMessage = errorResult.detail || '비밀번호가 일치하지 않습니다.';
                
                // 팝업 메시지 표시
                this.showErrorPopup(errorMessage);

                // 입력 초기화
                input.dataset.passwordValue = '';
                input.value = '';

                // 입력 필드에 에러 스타일 적용
                input.style.borderColor = '#dc3545';
                setTimeout(() => {
                    input.style.borderColor = '#dee2e6';
                }, 2000);
                return;
            }

            const result = await response.json();

            if (result.success) {
                // 비밀번호가 맞으면 리다이렉트
                window.location.href = redirectUrl;
            } else {
                // 비밀번호가 틀리면 팝업 메시지 표시 및 입력 초기화
                const errorMessage = result.message || result.detail || '비밀번호가 일치하지 않습니다.';
                this.showErrorPopup(errorMessage);

                // 입력 초기화
                input.dataset.passwordValue = '';
                input.value = '';

                // 입력 필드에 에러 스타일 적용
                input.style.borderColor = '#dc3545';
                setTimeout(() => {
                    input.style.borderColor = '#dee2e6';
                }, 2000);
            }
        } catch (error) {
            console.error('Password verification failed:', error);
            this.showErrorPopup('서버와 통신할 수 없습니다.');
        }
    }

    /**
     * API 호출 처리
     */
    async handleApiCall(action) {
        try {
            const options = {
                method: action.method || 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            };

            if (action.method === 'POST' && action.body) {
                options.body = JSON.stringify(action.body);
            }

            const response = await fetch(action.url, options);
            const data = await response.json();

            // 성공 콜백 처리
            if (action.onSuccess === 'showResult') {
                this.showResult(data);
            } else if (action.onSuccess && typeof window[action.onSuccess] === 'function') {
                window[action.onSuccess](data);
            }
        } catch (error) {
            console.error('API 호출 실패:', error);
            this.showResult({ error: error.message });
        }
    }

    /**
     * 결과 표시
     */
    showResult(data) {
        const resultContainer = document.getElementById('result-container');
        if (resultContainer) {
            resultContainer.innerHTML = '';
            const pre = document.createElement('pre');
            pre.style.margin = '0';
            pre.style.whiteSpace = 'pre-wrap';
            pre.style.wordBreak = 'break-word';
            pre.textContent = JSON.stringify(data, null, 2);
            resultContainer.appendChild(pre);
        } else {
            // result-container가 없으면 alert로 표시
            alert(JSON.stringify(data, null, 2));
        }
    }

    /**
     * 에러 팝업 표시
     */
    showErrorPopup(message) {
        // 이미 제거 중이면 무시
        if (this.errorPopupRemoving) {
            return;
        }
        
        // 기존 팝업이 있으면 즉시 제거 (애니메이션 없이)
        const existingPopup = document.getElementById('sdui-error-popup');
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
        overlay.id = 'sdui-error-popup';
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
            // 이벤트 전파 중지
            if (e) {
                e.stopPropagation();
                e.preventDefault();
            }
            
            // 이미 제거 중이거나 오버레이가 없으면 무시
            if (this.errorPopupRemoving || !overlay || !overlay.parentNode) {
                return;
            }
            
            this.errorPopupRemoving = true;
            
            // 오버레이의 모든 이벤트 리스너 제거를 위해 pointer-events 비활성화
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
                // 플래그 해제 (약간의 지연을 두어 깜빡임 방지)
                setTimeout(() => {
                    this.errorPopupRemoving = false;
                }, 100);
            }, 200);
        };

        // 팝업 박스 클릭 시 이벤트 전파 차단 (오버레이 클릭 방지)
        popup.addEventListener('click', (e) => {
            e.stopPropagation();
        });

        // 버튼 클릭 시 팝업 닫기
        button.addEventListener('click', (e) => {
            e.stopPropagation();
            removePopup(e);
        });

        // 오버레이 클릭 시 팝업 닫기 (오버레이 자체만 클릭했을 때)
        overlay.addEventListener('click', (e) => {
            // 오버레이 자체를 클릭한 경우에만 닫기
            if (e.target === overlay) {
                removePopup(e);
            }
        });

        popup.appendChild(button);
        overlay.appendChild(popup);
        document.body.appendChild(overlay);

        // 애니메이션 스타일 추가 (없는 경우)
        if (!document.getElementById('sdui-popup-styles')) {
            const style = document.createElement('style');
            style.id = 'sdui-popup-styles';
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
}

// 전역으로 노출 (필요한 경우)
if (typeof window !== 'undefined') {
    window.SDUIRenderer = SDUIRenderer;
}

