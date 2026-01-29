/**
 * SearchKeyboard - 검색 모달용 한글/영문 가상 키보드
 * 
 * 한글 조합 입력을 지원하는 터치 키보드 모듈
 */
(function() {
    'use strict';

    /**
     * SearchKeyboard 클래스
     */
    class SearchKeyboard {
        constructor(container, inputElement) {
            // 유효성 검사
            if (!container) {
                throw new Error('[SearchKeyboard] container가 필요합니다.');
            }
            if (!inputElement) {
                throw new Error('[SearchKeyboard] inputElement가 필요합니다.');
            }
            
            this.container = container;
            this.inputElement = inputElement;
            
            // 키보드 상태
            this.keyboardMode = 'ko'; // 'ko' 또는 'en'
            this.keyboardLayout = 'letters'; // 'letters' 또는 'numbers'
            this.isShiftPressed = false; // Shift 키 상태
            
            // 한글 조합 상태 추적
            this.composingState = {
                isComposing: false,
                currentComposition: '',
                lastChar: '',
                cho: null,  // 초성 (자음)
                jung: null, // 중성 (모음)
                jong: null // 종성 (자음)
            };
            
            // 한글 유니코드 범위
            this.HANGUL_START = 0xAC00;  // '가'
            this.HANGUL_END = 0xD7A3;    // '힣'
            this.CHO_START = 0x3131;     // 'ㄱ'
            this.CHO_END = 0x314E;       // 'ㅎ'
            this.JUNG_START = 0x314F;    // 'ㅏ'
            this.JUNG_END = 0x3163;      // 'ㅣ'
            
            // 초기 렌더링
            this.render();
        }
        
        /**
         * 한글 조합 함수
         */
        combineHangul(cho, jung, jong) {
            if (cho === null || jung === null) return null;
            
            const choChar = cho.charCodeAt(0);
            const jungChar = jung.charCodeAt(0);
            
            // 자음 매핑: ㄱ(0x3131)=0, ㄲ(0x3132)=1, ㄴ(0x3134)=2, ..., ㅎ(0x314E)=18
            const choMap = {
                0x3131: 0,  // ㄱ
                0x3132: 1,  // ㄲ
                0x3134: 2,  // ㄴ
                0x3137: 3,  // ㄷ
                0x3138: 4,  // ㄸ
                0x3139: 5,  // ㄹ
                0x3141: 6,  // ㅁ
                0x3142: 7,  // ㅂ
                0x3143: 8,  // ㅃ
                0x3145: 9,  // ㅅ
                0x3146: 10, // ㅆ
                0x3147: 11, // ㅇ
                0x3148: 12, // ㅈ
                0x3149: 13, // ㅉ
                0x314A: 14, // ㅊ
                0x314B: 15, // ㅋ
                0x314C: 16, // ㅌ
                0x314D: 17, // ㅍ
                0x314E: 18  // ㅎ
            };
            
            // 모음 매핑: ㅏ(0x314F)=0, ㅐ(0x3150)=1, ..., ㅣ(0x3163)=20
            const jungMap = {
                0x314F: 0,  // ㅏ
                0x3150: 1,  // ㅐ
                0x3151: 2,  // ㅑ
                0x3152: 3,  // ㅒ
                0x3153: 4,  // ㅓ
                0x3154: 5,  // ㅔ
                0x3155: 6,  // ㅕ
                0x3156: 7,  // ㅖ
                0x3157: 8,  // ㅗ
                0x3158: 9,  // ㅘ
                0x3159: 10, // ㅙ
                0x315A: 11, // ㅚ
                0x315B: 12, // ㅛ
                0x315C: 13, // ㅜ
                0x315D: 14, // ㅝ
                0x315E: 15, // ㅞ
                0x315F: 16, // ㅟ
                0x3160: 17, // ㅠ
                0x3161: 18, // ㅡ
                0x3162: 19, // ㅢ
                0x3163: 20  // ㅣ
            };
            
            // 종성 매핑 (없음=0, ㄱ=1, ㄲ=2, ㄳ=3, ㄴ=4, ..., ㅎ=27)
            const jongMap = {
                0x3131: 1,  // ㄱ
                0x3132: 2,  // ㄲ
                0x3133: 3,  // ㄳ
                0x3134: 4,  // ㄴ
                0x3135: 5,  // ㄵ
                0x3136: 6,  // ㄶ
                0x3137: 7,  // ㄷ
                0x3139: 8,  // ㄹ
                0x313A: 9,  // ㄺ
                0x313B: 10, // ㄻ
                0x313C: 11, // ㄼ
                0x313D: 12, // ㄽ
                0x313E: 13, // ㄾ
                0x313F: 14, // ㄿ
                0x3140: 15, // ㅀ
                0x3141: 16, // ㅁ
                0x3142: 17, // ㅂ
                0x3144: 18, // ㅄ
                0x3145: 19, // ㅅ
                0x3146: 20, // ㅆ
                0x3147: 21, // ㅇ
                0x3148: 22, // ㅈ
                0x314A: 23, // ㅊ
                0x314B: 24, // ㅋ
                0x314C: 25, // ㅌ
                0x314D: 26, // ㅍ
                0x314E: 27  // ㅎ
            };
            
            const choIdx = choMap[choChar];
            const jungIdx = jungMap[jungChar];
            
            // 종성 검증
            let jongIdx = 0;
            if (jong !== null) {
                const jongChar = jong.charCodeAt(0);
                const mappedJongIdx = jongMap[jongChar];
                if (mappedJongIdx === undefined) {
                    console.warn('[SearchKeyboard] 한글 조합 실패 - 종성으로 사용할 수 없는 자음:', { 
                        cho: String.fromCharCode(choChar), 
                        jung: String.fromCharCode(jungChar), 
                        jong: String.fromCharCode(jongChar),
                        jongChar: '0x' + jongChar.toString(16)
                    });
                    return null;
                }
                jongIdx = mappedJongIdx;
            }
            
            if (choIdx === undefined || jungIdx === undefined) {
                console.warn('[SearchKeyboard] 한글 조합 실패 - 매핑 없음:', { 
                    cho: String.fromCharCode(choChar), 
                    jung: String.fromCharCode(jungChar), 
                    choChar: '0x' + choChar.toString(16),
                    jungChar: '0x' + jungChar.toString(16),
                    choIdx, 
                    jungIdx 
                });
                return null;
            }
            
            const code = this.HANGUL_START + choIdx * 588 + jungIdx * 28 + jongIdx;
            const result = String.fromCharCode(code);
            console.log('[SearchKeyboard] 한글 조합 성공:', { 
                cho, 
                jung, 
                jong, 
                choIdx, 
                jungIdx, 
                jongIdx, 
                code: '0x' + code.toString(16), 
                result 
            });
            return result;
        }
        
        /**
         * 문자 타입 판별
         */
        getCharType(char) {
            const code = char.charCodeAt(0);
            if (code >= this.CHO_START && code <= this.CHO_END) return 'cho'; // 자음
            if (code >= this.JUNG_START && code <= this.JUNG_END) return 'jung'; // 모음
            if (code >= this.HANGUL_START && code <= this.HANGUL_END) return 'hangul'; // 완성형
            return 'other';
        }
        
        /**
         * 복합 모음 조합 함수
         */
        combineJung(jung1, jung2) {
            const complexJungMap = {
                'ㅗㅏ': 'ㅘ',
                'ㅗㅐ': 'ㅙ',
                'ㅗㅣ': 'ㅚ',
                'ㅜㅓ': 'ㅝ',
                'ㅜㅔ': 'ㅞ',
                'ㅜㅣ': 'ㅟ',
                'ㅡㅣ': 'ㅢ'
            };
            
            const key = jung1 + jung2;
            return complexJungMap[key] || null;
        }
        
        /**
         * 복합 종성 조합 함수
         */
        combineJong(jong1, jong2) {
            const complexJongMap = {
                'ㄹㄱ': 'ㄺ',
                'ㄹㅁ': 'ㄻ',
                'ㄹㅂ': 'ㄼ',
                'ㄹㅅ': 'ㄽ',
                'ㄹㅌ': 'ㄾ',
                'ㄹㅍ': 'ㄿ',
                'ㄹㅎ': 'ㅀ',
                'ㅂㅅ': 'ㅄ',
                'ㄴㅈ': 'ㄵ',
                'ㄴㅎ': 'ㄶ',
                'ㄱㅅ': 'ㄳ'
            };
            
            const key = jong1 + jong2;
            return complexJongMap[key] || null;
        }
        
        /**
         * 복합 종성인지 확인하는 함수
         */
        isComplexJong(jong) {
            const complexJongs = ['ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ', 'ㄵ', 'ㄶ', 'ㄳ'];
            return complexJongs.includes(jong);
        }
        
        /**
         * 종성을 초성으로 사용할 수 있는지 확인하는 함수
         */
        canUseJongAsCho(jong) {
            // 복합 종성은 초성으로 사용할 수 없음
            if (this.isComplexJong(jong)) {
                return false;
            }
            // 단일 종성은 초성으로 사용 가능
            return true;
        }
        
        /**
         * 복합 종성을 분해하는 함수 (첫 번째 자음만 반환)
         */
        decomposeComplexJong(complexJong) {
            const complexJongDecompose = {
                'ㄺ': 'ㄹ',
                'ㄻ': 'ㄹ',
                'ㄼ': 'ㄹ',
                'ㄽ': 'ㄹ',
                'ㄾ': 'ㄹ',
                'ㄿ': 'ㄹ',
                'ㅀ': 'ㄹ',
                'ㅄ': 'ㅂ',
                'ㄵ': 'ㄴ',
                'ㄶ': 'ㄴ',
                'ㄳ': 'ㄱ'
            };
            
            return complexJongDecompose[complexJong] || complexJong;
        }
        
        /**
         * 복합 종성을 분해하여 두 번째 자음을 반환하는 함수
         */
        getSecondJongFromComplex(complexJong) {
            const complexJongSecond = {
                'ㄺ': 'ㄱ',
                'ㄻ': 'ㅁ',
                'ㄼ': 'ㅂ',
                'ㄽ': 'ㅅ',
                'ㄾ': 'ㅌ',
                'ㄿ': 'ㅍ',
                'ㅀ': 'ㅎ',
                'ㅄ': 'ㅅ',
                'ㄵ': 'ㅈ',
                'ㄶ': 'ㅎ',
                'ㄳ': 'ㅅ'
            };
            
            return complexJongSecond[complexJong] || null;
        }
        
        /**
         * 완성형 한글을 초성, 중성, 종성으로 분해
         */
        decomposeHangul(hangul) {
            const code = hangul.charCodeAt(0);
            if (code < this.HANGUL_START || code > this.HANGUL_END) return null;
            
            const base = code - this.HANGUL_START;
            const choIdx = Math.floor(base / 588);
            const jungIdx = Math.floor((base % 588) / 28);
            const jongIdx = base % 28;
            
            // 초성 매핑 (역방향)
            const choChars = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];
            // 중성 매핑 (역방향)
            const jungChars = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'];
            // 종성 매핑 (역방향, 없음 포함)
            const jongChars = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];
            
            return {
                cho: choChars[choIdx] || null,
                jung: jungChars[jungIdx] || null,
                jong: jongIdx > 0 ? jongChars[jongIdx] || null : null
            };
        }
        
        /**
         * 키보드 입력 처리 함수
         */
        handleKeyboardInput(value) {
            console.log('[SearchKeyboard] handleKeyboardInput 호출:', { value, keyboardMode: this.keyboardMode, keyboardLayout: this.keyboardLayout });
            
            if (!this.inputElement) {
                console.warn('[SearchKeyboard] inputElement이 없습니다');
                return;
            }
            
            // 입력 필드에 포커스가 없으면 포커스 설정
            if (document.activeElement !== this.inputElement) {
                console.log('[SearchKeyboard] 포커스 설정');
                this.inputElement.focus();
            }
            
            const currentValue = this.inputElement.value;
            const selectionStart = this.inputElement.selectionStart || 0;
            const selectionEnd = this.inputElement.selectionEnd || 0;
            console.log('[SearchKeyboard] 현재 입력 상태:', { 
                currentValue, 
                selectionStart, 
                selectionEnd,
                isComposing: this.composingState.isComposing,
                currentComposition: this.composingState.currentComposition
            });
            
            if (value === 'backspace') {
                this.handleBackspace(currentValue, selectionStart, selectionEnd);
            } else if (value === ' ') {
                this.handleSpace(currentValue, selectionStart, selectionEnd);
            } else {
                this.handleCharInput(value, currentValue, selectionStart, selectionEnd);
            }
        }
        
        /**
         * 백스페이스 처리
         */
        handleBackspace(currentValue, selectionStart, selectionEnd) {
            console.log('[SearchKeyboard] 백스페이스 처리');
            if (selectionStart > 0) {
                const newValue = currentValue.substring(0, selectionStart - 1) + currentValue.substring(selectionEnd);
                this.inputElement.value = newValue;
                const newCursorPos = selectionStart - 1;
                this.inputElement.selectionStart = newCursorPos;
                this.inputElement.selectionEnd = newCursorPos;
                
                const inputEvent = new Event('input', { bubbles: true });
                this.inputElement.dispatchEvent(inputEvent);
                console.log('[SearchKeyboard] 백스페이스 처리 완료:', { newValue, cursorPosition: newCursorPos });
            }
        }
        
        /**
         * 스페이스 처리
         */
        handleSpace(currentValue, selectionStart, selectionEnd) {
            console.log('[SearchKeyboard] 스페이스 처리');
            const newValue = currentValue.substring(0, selectionStart) + ' ' + currentValue.substring(selectionEnd);
            this.inputElement.value = newValue;
            const newCursorPos = selectionStart + 1;
            this.inputElement.selectionStart = newCursorPos;
            this.inputElement.selectionEnd = newCursorPos;
            
            const inputEvent = new Event('input', { bubbles: true });
            this.inputElement.dispatchEvent(inputEvent);
            console.log('[SearchKeyboard] 스페이스 처리 완료:', { newValue, cursorPosition: newCursorPos });
        }
        
        /**
         * 문자 입력 처리 (한글 조합 로직 포함)
         */
        handleCharInput(value, currentValue, selectionStart, selectionEnd) {
            console.log('[SearchKeyboard] 문자 입력 처리:', { value, keyboardMode: this.keyboardMode, composingState: this.composingState });
            
            const charType = this.getCharType(value);
            console.log('[SearchKeyboard] 문자 타입:', { value, charType });
            
            let insertChar = value;
            let newComposingState = { ...this.composingState };
            
            // 한글 모드이고 자음/모음인 경우 조합 시도
            if (this.keyboardMode === 'ko' && (charType === 'cho' || charType === 'jung')) {
                const beforeChar = selectionStart > 0 ? currentValue[selectionStart - 1] : null;
                const beforeCharType = beforeChar ? this.getCharType(beforeChar) : null;
                
                console.log('[SearchKeyboard] 조합 시도:', { beforeChar, beforeCharType, currentChar: value, charType, composingState: this.composingState });
                
                if (charType === 'cho') {
                    // 자음 입력 처리
                    const result = this.handleChoInput(value, beforeChar, beforeCharType, currentValue, selectionStart, selectionEnd);
                    if (result) {
                        newComposingState = result.composingState;
                        if (result.handled) return;
                    }
                } else if (charType === 'jung') {
                    // 모음 입력 처리
                    const result = this.handleJungInput(value, beforeChar, beforeCharType, currentValue, selectionStart, selectionEnd);
                    if (result) {
                        newComposingState = result.composingState;
                        if (result.handled) return;
                    }
                }
            } else {
                // 영문이나 숫자 등은 조합 상태 초기화
                newComposingState = { isComposing: false, currentComposition: '', lastChar: '', cho: null, jung: null, jong: null };
            }
            
            // 문자 삽입
            const newValue = currentValue.substring(0, selectionStart) + insertChar + currentValue.substring(selectionEnd);
            this.inputElement.value = newValue;
            const newCursorPos = selectionStart + insertChar.length;
            this.inputElement.selectionStart = newCursorPos;
            this.inputElement.selectionEnd = newCursorPos;
            this.composingState = newComposingState;
            
            console.log('[SearchKeyboard] 문자 삽입 완료:', { 
                oldValue: currentValue, 
                newValue, 
                insertedChar: insertChar,
                cursorPosition: newCursorPos,
                composingState: this.composingState
            });
            
            // input 이벤트 발생
            const inputEvent = new Event('input', { bubbles: true });
            this.inputElement.dispatchEvent(inputEvent);
            console.log('[SearchKeyboard] input 이벤트 발생 완료');
        }
        
        /**
         * 자음 입력 처리
         */
        handleChoInput(value, beforeChar, beforeCharType, currentValue, selectionStart, selectionEnd) {
            if (beforeCharType === 'hangul') {
                // 완성형 한글 뒤에 자음이 오면 먼저 종성으로 조합 시도
                const decomposed = this.decomposeHangul(beforeChar);
                console.log('[SearchKeyboard] 완성형 한글 분해:', { beforeChar, decomposed });
                
                if (decomposed) {
                    if (decomposed.jong === null) {
                        // 종성이 없는 경우: 종성으로 조합 시도
                        const combined = this.combineHangul(decomposed.cho, decomposed.jung, value);
                        if (combined) {
                            const newValue = currentValue.substring(0, selectionStart - 1) + combined + currentValue.substring(selectionEnd);
                            this.inputElement.value = newValue;
                            const newCursorPos = selectionStart;
                            this.inputElement.selectionStart = newCursorPos;
                            this.inputElement.selectionEnd = newCursorPos;
                            const newComposingState = {
                                ...this.composingState,
                                cho: decomposed.cho,
                                jung: decomposed.jung,
                                jong: value,
                                isComposing: false
                            };
                            this.composingState = newComposingState;
                            
                            const inputEvent = new Event('input', { bubbles: true });
                            this.inputElement.dispatchEvent(inputEvent);
                            console.log('[SearchKeyboard] 종성 추가 조합 완료:', { combined, newValue, composingState: this.composingState });
                            return { handled: true, composingState: newComposingState };
                        }
                    } else {
                        // 종성이 있는 경우: 복합 종성 조합 시도
                        const complexJong = this.combineJong(decomposed.jong, value);
                        if (complexJong) {
                            const combined = this.combineHangul(decomposed.cho, decomposed.jung, complexJong);
                            if (combined) {
                                const newValue = currentValue.substring(0, selectionStart - 1) + combined + currentValue.substring(selectionEnd);
                                this.inputElement.value = newValue;
                                const newCursorPos = selectionStart;
                                this.inputElement.selectionStart = newCursorPos;
                                this.inputElement.selectionEnd = newCursorPos;
                                const newComposingState = {
                                    ...this.composingState,
                                    cho: decomposed.cho,
                                    jung: decomposed.jung,
                                    jong: complexJong,
                                    isComposing: false
                                };
                                this.composingState = newComposingState;
                                
                                const inputEvent = new Event('input', { bubbles: true });
                                this.inputElement.dispatchEvent(inputEvent);
                                console.log('[SearchKeyboard] 복합 종성 조합 완료:', { complexJong, combined, newValue, composingState: this.composingState });
                                return { handled: true, composingState: newComposingState };
                            }
                        }
                    }
                }
                // 종성이 이미 있거나 조합 실패 시 새로운 초성으로 시작
                console.log('[SearchKeyboard] 완성형 한글 뒤 자음 입력 - 새로운 초성으로 시작');
                return {
                    handled: false,
                    composingState: {
                        ...this.composingState,
                        cho: value,
                        jung: null,
                        jong: null,
                        isComposing: true
                    }
                };
            } else if (beforeCharType === 'jung') {
                // 모음 뒤에 자음이 오면
                if (this.composingState.cho && this.composingState.jung) {
                    // 조합 중인 상태: 종성으로 조합 시도
                    const combined = this.combineHangul(this.composingState.cho, this.composingState.jung, value);
                    if (combined) {
                        const newValue = currentValue.substring(0, selectionStart - 1) + combined + currentValue.substring(selectionEnd);
                        this.inputElement.value = newValue;
                        const newCursorPos = selectionStart;
                        this.inputElement.selectionStart = newCursorPos;
                        this.inputElement.selectionEnd = newCursorPos;
                        const newComposingState = {
                            ...this.composingState,
                            jong: value,
                            isComposing: false
                        };
                        this.composingState = newComposingState;
                        
                        const inputEvent = new Event('input', { bubbles: true });
                        this.inputElement.dispatchEvent(inputEvent);
                        console.log('[SearchKeyboard] 종성 조합 완료:', { combined, newValue, composingState: this.composingState });
                        return { handled: true, composingState: newComposingState };
                    }
                }
                // 모음 뒤에 자음이 오면 초성+중성으로 조합
                const combined = this.combineHangul(value, beforeChar, null);
                if (combined) {
                    const newValue = currentValue.substring(0, selectionStart - 1) + combined + currentValue.substring(selectionEnd);
                    this.inputElement.value = newValue;
                    const newCursorPos = selectionStart;
                    this.inputElement.selectionStart = newCursorPos;
                    this.inputElement.selectionEnd = newCursorPos;
                    const newComposingState = {
                        ...this.composingState,
                        cho: value,
                        jung: beforeChar,
                        jong: null,
                        isComposing: true
                    };
                    this.composingState = newComposingState;
                    
                    const inputEvent = new Event('input', { bubbles: true });
                    this.inputElement.dispatchEvent(inputEvent);
                    console.log('[SearchKeyboard] 모음 뒤 자음 조합 완료:', { combined, newValue, composingState: this.composingState });
                    return { handled: true, composingState: newComposingState };
                }
                // 조합 실패 시 새로운 초성 시작
                return {
                    handled: false,
                    composingState: {
                        ...this.composingState,
                        cho: value,
                        jung: null,
                        jong: null,
                        isComposing: true
                    }
                };
            } else {
                // 새로운 초성 시작
                return {
                    handled: false,
                    composingState: {
                        ...this.composingState,
                        cho: value,
                        jung: null,
                        jong: null,
                        isComposing: true
                    }
                };
            }
        }
        
        /**
         * 모음 입력 처리
         */
        handleJungInput(value, beforeChar, beforeCharType, currentValue, selectionStart, selectionEnd) {
            if (beforeCharType === 'cho') {
                // 자음 뒤에 모음이 오면 조합
                const combined = this.combineHangul(beforeChar, value, null);
                if (combined) {
                    const newValue = currentValue.substring(0, selectionStart - 1) + combined + currentValue.substring(selectionEnd);
                    this.inputElement.value = newValue;
                    const newCursorPos = selectionStart;
                    this.inputElement.selectionStart = newCursorPos;
                    this.inputElement.selectionEnd = newCursorPos;
                    const newComposingState = {
                        ...this.composingState,
                        cho: beforeChar,
                        jung: value,
                        jong: null,
                        isComposing: true
                    };
                    this.composingState = newComposingState;
                    
                    const inputEvent = new Event('input', { bubbles: true });
                    this.inputElement.dispatchEvent(inputEvent);
                    console.log('[SearchKeyboard] 초성+중성 조합 완료:', { combined, newValue, composingState: this.composingState });
                    return { handled: true, composingState: newComposingState };
                }
            } else if (beforeCharType === 'hangul') {
                // 완성형 한글 뒤에 모음이 오면
                const decomposed = this.decomposeHangul(beforeChar);
                console.log('[SearchKeyboard] 완성형 한글 뒤 모음 입력:', { beforeChar, decomposed, value });
                
                if (decomposed && decomposed.jong !== null) {
                    // 종성이 있는 경우: 종성을 제거하고 새로운 글자 조합 시도
                    if (this.canUseJongAsCho(decomposed.jong)) {
                        const combined = this.combineHangul(decomposed.jong, value, null);
                        if (combined) {
                            const prevWithoutJong = this.combineHangul(decomposed.cho, decomposed.jung, null);
                            const newValue = currentValue.substring(0, selectionStart - 1) + prevWithoutJong + combined + currentValue.substring(selectionEnd);
                            this.inputElement.value = newValue;
                            const newCursorPos = selectionStart + 1;
                            this.inputElement.selectionStart = newCursorPos;
                            this.inputElement.selectionEnd = newCursorPos;
                            const newComposingState = {
                                ...this.composingState,
                                cho: decomposed.jong,
                                jung: value,
                                jong: null,
                                isComposing: true
                            };
                            this.composingState = newComposingState;
                            
                            const inputEvent = new Event('input', { bubbles: true });
                            this.inputElement.dispatchEvent(inputEvent);
                            console.log('[SearchKeyboard] 종성 제거 후 새 글자 조합:', { prevWithoutJong, combined, newValue, composingState: this.composingState });
                            return { handled: true, composingState: newComposingState };
                        } else {
                            // 복합 종성 처리
                            return this.handleComplexJongWithJung(decomposed, value, currentValue, selectionStart, selectionEnd);
                        }
                    } else {
                        // 복합 종성인 경우
                        return this.handleComplexJongWithJung(decomposed, value, currentValue, selectionStart, selectionEnd);
                    }
                } else if (decomposed && decomposed.jong === null) {
                    // 종성이 없는 경우: 복합 모음 조합 시도
                    const complexJung = this.combineJung(decomposed.jung, value);
                    if (complexJung) {
                        const combined = this.combineHangul(decomposed.cho, complexJung, null);
                        if (combined) {
                            const newValue = currentValue.substring(0, selectionStart - 1) + combined + currentValue.substring(selectionEnd);
                            this.inputElement.value = newValue;
                            const newCursorPos = selectionStart;
                            this.inputElement.selectionStart = newCursorPos;
                            this.inputElement.selectionEnd = newCursorPos;
                            const newComposingState = {
                                ...this.composingState,
                                cho: decomposed.cho,
                                jung: complexJung,
                                jong: null,
                                isComposing: true
                            };
                            this.composingState = newComposingState;
                            
                            const inputEvent = new Event('input', { bubbles: true });
                            this.inputElement.dispatchEvent(inputEvent);
                            console.log('[SearchKeyboard] 복합 모음 조합 완료:', { complexJung, combined, newValue, composingState: this.composingState });
                            return { handled: true, composingState: newComposingState };
                        }
                    }
                }
                // 종성이 없거나 조합 실패 시 그냥 삽입
                return {
                    handled: false,
                    composingState: {
                        ...this.composingState,
                        cho: null,
                        jung: value,
                        jong: null,
                        isComposing: false
                    }
                };
            } else {
                // 새로운 중성 시작 (초성이 없으면 그냥 삽입)
                return {
                    handled: false,
                    composingState: {
                        ...this.composingState,
                        cho: null,
                        jung: value,
                        jong: null,
                        isComposing: false
                    }
                };
            }
        }
        
        /**
         * 복합 종성과 모음 입력 처리
         */
        handleComplexJongWithJung(decomposed, value, currentValue, selectionStart, selectionEnd) {
            if (this.isComplexJong(decomposed.jong)) {
                // 복합 종성인 경우: 두 번째 자음을 초성으로 사용하여 새 글자 조합 시도
                const secondJong = this.getSecondJongFromComplex(decomposed.jong);
                if (secondJong && this.canUseJongAsCho(secondJong)) {
                    const combined = this.combineHangul(secondJong, value, null);
                    if (combined) {
                        // 이전 글자에서 복합 종성의 첫 번째 자음만 남기기
                        const firstJong = this.decomposeComplexJong(decomposed.jong);
                        const prevWithFirstJong = this.combineHangul(decomposed.cho, decomposed.jung, firstJong);
                        const newValue = currentValue.substring(0, selectionStart - 1) + prevWithFirstJong + combined + currentValue.substring(selectionEnd);
                        this.inputElement.value = newValue;
                        const newCursorPos = selectionStart + 1;
                        this.inputElement.selectionStart = newCursorPos;
                        this.inputElement.selectionEnd = newCursorPos;
                        const newComposingState = {
                            ...this.composingState,
                            cho: secondJong,
                            jung: value,
                            jong: null,
                            isComposing: true
                        };
                        this.composingState = newComposingState;
                        
                        const inputEvent = new Event('input', { bubbles: true });
                        this.inputElement.dispatchEvent(inputEvent);
                        console.log('[SearchKeyboard] 복합 종성에서 두 번째 자음으로 새 글자 조합:', { prevWithFirstJong, combined, newValue, composingState: this.composingState });
                        return { handled: true, composingState: newComposingState };
                    }
                }
                // 두 번째 자음으로 조합 실패 시: 복합 종성의 첫 번째 자음만 남기고 모음 추가
                const firstJong = this.decomposeComplexJong(decomposed.jong);
                const prevWithFirstJong = this.combineHangul(decomposed.cho, decomposed.jung, firstJong);
                const newValue = currentValue.substring(0, selectionStart - 1) + prevWithFirstJong + value + currentValue.substring(selectionEnd);
                this.inputElement.value = newValue;
                const newCursorPos = selectionStart + 1;
                this.inputElement.selectionStart = newCursorPos;
                this.inputElement.selectionEnd = newCursorPos;
                const newComposingState = { isComposing: false, currentComposition: '', lastChar: '', cho: null, jung: value, jong: null };
                this.composingState = newComposingState;
                
                const inputEvent = new Event('input', { bubbles: true });
                this.inputElement.dispatchEvent(inputEvent);
                console.log('[SearchKeyboard] 복합 종성 제거 후 새 글자 시작:', { prevWithFirstJong, newValue, composingState: this.composingState });
                return { handled: true, composingState: newComposingState };
            } else {
                // 단일 종성인데 초성으로 사용할 수 없거나 조합 실패 시: 종성을 제거하고 모음만 추가
                const prevWithoutJong = this.combineHangul(decomposed.cho, decomposed.jung, null);
                const newValue = currentValue.substring(0, selectionStart - 1) + prevWithoutJong + value + currentValue.substring(selectionEnd);
                this.inputElement.value = newValue;
                const newCursorPos = selectionStart + 1;
                this.inputElement.selectionStart = newCursorPos;
                this.inputElement.selectionEnd = newCursorPos;
                const newComposingState = { isComposing: false, currentComposition: '', lastChar: '', cho: null, jung: value, jong: null };
                this.composingState = newComposingState;
                
                const inputEvent = new Event('input', { bubbles: true });
                this.inputElement.dispatchEvent(inputEvent);
                console.log('[SearchKeyboard] 종성 제거 후 모음만 추가:', { prevWithoutJong, newValue, composingState: this.composingState });
                return { handled: true, composingState: newComposingState };
            }
        }
        
        /**
         * 키 버튼 생성 함수
         */
        createKeyButton(label, value, size, fontSize, isSpecial = false, isSpace = false) {
            const button = document.createElement('button');
            button.textContent = label;
            button.dataset.keyValue = value;
            
            const baseBgColor = isSpecial ? '#e9ecef' : '#ffffff';
            const baseTextColor = isSpecial ? '#dc3545' : '#333333';
            
            const buttonStyle = {
                minWidth: isSpace ? 'auto' : size,
                minHeight: isSpace ? size : size,
                fontSize: fontSize,
                fontWeight: 'bold',
                border: '1px solid #dee2e6',
                borderRadius: '6px',
                backgroundColor: baseBgColor,
                color: baseTextColor,
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: '0 1px 2px rgba(0,0,0,0.1)',
                userSelect: 'none',
                padding: isSpace ? '4px 8px' : '0',
            };
            
            Object.assign(button.style, buttonStyle);
            
            // 호버 효과 개선
            button.addEventListener('mouseenter', function() {
                this.style.backgroundColor = '#667eea';
                this.style.color = '#ffffff';
                this.style.transform = 'scale(1.05)';
                this.style.boxShadow = '0 2px 8px rgba(102, 126, 234, 0.3)';
            });
            button.addEventListener('mouseleave', function() {
                this.style.backgroundColor = baseBgColor;
                this.style.color = baseTextColor;
                this.style.transform = 'scale(1)';
                this.style.boxShadow = '0 1px 2px rgba(0,0,0,0.1)';
            });
            
            // mousedown 이벤트 추가 (터치 디바이스 대응)
            button.addEventListener('mousedown', (e) => {
                e.preventDefault();
            });
            
            // 클릭 이벤트
            button.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.handleKeyboardInput(value);
            });
            
            return button;
        }
        
        /**
         * 키보드 레이아웃 렌더링
         */
        render() {
            if (!this.container) {
                console.warn('[SearchKeyboard] container가 없어서 렌더링할 수 없습니다.');
                return;
            }
            
            this.container.innerHTML = '';
            
            const keyboardLayoutDiv = document.createElement('div');
            keyboardLayoutDiv.style.display = 'grid';
            keyboardLayoutDiv.style.gap = '4px';
            keyboardLayoutDiv.style.width = '100%';
            keyboardLayoutDiv.style.maxWidth = '100%';
            keyboardLayoutDiv.style.overflowX = 'hidden';
            keyboardLayoutDiv.style.overflowY = 'visible';
            keyboardLayoutDiv.style.boxSizing = 'border-box';
            keyboardLayoutDiv.style.paddingBottom = '5px';
            
            // 해상도별 버튼 크기
            const screenWidth = window.innerWidth;
            const screenHeight = window.innerHeight;
            const isPOS = screenWidth === 1024 && screenHeight === 768;
            const isKIOSK = screenWidth === 1024 && screenHeight === 1920;
            const buttonSize = isPOS ? '50px' : isKIOSK ? '70px' : '60px';
            const buttonFontSize = isPOS ? '18px' : isKIOSK ? '26px' : '22px';
            
            if (this.keyboardLayout === 'letters') {
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
                        rowDiv.style.width = '100%';
                        rowDiv.style.maxWidth = '100%';
                        rowDiv.style.flexWrap = 'nowrap';
                        rowDiv.style.boxSizing = 'border-box';
                        
                        row.forEach(key => {
                            // Shift가 눌렸을 때는 대문자, 아니면 소문자
                            const displayKey = this.isShiftPressed ? key.toUpperCase() : key.toLowerCase();
                            const inputKey = this.isShiftPressed ? key.toUpperCase() : key.toLowerCase();
                            const keyButton = this.createKeyButton(displayKey, inputKey, buttonSize, buttonFontSize);
                            rowDiv.appendChild(keyButton);
                        });
                        
                        if (rowIndex === 2) {
                            const backspaceBtn = this.createKeyButton('⌫', 'backspace', buttonSize, buttonFontSize, true);
                            rowDiv.appendChild(backspaceBtn);
                        }
                        
                        keyboardLayoutDiv.appendChild(rowDiv);
                    });
                } else {
                    // 한글 두벌식 레이아웃
                    const doubleConsonantMap = {
                        'ㄱ': 'ㄲ',
                        'ㄷ': 'ㄸ',
                        'ㅂ': 'ㅃ',
                        'ㅅ': 'ㅆ',
                        'ㅈ': 'ㅉ'
                    };
                    
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
                        rowDiv.style.width = '100%';
                        rowDiv.style.maxWidth = '100%';
                        rowDiv.style.flexWrap = 'nowrap';
                        rowDiv.style.boxSizing = 'border-box';
                        
                        row.forEach(key => {
                            const displayKey = this.isShiftPressed && doubleConsonantMap[key] ? doubleConsonantMap[key] : key;
                            const inputKey = this.isShiftPressed && doubleConsonantMap[key] ? doubleConsonantMap[key] : key;
                            const keyButton = this.createKeyButton(displayKey, inputKey, buttonSize, buttonFontSize);
                            rowDiv.appendChild(keyButton);
                        });
                        
                        if (rowIndex === 2) {
                            const backspaceBtn = this.createKeyButton('⌫', 'backspace', buttonSize, buttonFontSize, true);
                            rowDiv.appendChild(backspaceBtn);
                        }
                        
                        keyboardLayoutDiv.appendChild(rowDiv);
                    });
                }
            } else {
                // 숫자 레이아웃 (모든 특수 문자 포함, Shift 기능 없음)
                const numberRows = [
                    // 첫 번째 행: 숫자
                    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
                    // 두 번째 행: 기본 특수 문자
                    ['-', '/', ':', ';', '(', ')', '$', '&', '@', '"'],
                    // 세 번째 행: 추가 특수 문자
                    ['.', ',', '?', '!', "'", '~', '#', '%', '^', '*'],
                    // 네 번째 행: 추가 특수 문자 (한 줄에 모두 배치)
                    ['_', '=', '+', '[', ']', '\\', '|', '`', '{', '}', '<', '>']
                ];
                
                numberRows.forEach((row, rowIndex) => {
                    const rowDiv = document.createElement('div');
                    rowDiv.style.display = 'flex';
                    rowDiv.style.justifyContent = 'center';
                    rowDiv.style.gap = '3px';
                    rowDiv.style.marginBottom = '3px';
                    rowDiv.style.width = '100%';
                    rowDiv.style.flexWrap = 'nowrap';
                    rowDiv.style.boxSizing = 'border-box';
                    
                    row.forEach(key => {
                        const keyButton = this.createKeyButton(key, key, buttonSize, buttonFontSize);
                        rowDiv.appendChild(keyButton);
                    });
                    
                    if (rowIndex === 2) {
                        const backspaceBtn = this.createKeyButton('⌫', 'backspace', buttonSize, buttonFontSize, true);
                        rowDiv.appendChild(backspaceBtn);
                    }
                    
                    keyboardLayoutDiv.appendChild(rowDiv);
                });
            }
            
            // 스페이스바 행
            const spaceRow = document.createElement('div');
            spaceRow.style.display = 'flex';
            spaceRow.style.justifyContent = 'center';
            spaceRow.style.alignItems = 'center';
            spaceRow.style.gap = '4px';
            spaceRow.style.marginTop = '4px';
            
            // Shift 키 버튼 (한글/영문 모드 letters 레이아웃)
            if (this.keyboardLayout === 'letters') {
                const shiftButton = document.createElement('button');
                shiftButton.textContent = '⇧';
                shiftButton.style.cssText = `
                    padding: 6px 12px;
                    border-radius: 6px;
                    border: 2px solid #dee2e6;
                    background-color: ${this.isShiftPressed ? '#667eea' : '#ffffff'};
                    color: ${this.isShiftPressed ? '#ffffff' : '#333333'};
                    cursor: pointer;
                    font-size: ${buttonFontSize};
                    min-width: ${buttonSize};
                    min-height: ${buttonSize};
                    font-weight: bold;
                `;
                shiftButton.addEventListener('click', () => {
                    this.isShiftPressed = !this.isShiftPressed;
                    this.render();
                });
                spaceRow.appendChild(shiftButton);
            }
            
            // 한/영 전환 버튼
            const langButton = document.createElement('button');
            langButton.textContent = this.keyboardMode === 'en' ? '한글' : '영문';
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
                this.isShiftPressed = false;
                this.render();
            });
            
            // 스페이스바
            const spaceBtn = this.createKeyButton('Space', ' ', 'auto', buttonFontSize, false, true);
            spaceBtn.style.flex = '1';
            spaceBtn.style.maxWidth = '400px';
            spaceBtn.style.minHeight = buttonSize;
            
            // 레이아웃 전환 버튼
            const layoutButton = document.createElement('button');
            layoutButton.textContent = this.keyboardLayout === 'letters' ? '123' : (this.keyboardMode === 'ko' ? 'ㄱㄴㄷ' : 'ABC');
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
                this.isShiftPressed = false;
                this.render();
            });
            
            spaceRow.appendChild(langButton);
            spaceRow.appendChild(spaceBtn);
            spaceRow.appendChild(layoutButton);
            keyboardLayoutDiv.appendChild(spaceRow);
            
            this.container.appendChild(keyboardLayoutDiv);
        }
        
        /**
         * 키보드 인스턴스 정리 (메모리 정리)
         * 주의: 모달이 완전히 제거된 후에 호출되어야 합니다.
         */
        destroy() {
            try {
                // 참조만 정리 (DOM은 모달 제거 시 자동으로 정리됨)
                // innerHTML을 비우지 않아서 애니메이션 중에도 키보드가 보이도록 함
                this.container = null;
                this.inputElement = null;
                this.composingState = null;
                
                console.log('[SearchKeyboard] 인스턴스 정리 완료');
            } catch (error) {
                console.error('[SearchKeyboard] 인스턴스 정리 중 오류:', error);
            }
        }
    }
    
    // 전역으로 노출
    window.SearchKeyboard = SearchKeyboard;
    
    console.log('[SearchKeyboard] 모듈 로드 완료');
})();

