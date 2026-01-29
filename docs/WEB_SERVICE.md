# iScan 웹 서비스 문서

## 개요

iScan 웹 서비스는 **SDUI (Server-Driven UI)** 구조를 기반으로 한 키오스크/POS 환경을 위한 웹 인터페이스입니다. 서버에서 UI 구성 정보를 JSON 형태로 제공하고, 클라이언트(브라우저)가 이를 동적으로 렌더링합니다.

## 주요 특징

- **SDUI 구조**: 서버가 UI 구성을 설계하고, 클라이언트는 해석과 렌더링만 담당
- **키오스크/POS 최적화**: 터치스크린 환경에 최적화된 UI 컴포넌트
- **재사용 가능한 컴포넌트**: 비밀번호 입력, 네비게이션 바 등 재사용 가능한 UI 컴포넌트
- **페이지 기반 구조**: 각 페이지가 파일 단위로 분리되어 관리

## 디렉토리 구조

```
app/
├── static/                    # 정적 파일 (HTML, JavaScript)
│   ├── index.html            # 메인 페이지
│   ├── product.html          # 상품 관리 페이지
│   ├── settings.html         # 환경설정 페이지
│   ├── password.html         # 비밀번호 입력 페이지
│   ├── common.js            # 공통 초기화 스크립트
│   └── sdui.js              # SDUI 렌더러 (클라이언트 측)
│
├── services/                  # 비즈니스 로직 서비스
│   ├── web_service.py        # UI 구성 서비스
│   ├── product_service.py    # 상품 관리 서비스
│   └── settings_service.py   # 환경설정 서비스
│
└── main.py                   # FastAPI 라우터 및 API 엔드포인트
```

## 페이지 구성

### 1. 메인 페이지 (`/`)

**기능:**
- 아이콘 그리드 형태로 주요 페이지 선택
- 상단에 "더 효율적인 운영의 시작, 기산전자의 iScan 과 함께하세요." 문구 표시
- 네비게이션 바 높이를 고려한 중앙 정렬 레이아웃 (POS/KIOSK 해상도 모두 지원)

**구성:**
- 상품 관리 아이콘 카드
- 환경설정 아이콘 카드 (비밀번호 입력 페이지로 이동)

**레이아웃:**
- 헤더와 네비게이션 바 높이를 자동 계산하여 컨텐츠를 중앙에 배치
- 해상도별 패딩 자동 조정 (POS: 20px, KIOSK: 40px 20px)
- 브라우저 크기 변경 시 자동 재조정

**네비게이션 바:**
- ◀ 뒤로 가기
- ▶ 앞으로 가기
- ↻ 새로고침
- | (구분선)
- ↑ 위로 스크롤
- ↓ 아래로 스크롤
- ✕ 브라우저 종료 (메인 페이지에서만 표시)

### 2. 상품 관리 페이지 (`/product`)

**기능:**
- 상품 정보 관리 (향후 구현 예정)

**네비게이션 바:**
- ◀ 뒤로 가기
- ▶ 앞으로 가기
- ↻ 새로고침
- 🏠 홈으로
- | (구분선)
- ↑ 위로 스크롤
- ↓ 아래로 스크롤

### 3. 환경설정 페이지 (`/settings`)

**기능:**
- 시스템 설정 관리 (향후 구현 예정)
- 비밀번호 인증 후 접근 가능

**접근 방법:**
1. 메인 페이지에서 환경설정 아이콘 클릭
2. 비밀번호 입력 페이지로 이동
3. 비밀번호 "7890" 입력
4. 환경설정 페이지로 이동

### 4. 비밀번호 입력 페이지 (`/password`)

**기능:**
- 관리자 인증을 위한 비밀번호 입력
- 숫자 키패드를 통한 비밀번호 입력
- 재사용 가능한 구조 (리다이렉트 URL 파라미터 지원)

**사용 방법:**
```
/password?redirect=/settings
```

**컴포넌트:**
- 비밀번호 입력 필드 (마스킹 표시)
- 숫자 키패드 (0-9, 지우기, 전체 삭제)
- 입력완료 버튼

**기본 비밀번호:**
- `7890`

## SDUI 컴포넌트

### 지원하는 컴포넌트 타입

1. **container** - 컨테이너 (수직/수평 레이아웃)
2. **header** - 헤더 (제목, 부제목)
3. **card** - 카드 컨테이너
4. **icon_card** - 아이콘 카드 (메인 페이지의 페이지 선택 카드)
5. **button** - 버튼
6. **text** - 텍스트
7. **table** - 테이블
8. **navigation_bar** - 하단 네비게이션 바
9. **password_input** - 비밀번호 입력 필드
10. **numpad** - 숫자 키패드

### 액션 타입

1. **navigate** - 페이지 이동
2. **api_call** - API 호출
3. **go_back** - 뒤로 가기
4. **go_forward** - 앞으로 가기
5. **refresh** - 새로고침
6. **go_home** - 홈으로 이동
7. **scroll_up** - 위로 스크롤
8. **scroll_down** - 아래로 스크롤
9. **close** - 브라우저 종료 (메인 페이지에서만)
10. **verify_password** - 비밀번호 검증

## API 엔드포인트

### 페이지 라우팅

#### `GET /`
메인 페이지를 반환합니다.

**응답:**
- `200`: HTML 파일 반환 (`app/static/index.html`)
- `500`: 파일을 찾을 수 없을 경우 JSON 응답

#### `GET /web`
웹 인터페이스 페이지를 반환합니다. (`/`와 동일)

#### `GET /product`
상품 관리 페이지를 반환합니다.

**응답:**
- `200`: HTML 파일 반환 (`app/static/product.html`)
- `500`: 파일을 찾을 수 없을 경우 에러 반환

#### `GET /password`
비밀번호 입력 페이지를 반환합니다.

**응답:**
- `200`: HTML 파일 반환 (`app/static/password.html`)
- `500`: 파일을 찾을 수 없을 경우 에러 반환

#### `GET /settings`
환경설정 페이지를 반환합니다.

**응답:**
- `200`: HTML 파일 반환 (`app/static/settings.html`)
- `500`: 파일을 찾을 수 없을 경우 에러 반환

#### `GET /favicon.ico`
파비콘을 반환합니다.

**응답:**
- `200`: 파비콘 파일 반환 (`app/static/favicon.ico`)
- `404`: 파일이 없을 경우 조용히 404 반환 (에러 로그 없음)

**참고:**
- 모든 정적 파일은 절대 경로를 사용하여 작업 디렉토리와 무관하게 동작합니다.
- 파일이 없을 경우 적절한 에러 응답을 반환합니다.

### UI 구성 API

#### `GET /api/ui/config`

UI 구성 정보를 JSON으로 반환합니다.

**파라미터:**
- `page` (query, optional): 페이지 이름 ("main", "product", "settings", "password")
- `redirect` (query, optional): 비밀번호 페이지의 리다이렉트 URL (기본값: "/settings")

**응답 예시:**
```json
{
  "title": "iScan 관리 페이지",
  "version": "1.0.0",
  "layout": {
    "type": "container",
    "direction": "vertical",
    "children": [...]
  },
  "theme": {...}
}
```

### 인증 API

#### `POST /api/auth/verify-password`

비밀번호를 검증합니다.

**요청 본문:**
```json
{
  "password": "7890",
  "redirect_url": "/settings"
}
```

**응답 (성공):**
```json
{
  "success": true,
  "message": "비밀번호가 일치합니다.",
  "redirect_url": "/settings"
}
```

**응답 (실패):**
```json
{
  "detail": "비밀번호가 일치하지 않습니다."
}
```

**상태 코드:**
- `200`: 성공
- `401`: 비밀번호 불일치
- `400`: 요청 오류

### UI 액션 API

#### `POST /api/ui/action`

UI 액션을 처리합니다.

**요청 본문:**
```json
{
  "action_type": "api_call",
  "payload": {...}
}
```

### EdgeMan API

#### `POST /api/edgeman/terminate-browser`

브라우저 종료 요청을 EdgeMan 서버로 전달합니다.

**기능:**
- 웹 브라우저가 실행되는 PC의 HTTP 서버로 브라우저 종료 요청을 전달
- 클라이언트 IP를 기반으로 EdgeMan 서버 주소를 자동 구성
- X-Forwarded-For 헤더 지원 (프록시/로드밸런서 환경)

**요청 본문:**
```json
{}
```

**응답 (성공):**
```json
{
  "success": true,
  "message": "브라우저 종료 요청이 EdgeMan 서버로 전달되었습니다.",
  "edgeman_response": {...}
}
```

**응답 (EdgeMan 서버 통신 실패 - 성공으로 처리):**
```json
{
  "success": true,
  "message": "브라우저 종료 요청이 처리되었습니다.",
  "warning": "EdgeMan 서버 통신 실패 (무시됨): ..."
}
```

**응답 (EdgeMan 서버 오류):**
```json
{
  "success": false,
  "message": "EdgeMan 서버에서 오류가 발생했습니다 (HTTP 500)",
  "error": "..."
}
```

**상태 코드:**
- `200`: 성공 (EdgeMan 서버 통신 성공 또는 실패 시에도 성공으로 처리)
- `500`: 서버 내부 오류

**동작 방식:**
1. 클라이언트 IP 주소 추출 (X-Forwarded-For 헤더 우선)
2. EdgeMan 서버 URL 구성: `http://{client_ip}:18081/api/edgeman/terminate-browser`
3. EdgeMan 서버로 POST 요청 전달 (타임아웃: 5초)
4. EdgeMan 서버 응답에 따라 결과 반환

**참고:**
- 기본 포트는 18081 (향후 필요 시 config.json에서 설정 가능하도록 개선 가능)
- EdgeMan 서버와 통신할 수 없는 경우에도 성공 처리 (브라우저가 다른 서버에서 실행될 수 있음)

## 서비스 클래스

### WebUIService

**위치:** `app/services/web_service.py`

**주요 메서드:**

- `generate_ui_config(page: str = "main")` - 페이지별 UI 구성 생성
- `_generate_main_page_config()` - 메인 페이지 UI 구성
- `_generate_product_page_config()` - 상품 관리 페이지 UI 구성
- `_generate_settings_page_config()` - 환경설정 페이지 UI 구성
- `_generate_password_page_config(redirect_url: str)` - 비밀번호 입력 페이지 UI 구성
- `_get_navigation_bar(is_main_page: bool)` - 네비게이션 바 구성

### ProductService

**위치:** `app/services/product_service.py`

**주요 메서드:**
- `get_product_list()` - 상품 목록 조회
- `get_product_detail(product_id)` - 상품 상세 정보 조회
- `create_product(product_data)` - 상품 생성
- `update_product(product_id, product_data)` - 상품 수정
- `delete_product(product_id)` - 상품 삭제

### SettingsService

**위치:** `app/services/settings_service.py`

**주요 메서드:**
- `get_settings()` - 설정 조회
- `update_settings(settings)` - 설정 업데이트

## 클라이언트 측 렌더러

### SDUIRenderer 클래스

**위치:** `app/static/sdui.js`

**주요 메서드:**

#### 렌더링 메서드
- `render(container)` - UI 렌더링 시작
- `renderElement(element, parent)` - 요소 렌더링 (재귀적)
- `renderContainer()` - 컨테이너 렌더링
- `renderHeader()` - 헤더 렌더링 (해상도별 크기 자동 조정)
- `renderCard()` - 카드 렌더링
- `renderIconCard()` - 아이콘 카드 렌더링
- `renderButton()` - 버튼 렌더링 (해상도별 크기 자동 조정)
- `renderText()` - 텍스트 렌더링
- `renderTable()` - 테이블 렌더링
- `renderNavigationBar()` - 네비게이션 바 렌더링 (메인 페이지 중앙 정렬 포함)
- `renderPasswordInput()` - 비밀번호 입력 필드 렌더링 (해상도별 크기 자동 조정)
- `renderNumpad()` - 숫자 키패드 렌더링 (해상도별 크기 자동 조정)

#### 유틸리티 메서드
- `getResolutionType()` - 해상도 타입 판별 (KIOSK/POS/기타) - 정적 메서드

#### 액션 처리 메서드
- `handleAction(action)` - 액션 처리 라우터
- `handleNavigate()` - 페이지 이동
- `handleGoBack()` - 뒤로 가기
- `handleGoForward()` - 앞으로 가기
- `handleRefresh()` - 새로고침
- `handleGoHome()` - 홈으로 이동
- `handleScrollUp()` - 위로 스크롤
- `handleScrollDown()` - 아래로 스크롤
- `handleClose()` - 브라우저 종료
- `handleVerifyPassword()` - 비밀번호 검증
- `handleApiCall()` - API 호출
- `handleNumpadInput()` - 키패드 입력 처리
- `showErrorPopup(message)` - 에러 팝업 표시

## 네비게이션 바 구성

### 메인 페이지 네비게이션 바

- 홈 버튼: 숨김 (이미 홈이므로)
- X 버튼: 표시 (브라우저 종료)

### 다른 페이지 네비게이션 바

- 홈 버튼: 표시
- X 버튼: 숨김

## 비밀번호 입력 페이지 재사용

다른 페이지에서도 비밀번호 입력 페이지를 사용할 수 있습니다:

```javascript
// 예시: 상품 관리 페이지에 비밀번호 보호 추가
{
  "type": "icon_card",
  "title": "상품 관리",
  "action": {
    "type": "navigate",
    "url": "/password?redirect=/product"
  }
}
```

## 키오스크/POS 환경 최적화

### 해상도별 반응형 디자인

시스템은 두 가지 주요 해상도를 지원합니다:

#### 1. POS 해상도 (1024x768)
- 작은 화면에 최적화된 컴포넌트 크기
- 네비게이션 바: 패딩 `8px 12px`, 버튼 크기 `50x50px`, 폰트 크기 `20px`
- 헤더: 패딩 `12px`, 제목 폰트 `24px`, 부제목 폰트 `16px`
- 메인 페이지 컨테이너: 패딩 `20px`, 중앙 정렬
- 넘버패드: 너비 `360px`, 버튼 폰트 `26px`, 간격 `8px`
- 입력완료 버튼: 폰트 `18px`, 최소 높이 `42px`, 패딩 `10px 20px`
- 비밀번호 입력 필드: 폰트 `14px`, 최소 높이 `40px`, 패딩 `8px`

#### 2. KIOSK 해상도 (1024x1920)
- 큰 화면에 최적화된 컴포넌트 크기
- 네비게이션 바: 패딩 `15px 20px`, 버튼 크기 `70x70px`, 폰트 크기 `28px`
- 헤더: 패딩 `30px`, 제목 폰트 `28px`, 부제목 폰트 `18px`
- 메인 페이지 컨테이너: 패딩 `40px 20px`, 중앙 정렬
- 넘버패드: 너비 `500px`, 버튼 폰트 `40px`, 간격 `20px`
- 입력완료 버튼: 폰트 `24px`, 최소 높이 `50px`, 패딩 `20px 30px`
- 비밀번호 입력 필드: 폰트 `24px`, 최소 높이 `55px`, 패딩 `20px`

#### 3. 기타 해상도
- 화면 크기에 비례하여 자동 조정
- 최소/최대 크기 제한 적용
- 범위 기반 해상도 감지 (KIOSK: 세로 ≥ 1800px, 가로 1000~1200px / POS: 세로 700~900px, 가로 1000~1200px)

### 동적 크기 조정

모든 컴포넌트는 `window.addEventListener('resize')`를 통해 브라우저 크기 변경 시 자동으로 크기를 조정합니다:
- 디바운싱 적용 (100ms 지연)
- `requestAnimationFrame`을 통한 부드러운 전환
- CSS `!important`를 통한 스타일 우선순위 보장

### 메인 페이지 중앙 정렬

메인 페이지의 컨테이너는 네비게이션 바와 헤더 높이를 자동으로 계산하여 컨텐츠를 중앙에 배치합니다:

**동작 방식:**
1. 헤더 높이 계산 (`header.offsetHeight`)
2. 네비게이션 바 높이 계산 (`navBar.offsetHeight`)
3. 사용 가능한 높이 계산 (`window.innerHeight - headerHeight - navBarHeight`)
4. 컨테이너에 `minHeight`와 `maxHeight` 설정
5. `justifyContent: center` 적용하여 중앙 정렬

**해상도별 패딩:**
- POS (1024x768): `padding: 20px`
- KIOSK (1024x1920): `padding: 40px 20px`
- 기타: `padding: 30px 20px`

**렌더링 타이밍:**
- 즉시 적용
- 100ms 후 재적용 (헤더 높이 정확도 향상)
- 300ms 후 재적용 (네비게이션 바 높이 정확도 향상)
- 리사이즈 이벤트 시 자동 재적용

### 터치 최적화
- 큰 버튼 크기 (POS: 50x50px, KIOSK: 70x70px)
- 터치 피드백 (scale 애니메이션)
- 명확한 시각적 피드백
- 최소 터치 영역 보장

### 키보드/마우스 없음 대응
- 모든 기능을 터치로 조작 가능
- 하단 네비게이션 바로 모든 네비게이션 제공
- 숫자 키패드로 비밀번호 입력
- 스크롤 버튼으로 페이지 탐색

## UI 컴포넌트 상세

### 에러 팝업 (`showErrorPopup`)

에러 메시지를 모달 팝업으로 표시합니다.

**특징:**
- 중복 팝업 방지 (전역 플래그 사용)
- 이벤트 버블링 방지 (`stopPropagation`)
- 부드러운 애니메이션 (fadeIn/fadeOut)
- 오버레이 클릭 또는 확인 버튼으로 닫기
- 팝업 박스 클릭 시 이벤트 전파 차단

**사용 예시:**
```javascript
this.showErrorPopup('비밀번호가 일치하지 않습니다.');
```

### 네비게이션 바 동적 크기 조정

네비게이션 바는 해상도에 따라 자동으로 크기가 조정됩니다:

- **POS 해상도 (1024x768)**: 작은 크기로 최적화
- **KIOSK 해상도 (1024x1920)**: 큰 크기로 최적화
- **기타 해상도**: 비례적으로 조정

### 스크롤바 처리

- 기본적으로 스크롤바 숨김 (`overflow: hidden`)
- 컨텐츠가 뷰포트를 초과할 때만 스크롤바 표시 (`overflowY: auto`)
- 네비게이션 바 높이를 고려한 컨테이너 높이 계산
- POS 해상도: `calc(100vh - 68px)`
- KIOSK 해상도: `calc(100vh - 102px)`

## 보안

### 비밀번호 관리
- 기본 비밀번호: `7890`
- 향후 설정 파일 또는 데이터베이스에서 관리 가능
- 비밀번호는 서버 측에서 검증
- 클라이언트 측에서는 마스킹 표시

### 접근 제어
- 환경설정 페이지는 비밀번호 인증 후 접근
- 다른 페이지에도 동일한 방식으로 보호 추가 가능
- 비밀번호 입력 페이지는 재사용 가능한 구조

### IP 화이트리스트
- IP 기반 접근 제어 지원
- `config.json`의 `security.allowed_ips`에서 허용된 IP 목록 관리
- `X-Forwarded-For` 헤더 지원 (프록시/로드밸런서 환경)
- localhost (`127.0.0.1`, `::1`)는 항상 허용
- IP 화이트리스트 비활성화 옵션 제공 (`security.enable_ip_whitelist`)

## 파일 경로 처리

### 정적 파일 경로
- 모든 정적 파일은 절대 경로를 사용하여 작업 디렉토리와 무관하게 동작
- `app/main.py`의 `STATIC_DIR` 변수를 통해 정적 파일 디렉토리 경로 관리
- 서버 시작 시 정적 파일 디렉토리 존재 여부 확인 및 로그 출력

### 설정 파일 경로
- `config.json` 파일은 `app/core/config.py`의 `CONFIG_FILE` 상수를 통해 절대 경로로 관리
- 작업 디렉토리와 무관하게 프로젝트 루트의 `config.json`을 찾음

### 에러 처리
- 파일을 찾을 수 없을 경우 적절한 HTTP 상태 코드 반환
- `favicon.ico` 같은 브라우저 자동 요청은 조용히 404 반환 (에러 로그 없음)
- 중요한 에러는 상세한 로그와 함께 처리

## 구현된 기능

### 완료된 기능
1. ✅ **SDUI 구조 구현**
   - 서버에서 UI 구성 JSON 제공
   - 클라이언트 측 동적 렌더링

2. ✅ **메인 페이지**
   - 아이콘 카드 형태의 페이지 선택
   - 상단 문구 표시
   - 네비게이션 바 높이를 고려한 중앙 정렬 레이아웃
   - 해상도별 자동 패딩 조정

3. ✅ **비밀번호 입력 페이지**
   - 숫자 키패드 입력
   - 비밀번호 검증 API
   - 재사용 가능한 구조

4. ✅ **네비게이션 바**
   - 하단 고정 위치
   - 페이지별 버튼 표시/숨김
   - 해상도별 크기 조정

5. ✅ **해상도별 반응형 디자인**
   - POS 해상도 (1024x768) 최적화
   - KIOSK 해상도 (1024x1920) 최적화
   - 동적 크기 조정

6. ✅ **에러 팝업 시스템**
   - 모달 팝업 표시
   - 중복 팝업 방지
   - 부드러운 애니메이션

7. ✅ **스크롤바 처리**
   - 네비게이션 바 높이 고려
   - 필요 시에만 스크롤바 표시

8. ✅ **메인 페이지 중앙 정렬**
   - 헤더와 네비게이션 바 높이 자동 계산
   - 컨텐츠 중앙 배치 (POS/KIOSK 해상도 모두 지원)
   - 브라우저 크기 변경 시 자동 재조정
   - 해상도별 패딩 자동 조정

9. ✅ **파일 경로 처리 개선**
   - 정적 파일 절대 경로 사용 (작업 디렉토리 무관)
   - 설정 파일 절대 경로 사용 (`CONFIG_FILE` 상수)
   - 서버 시작 시 정적 파일 디렉토리 존재 여부 확인

10. ✅ **에러 처리 개선**
    - 파일을 찾을 수 없을 경우 적절한 HTTP 상태 코드 반환
    - `favicon.ico` 같은 브라우저 자동 요청은 조용히 처리
    - IP 화이트리스트 미들웨어 에러 처리 강화
    - 상세한 에러 로깅 및 traceback 출력

11. ✅ **EdgeMan 브라우저 종료 API**
    - `/api/edgeman/terminate-browser` 엔드포인트로 POST 하여 EdgeMan 이 브라우저를 종료하게 함
    - 클라이언트 IP 기반 EdgeMan 서버 주소 자동 구성
    - X-Forwarded-For 헤더 지원 (프록시/로드밸런서 환경)
    - EdgeMan 서버 통신 실패 시에도 안전하게 처리

## 향후 확장 계획

1. **비밀번호 관리**
   - 설정 파일 또는 데이터베이스에서 비밀번호 관리
   - 비밀번호 변경 기능
   - 비밀번호 복잡도 검증

2. **상품 관리 기능**
   - 상품 목록 조회
   - 상품 등록/수정/삭제
   - 상품 검색
   - 상품 이미지 업로드

3. **환경설정 기능**
   - 시스템 설정 조회/수정
   - 설정 백업/복원
   - 네트워크 설정
   - 디바이스 설정

4. **추가 컴포넌트**
   - 폼 입력 컴포넌트
   - 파일 업로드 컴포넌트
   - 모달 다이얼로그
   - 드롭다운 메뉴
   - 체크박스/라디오 버튼

5. **성능 최적화**
   - 이미지 지연 로딩
   - 컴포넌트 캐싱
   - API 응답 최적화

## 참고 사항

- 모든 페이지는 SDUI 구조를 따릅니다
- UI 구성은 서버에서 JSON으로 제공되며, 클라이언트가 동적으로 렌더링합니다
- 새로운 페이지 추가 시 `web_service.py`에 UI 구성 함수를 추가하고, `main.py`에 라우터를 추가합니다
- 새로운 컴포넌트 추가 시 `sdui.js`에 렌더러를 추가합니다

