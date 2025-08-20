# AI 블록코딩 API 명세서 v2

## 1. 개요

### 1.1 시스템 아키텍처
- **Frontend**: React/Vue/Vanilla JS (블록 UI, 코드 미리보기, 데이터 시각화)
- **Backend**: Flask (코드 생성, 실행, 로그 스트리밍)
- **Storage**: 사용자별 워크스페이스 (사용자 ID 기반 격리)
- **CORS**: 활성화됨 (Cross-Origin 요청 지원)

### 1.2 핵심 개념
- **사용자 ID**: 각 사용자를 식별하는 고유 문자열 (영문, 숫자, _, - 만 허용)
- **블록(Block)**: AI 파이프라인의 각 단계를 나타내는 UI 컴포넌트
- **스니펫(Snippet)**: 블록 설정에 따라 생성되는 Python 코드 조각
- **워크스페이스(Workspace)**: 사용자별 작업 공간 (코드, 데이터, 로그)
- **스테이지(Stage)**: 파이프라인 단계 (pre/model/train/eval)

---

## 2. 사용자 ID 관리

### 2.1 사용자 ID 규칙
- **형식**: 영문자, 숫자, 밑줄(_), 하이픈(-) 만 허용
- **길이**: 1~50자
- **예시**: `hong_gildong`, `user123`, `ai-researcher-01`
- **금지**: 특수문자, 공백, 한글 등 (`admin`, `root` 같은 시스템 예약어 권장하지 않음)

### 2.2 사용자 ID 전달 방식

#### A. 웹 UI 접근 (권장)
```
GET /app?user_id=hong_gildong
```

#### B. API 요청 시 포함
모든 API 요청에 사용자 ID를 다음 방식 중 하나로 포함:

1. **POST 요청**: form-data 또는 JSON body에 포함
```javascript
// Form-data 방식
const formData = new FormData();
formData.append('user_id', 'hong_gildong');
formData.append('stage', 'pre');
// ... 기타 파라미터

fetch('/convert', {
  method: 'POST',
  body: formData
});

// JSON 방식
fetch('/run/pre', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: 'hong_gildong'
  })
});
```

2. **GET 요청**: 쿼리 파라미터로 포함
```javascript
fetch('/logs/stream?stage=train&user_id=hong_gildong');
fetch('/download/pre?user_id=hong_gildong');
```

### 2.3 워크스페이스 구조
```
workspace/
├── hong_gildong/           # 사용자 ID 기반 폴더
│   ├── preprocessing.py
│   ├── model.py
│   ├── training.py
│   ├── evaluation.py
│   ├── inputs_pre.json
│   ├── inputs_model.json
│   ├── inputs_train.json
│   ├── inputs_eval.json
│   ├── data/
│   │   └── dataset.pt
│   └── artifacts/
│       ├── best_model.pth
│       └── training_history.json
└── user123/               # 다른 사용자
    ├── preprocessing.py
    └── ...
```

### 2.4 에러 처리
- **사용자 ID 누락**: `400 Bad Request - user_id가 필요합니다.`
- **잘못된 형식**: `400 Bad Request - 영문, 숫자, _, - 만 사용 가능합니다.`
- **파일 없음**: `404 Not Found - preprocessing.py 파일이 없습니다. 먼저 코드를 생성해주세요.`

---

## 3. API 엔드포인트

### 3.1 페이지 렌더링

#### GET `/`
**설명**: 루트 접근 시 `/app`으로 리다이렉트

#### GET `/app`
**설명**: 메인 애플리케이션 페이지  
**쿼리 파라미터**:
- `user_id` (선택): 사용자 ID. 없으면 'anonymous'

**응답**: HTML (index.html)  
**템플릿 변수**:
```python
{
    "options": ["mnist_train.csv", ...],
    "form_state": {...},           # 해당 사용자의 저장된 폼 상태
    "current_user_id": "hong_gildong",  # 현재 사용자 ID
    "snippet_pre": "...",          # 해당 사용자의 전처리 코드
    "snippet_model": "...",        # 해당 사용자의 모델 코드
    "snippet_train": "...",        # 해당 사용자의 학습 코드
    "snippet_eval": "..."          # 해당 사용자의 평가 코드
}
```

---

### 3.2 코드 생성 및 변환

#### POST `/convert`
**설명**: 블록 설정을 Python 코드로 변환  
**Content-Type**: `multipart/form-data` 또는 `application/json`
**요청 본문**:

##### 필수 파라미터
| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| user_id | string | ✓ | 사용자 식별자 (1-50자, 영문/숫자/_/- 만) |
| stage | string | ✓ | 변환 대상: "pre", "model", "train", "eval", "all" |

##### 전처리 블록 파라미터 (stage=pre 또는 all)
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| dataset | string | - | 훈련 데이터 CSV 파일명 |
| is_test | string | "false" | 테스트 데이터 사용 여부 |
| testdataset | string | - | 테스트 데이터 CSV 파일명 |
| a | number | 80 | 훈련 데이터 비율 (%) |
| drop_na | checkbox | - | 결측치 제거 여부 |
| drop_bad | checkbox | - | 잘못된 라벨 제거 여부 |
| min_label | number | 0 | 최소 라벨값 |
| max_label | number | 9 | 최대 라벨값 |
| split_xy | checkbox | - | X/y 분리 여부 |
| resize_n | number | 28 | 이미지 리사이즈 크기 |
| augment_method | string | - | 증강 방법: "rotate", "hflip", "vflip", "translate" |
| augment_param | number | - | 증강 파라미터 |
| normalize | string | - | 정규화 방법: "0-1", "-1-1" |

##### 모델 설계 블록 파라미터 (stage=model 또는 all)
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| input_w | number | 28 | 입력 이미지 너비 |
| input_h | number | 28 | 입력 이미지 높이 |
| input_c | number | 1 | 입력 채널 수 |
| conv1_filters | number | 32 | Conv1 필터 수 |
| conv1_kernel | number | 3 | Conv1 커널 크기 |
| conv1_padding | string | "same" | Conv1 패딩 |
| conv1_activation | string | "relu" | Conv1 활성함수 |
| pool1_type | string | "max" | Pool1 종류 |
| pool1_size | number | 2 | Pool1 크기 |
| pool1_stride | number | 2 | Pool1 스트라이드 |
| use_conv2 | checkbox | - | Conv2 사용 여부 |
| conv2_filters | number | 64 | Conv2 필터 수 |
| conv2_kernel | number | 3 | Conv2 커널 크기 |
| conv2_activation | string | "relu" | Conv2 활성함수 |
| use_dropout | checkbox | - | 드롭아웃 사용 여부 |
| dropout_p | number | 0.25 | 드롭아웃 비율 |
| dense_units | number | 128 | Dense 레이어 유닛 수 |
| dense_activation | string | "relu" | Dense 활성함수 |
| num_classes | number | 10 | 출력 클래스 수 |

##### 학습 블록 파라미터 (stage=train 또는 all)
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| loss_method | string | "CrossEntropy" | 손실함수 |
| optimizer_method | string | "Adam" | 옵티마이저 |
| learning_rate | number | 0.001 | 학습률 |
| epochs | number | 10 | 에폭 수 |
| batch_size | number | 64 | 배치 크기 |
| patience | number | 3 | Early stopping patience |

##### 평가 블록 파라미터 (stage=eval 또는 all)
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| metrics | array | ["accuracy"] | 평가 메트릭 |
| average | string | "macro" | 평균 방식 |
| topk_k | number | 3 | Top-K의 K값 |
| show_classification_report | checkbox | - | 분류 리포트 출력 |
| show_confusion_matrix | checkbox | - | 혼동 행렬 출력 |
| cm_normalize | checkbox | - | 혼동 행렬 정규화 |
| viz_samples | number | 10 | 시각화할 예측 샘플 수 |
| viz_mis | number | 5 | 시각화할 오분류 샘플 수 |
| eval_batch | number | 128 | 평가 배치 크기 |
| num_classes | number | 10 | 클래스 수 |
| class_names | string | - | 클래스 이름 (쉼표 구분) |
| force_cpu | checkbox | - | CPU 강제 사용 |

**응답**: 
- **Content-Type**: `text/plain; charset=utf-8`
- **성공 시 (200)**: 생성된 Python 코드 (텍스트)
- **실패 시 (400/500)**: 오류 메시지 (텍스트)

**요청 예시**:
```javascript
const formData = new FormData();
formData.append('user_id', 'hong_gildong');
formData.append('stage', 'pre');
formData.append('dataset', 'mnist_train.csv');
formData.append('drop_na', 'on');
formData.append('normalize', '0-1');

fetch('/convert', {
  method: 'POST',
  body: formData
})
.then(response => response.text())
.then(code => {
  console.log('생성된 코드:', code);
});
```

---

### 3.3 코드 실행 및 로그

#### POST `/run/<stage>`
**설명**: 특정 스테이지 코드 실행  
**경로 파라미터**: 
- `stage`: "pre", "model", "train", "eval"

**요청 본문**:
| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| user_id | string | ✓ | 사용자 식별자 |

**응답**:
```json
// 성공
{
    "ok": true,
    "pid": 12345
}

// 실패
{
    "error": "preprocessing.py 파일이 없습니다. 먼저 코드를 생성해주세요."
}
```

**요청 예시**:
```javascript
const formData = new FormData();
formData.append('user_id', 'hong_gildong');

fetch('/run/pre', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(result => {
  if (result.ok) {
    console.log('실행 시작됨, PID:', result.pid);
  } else {
    console.error('실행 실패:', result.error);
  }
});
```

#### GET `/logs/stream`
**설명**: Server-Sent Events로 실행 로그 스트리밍  
**쿼리 파라미터**:
- `user_id` (필수): 사용자 식별자
- `stage` (선택): 로그를 볼 스테이지 (기본값: "train")

**응답**: `text/event-stream`
```
data: [pre][2025-01-01 12:00:00] Processing started...
data: [pre][2025-01-01 12:00:01] Loading data...
```

**요청 예시**:
```javascript
const eventSource = new EventSource('/logs/stream?user_id=hong_gildong&stage=pre');

eventSource.onmessage = function(event) {
  console.log('로그:', event.data);
  document.getElementById('log-output').textContent += event.data + '\n';
};

eventSource.onerror = function(event) {
  console.error('SSE 연결 오류:', event);
  eventSource.close();
};
```

---

### 3.4 데이터 정보 조회

#### GET `/data-info`
**설명**: CSV 데이터셋 정보 조회  
**쿼리 파라미터**:
- `file` (필수): CSV 파일명
- `type` (선택): 정보 유형 ("shape", "structure", "sample", "images")
- `n` (선택): 샘플/이미지 개수 (type=sample/images일 때)

**참고**: 이 API는 사용자 ID가 불필요합니다 (공용 데이터셋 조회)

---

### 3.5 파일 다운로드

#### GET `/download/<stage>`
**설명**: 생성된 코드 파일 다운로드  
**경로 파라미터**:
- `stage`: "pre", "model", "train", "eval", "all"

**쿼리 파라미터**:
- `user_id` (필수): 사용자 식별자

**응답**: 
- stage="pre": preprocessing.py 파일
- stage="model": model.py 파일
- stage="train": training.py 파일
- stage="eval": evaluation.py 파일
- stage="all": workspace_<user_id>_<timestamp>.zip 파일

**요청 예시**:
```javascript
// 단일 파일 다운로드
window.open('/download/pre?user_id=hong_gildong', '_blank');

// 전체 ZIP 다운로드
window.open('/download/all?user_id=hong_gildong', '_blank');
```

---

## 4. 프론트엔드 통합 가이드

### 4.1 기본 흐름

1. **사용자 ID 입력**: 사용자가 고유 ID 입력
2. **페이지 로드**: `/app?user_id=<user_id>`로 접근하여 해당 사용자의 저장된 상태 로드
3. **블록 설정**: UI에서 AI 파이프라인 단계별 설정
4. **코드 생성**: `/convert` API로 Python 코드 생성 및 저장
5. **코드 실행**: `/run/<stage>` API로 단계별 실행
6. **로그 모니터링**: `/logs/stream` SSE로 실시간 로그 확인
7. **결과 다운로드**: `/download/<stage>` API로 코드 파일 다운로드

### 4.2 상태 관리

```javascript
class AIBlockCodingClient {
  constructor(userId) {
    this.userId = userId;
    this.baseURL = 'http://127.0.0.1:9000';
  }

  // 코드 변환
  async convertCode(stage, blockData) {
    const formData = new FormData();
    formData.append('user_id', this.userId);
    formData.append('stage', stage);
    
    // 블록 데이터 추가
    Object.entries(blockData).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        value.forEach(v => formData.append(key, v));
      } else {
        formData.append(key, value);
      }
    });

    const response = await fetch(`${this.baseURL}/convert`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    return await response.text();
  }

  // 코드 실행
  async runStage(stage) {
    const formData = new FormData();
    formData.append('user_id', this.userId);

    const response = await fetch(`${this.baseURL}/run/${stage}`, {
      method: 'POST',
      body: formData
    });

    return await response.json();
  }

  // 로그 스트리밍
  streamLogs(stage, onMessage, onError) {
    const eventSource = new EventSource(
      `${this.baseURL}/logs/stream?user_id=${encodeURIComponent(this.userId)}&stage=${stage}`
    );

    eventSource.onmessage = onMessage;
    eventSource.onerror = onError;

    return eventSource;
  }

  // 파일 다운로드
  downloadFile(stage) {
    const url = `${this.baseURL}/download/${stage}?user_id=${encodeURIComponent(this.userId)}`;
    window.open(url, '_blank');
  }
}

// 사용 예시
const client = new AIBlockCodingClient('hong_gildong');

// 전처리 코드 생성
const code = await client.convertCode('pre', {
  dataset: 'mnist_train.csv',
  drop_na: true,
  normalize: '0-1'
});

// 전처리 실행
const result = await client.runStage('pre');

// 로그 모니터링
const eventSource = client.streamLogs('pre', 
  (event) => console.log('로그:', event.data),
  (error) => console.error('SSE 오류:', error)
);
```

### 4.3 에러 처리

```javascript
// 사용자 ID 검증
function validateUserId(userId) {
  if (!userId || userId.trim() === '') {
    throw new Error('사용자 ID가 필요합니다.');
  }
  
  if (!/^[a-zA-Z0-9_-]+$/.test(userId)) {
    throw new Error('사용자 ID는 영문, 숫자, _, - 만 사용 가능합니다.');
  }
  
  if (userId.length > 50) {
    throw new Error('사용자 ID는 50자 이하여야 합니다.');
  }
  
  return true;
}

// API 요청 래퍼
async function apiRequest(url, options = {}) {
  try {
    const response = await fetch(url, options);
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }
    
    return response;
  } catch (error) {
    console.error('API 요청 실패:', error);
    throw error;
  }
}
```

---

## 5. 보안 고려사항

1. **사용자 ID 검증**: 서버에서 정규식으로 안전한 문자만 허용
2. **워크스페이스 격리**: 사용자별 독립된 폴더로 파일 격리
3. **경로 순회 방지**: 상위 디렉터리 접근 차단
4. **입력 검증**: 모든 폼 입력값에 대한 검증
5. **프로세스 격리**: 각 실행은 독립 프로세스로 실행

---

## 6. 확장 계획

### 6.1 추가 예정 기능
- [ ] 사용자 인증 시스템 (JWT 토큰)
- [ ] 워크스페이스 공유 기능
- [ ] 실시간 협업 (WebSocket)
- [ ] 클라우드 저장소 연동
- [ ] 버전 관리 시스템

### 6.2 API 버전 관리
- 현재 버전: v2
- 호환성: 사용자 ID 기반 시스템으로 전환




-----------
## result 추가

### 1. `/result` 엔드포인트
- **GET** 요청으로 `user_id` 파라미터 필수
- 평가 결과를 JSON으로 반환:
  ```json
  {
    "ok": true,
    "user_id": "hong_gildong",
    "accuracy": 0.8575,
    "confusion_matrix": "iVBORw0KGgo...", // base64 문자열
    "misclassified_samples": "iVBORw0KGgo...", // base64 문자열
    "prediction_samples": "iVBORw0KGgo...", // base64 문자열
    "message": "평가 결과를 성공적으로 가져왔습니다."
  }
  ```

### 2. `/result/status` 엔드포인트 (보조)
- 결과 파일들의 존재 여부를 미리 확인
- 프론트엔드에서 "결과보기" 버튼 활성화 여부 판단에 활용 가능

## 사용 방법

### JavaScript에서 호출 예시:
```javascript
// 결과 상태 확인
async function checkResultStatus(userId) {
  const response = await fetch(`/result/status?user_id=${encodeURIComponent(userId)}`);
  const status = await response.json();
  return status.ready; // true/false
}

// 결과 가져오기
async function getResults(userId) {
  const response = await fetch(`/result?user_id=${encodeURIComponent(userId)}`);
  const results = await response.json();
  
  if (results.ok) {
    console.log('정확도:', results.accuracy);
    
    // 이미지 표시 예시
    if (results.confusion_matrix) {
      const img = document.createElement('img');
      img.src = `data:image/png;base64,${results.confusion_matrix}`;
      document.body.appendChild(img);
    }
  }
  
  return results;
}
```

## 에러 처리

- **사용자 ID 누락**: `400 Bad Request`
- **파일 접근 오류**: `500 Internal Server Error`
- **일부 파일 누락**: 성공 응답이지만 `warning` 필드 포함
