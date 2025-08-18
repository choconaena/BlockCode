# AI 블록코딩 API 명세서

## 1. 개요

### 1.1 시스템 아키텍처
- **Frontend**: HTML/CSS/JS (블록 UI, 코드 미리보기, 데이터 시각화)
- **Backend**: Flask (코드 생성, 실행, 로그 스트리밍)
- **Storage**: 사용자별 워크스페이스 (UUID 기반 격리)

### 1.2 핵심 개념
- **블록(Block)**: AI 파이프라인의 각 단계를 나타내는 UI 컴포넌트
- **스니펫(Snippet)**: 블록 설정에 따라 생성되는 Python 코드 조각
- **워크스페이스(Workspace)**: 사용자별 작업 공간 (코드, 데이터, 로그)
- **스테이지(Stage)**: 파이프라인 단계 (pre/model/train/eval)

---

## 2. API 엔드포인트

### 2.1 페이지 렌더링

#### GET `/`
**설명**: 루트 접근 시 `/app`으로 리다이렉트  
**응답**: 302 Redirect to `/app`  
**쿠키**: `uid` 설정 (없을 경우 새로 생성)

#### GET `/app`
**설명**: 메인 애플리케이션 페이지  
**응답**: HTML (index.html)  
**템플릿 변수**:
```python
{
    "options": ["mnist_train.csv", ...],  # dataset/ 폴더의 CSV 목록
    "form_state": {...},                  # 저장된 폼 상태 (JSON)
    "snippet_pre": "...",                 # 전처리 코드
    "snippet_model": "...",               # 모델 코드
    "snippet_train": "...",               # 학습 코드
    "snippet_eval": "..."                 # 평가 코드
}
```

---

### 2.2 코드 생성 및 변환

#### POST `/convert`
**설명**: 블록 설정을 Python 코드로 변환  
**요청 본문** (form-data):

##### 공통 파라미터
| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| stage | string | ✓ | 변환 대상: "pre", "model", "train", "eval", "all" |

##### 전처리 블록 파라미터 (stage=pre)
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| dataset | string | - | 훈련 데이터 CSV 파일명 |
| is_test | string | "false" | 테스트 데이터 사용 여부 ("true"/"false") |
| testdataset | string | - | 테스트 데이터 CSV 파일명 (is_test=true일 때) |
| a | number | 80 | 훈련 데이터 비율 (%) (is_test=false일 때) |
| drop_na | checkbox | - | 결측치 제거 여부 |
| drop_bad | checkbox | - | 잘못된 라벨 제거 여부 |
| min_label | number | 0 | 최소 라벨값 (drop_bad=true일 때) |
| max_label | number | 9 | 최대 라벨값 (drop_bad=true일 때) |
| split_xy | checkbox | - | X/y 분리 여부 |
| resize_n | number | 28 | 이미지 리사이즈 크기 (n×n) |
| augment_method | string | - | 증강 방법: "rotate", "hflip", "vflip", "translate" |
| augment_param | number | - | 증강 파라미터 (각도/픽셀) |
| normalize | string | - | 정규화 방법: "0-1", "-1-1" |

##### 모델 설계 블록 파라미터 (stage=model)
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| input_w | number | 28 | 입력 이미지 너비 |
| input_h | number | 28 | 입력 이미지 높이 |
| input_c | number | 1 | 입력 채널 수 (1:흑백, 3:컬러) |
| conv1_filters | number | 32 | Conv1 필터 수 |
| conv1_kernel | number | 3 | Conv1 커널 크기 |
| conv1_padding | string | "valid" | Conv1 패딩: "same", "valid" |
| conv1_activation | string | "relu" | Conv1 활성함수 |
| pool1_type | string | - | Pool1 종류: "max", "avg" |
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

##### 학습 블록 파라미터 (stage=train)
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| loss_method | string | "CrossEntropy" | 손실함수: "CrossEntropy", "MSE" |
| optimizer_method | string | "Adam" | 옵티마이저: "Adam", "SGD", "RMSprop" |
| learning_rate | number | 0.00001 | 학습률 |
| epochs | number | 10 | 에폭 수 |
| batch_size | number | 64 | 배치 크기 |
| patience | number | 3 | Early stopping patience |

##### 평가 블록 파라미터 (stage=eval)
| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| metrics | checkbox[] | ["accuracy"] | 평가 메트릭 (복수 선택) |
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

**응답**: HTML (index.html) - 생성된 코드와 함께 재렌더링

---

### 2.3 코드 실행 및 로그

#### POST `/run/<stage>`
**설명**: 특정 스테이지 코드 실행  
**경로 파라미터**: 
- `stage`: "pre", "model", "train", "eval"

**응답**:
```json
{
    "ok": true
}
```
또는
```json
{
    "error": "error message"
}
```

#### GET `/logs/stream`
**설명**: Server-Sent Events로 실행 로그 스트리밍  
**쿼리 파라미터**:
- `stage`: 로그를 볼 스테이지 (기본값: "train")

**응답**: text/event-stream
```
data: [pre][2025-01-01 12:00:00] Processing started...
data: [pre][2025-01-01 12:00:01] Loading data...
```

---

### 2.4 데이터 정보 조회

#### GET `/data-info`
**설명**: CSV 데이터셋 정보 조회  
**쿼리 파라미터**:
- `file`: CSV 파일명
- `type`: 정보 유형 ("shape", "structure", "sample", "images")
- `n`: 샘플/이미지 개수 (type=sample/images일 때)

**응답 예시**:

##### type=shape
```json
{
    "rows": 60000,
    "cols": 785
}
```

##### type=structure
```json
{
    "columns": [
        {"name": "label", "dtype": "int64"},
        {"name": "pixel0", "dtype": "int64"},
        ...
    ]
}
```

##### type=sample
```json
{
    "columns": ["label", "pixel0", ...],
    "sample": [
        [5, 0, 0, ...],
        [0, 0, 0, ...],
        ...
    ]
}
```

##### type=images
```json
{
    "images": [
        "data:image/png;base64,iVBORw0KGgo...",
        ...
    ]
}
```

---

### 2.5 파일 다운로드

#### GET `/download/<stage>`
**설명**: 생성된 코드 파일 다운로드  
**경로 파라미터**:
- `stage`: "pre", "model", "train", "eval", "all"

**응답**: 
- stage="pre": preprocessing.py
- stage="model": model.py
- stage="train": training.py
- stage="eval": evaluation.py
- stage="all": workspace_<uid>_<timestamp>.zip

---

## 3. 파일 구조 및 생성 규칙

### 3.1 워크스페이스 구조
```
workspace/<uid>/
├── preprocessing.py    # 전처리 코드
├── model.py           # 모델 정의
├── training.py        # 학습 코드
├── evaluation.py      # 평가 코드
├── inputs_pre.json    # 전처리 입력값 저장
├── inputs_model.json  # 모델 입력값 저장
├── inputs_train.json  # 학습 입력값 저장
├── inputs_eval.json   # 평가 입력값 저장
├── data/
│   └── dataset.pt     # 전처리된 데이터
└── artifacts/
    ├── best_model.pth # 학습된 모델
    ├── confusion_matrix.png
    ├── samples.png
    └── misclassified.png
```

### 3.2 코드 생성 규칙

#### 전처리 (preprocessing.py)
1. 데이터 로드 (필수)
2. 결측치 제거 (선택)
3. 라벨 필터링 (선택)
4. X/y 분리 (선택)
5. 이미지 리사이즈 (선택)
6. 데이터 증강 (선택)
7. 정규화 (선택)
8. 데이터 저장 (필수)

#### 모델 (model.py)
1. CNN 클래스 정의
2. Conv 레이어 (1~2개)
3. Pooling 레이어
4. Dropout (선택)
5. Flatten
6. Dense 레이어
7. Output 레이어

#### 학습 (training.py)
1. 데이터 로드
2. 모델 생성
3. 손실함수/옵티마이저 설정
4. 학습 루프
5. Early stopping
6. 체크포인트 저장

#### 평가 (evaluation.py)
1. 데이터/모델 로드
2. 추론
3. 메트릭 계산
4. 리포트 생성
5. 시각화

---

## 4. 에러 처리

### HTTP 상태 코드
- 200: 성공
- 302: 리다이렉트
- 400: 잘못된 요청 (unknown stage)
- 404: 파일 없음
- 500: 서버 에러

### 에러 응답 형식
```json
{
    "error": "error description"
}
```

---

## 5. 보안 고려사항

1. **경로 순회 방지**: 모든 파일 접근은 화이트리스트 기반
2. **세션 격리**: UUID 기반 사용자별 워크스페이스
3. **쿠키 보안**: httponly=True, samesite="Lax"
4. **입력 검증**: 모든 숫자 입력에 min/max 제한
5. **프로세스 격리**: 각 실행은 독립 프로세스

---

## 6. 확장 계획

### 추가 예정 기능
- [ ] RNN/LSTM 모델 지원
- [ ] 다양한 데이터셋 형식 지원 (이미지 폴더, JSON)
- [ ] 하이퍼파라미터 튜닝
- [ ] 모델 비교/앙상블
- [ ] 실시간 학습 그래프
- [ ] 코드 export (Jupyter Notebook)
- [ ] 클라우드 학습 연동

### API 버전 관리
- 현재 버전: v1.0
- 버전 업데이트 시 `/api/v2/` 경로 사용 예정