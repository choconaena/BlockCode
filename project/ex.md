# AI 블록코딩 프로젝트 - 개선된 구조

## 📁 프로젝트 구조

```
project/
├── app.py                      # 메인 Flask 애플리케이션 (개선됨)
├── requirements.txt            # 파이썬 패키지 의존성
├── README.md                   # 프로젝트 문서
├── api_spec.md                # API 명세서
│
├── generators/                 # 코드 생성 모듈 (새로 추가)
│   ├── __init__.py
│   ├── base.py                # 베이스 생성기 클래스
│   ├── preprocessing.py       # 전처리 코드 생성기
│   ├── model.py              # 모델 코드 생성기
│   ├── training.py           # 학습 코드 생성기
│   └── evaluation.py         # 평가 코드 생성기
│
├── templates/                 # HTML 템플릿
│   ├── layout.html           # 기본 레이아웃
│   ├── index.html            # 메인 페이지
│   ├── sidebar.html          # 좌측 블록 UI
│   ├── main_code.html        # 코드 탭
│   ├── main_data.html        # 데이터 구조 탭
│   └── main_log.html         # 로그 탭
│
├── static/                    # 정적 파일
│   ├── css/
│   │   └── style.css         # 스타일시트
│   └── js/
│       ├── sidebar.js        # 사이드바 인터랙션
│       ├── tabs.js           # 탭 전환
│       ├── data_info.js      # 데이터 정보 조회
│       └── logs.js           # 로그 스트리밍
│
├── dataset/                   # CSV 데이터셋 폴더
│   ├── mnist_train.csv
│   └── mnist_test.csv
│
├── workspace/                 # 사용자별 작업 공간
│   └── <uid>/
│       ├── preprocessing.py
│       ├── model.py
│       ├── training.py
│       ├── evaluation.py
│       ├── inputs_*.json
│       ├── data/
│       │   └── dataset.pt
│       └── artifacts/
│           ├── best_model.pth
│           ├── training_history.json
│           └── evaluation_results.json
│
└── logs/                      # 실행 로그
    └── <uid>_<stage>.log
```

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
# requirements.txt 생성
cat > requirements.txt << EOF
Flask==2.3.0
pandas==2.0.0
numpy==1.24.0
torch==2.0.0
torchvision==0.15.0
Pillow==10.0.0
scikit-learn==1.3.0
matplotlib==3.7.0
tqdm==4.65.0
EOF

# 패키지 설치
pip install -r requirements.txt
```

### 2. 프로젝트 초기 설정

```bash
# 필수 디렉터리 생성
mkdir -p generators templates static/css static/js dataset workspace logs

# generators/__init__.py 생성
cat > generators/__init__.py << EOF
from .preprocessing import PreprocessingGenerator
from .model import ModelGenerator
from .training import TrainingGenerator
from .evaluation import EvaluationGenerator

__all__ = [
    'PreprocessingGenerator',
    'ModelGenerator',
    'TrainingGenerator',
    'EvaluationGenerator'
]
EOF
```

### 3. 샘플 데이터 준비

```python
# prepare_sample_data.py
import pandas as pd
import numpy as np

# MNIST 스타일 더미 데이터 생성
n_samples = 1000
n_features = 784  # 28x28

# 훈련 데이터
train_data = np.random.randint(0, 256, (n_samples, n_features))
train_labels = np.random.randint(0, 10, n_samples)
train_df = pd.DataFrame(train_data)
train_df.insert(0, 'label', train_labels)
train_df.to_csv('dataset/mnist_train.csv', index=False)

# 테스트 데이터
test_data = np.random.randint(0, 256, (n_samples//5, n_features))
test_labels = np.random.randint(0, 10, n_samples//5)
test_df = pd.DataFrame(test_data)
test_df.insert(0, 'label', test_labels)
test_df.to_csv('dataset/mnist_test.csv', index=False)

print("Sample data created!")
```

### 4. 서버 실행

```bash
# 개발 서버 실행
python app.py

# 프로덕션 서버 (Gunicorn)
pip install gunicorn
gunicorn -w 4 -b 127.0.0.1:9000 app:app
```

## 🔧 주요 개선 사항

### 1. 코드 구조 개선
- ✅ 코드 생성 로직을 별도 모듈로 분리 (`generators/`)
- ✅ 클래스 기반 아키텍처로 재구성
- ✅ 명확한 책임 분리 (WorkspaceManager, DatasetManager, ProcessManager)

### 2. API 체계화
- ✅ RESTful 엔드포인트 설계
- ✅ 일관된 에러 처리
- ✅ 명확한 요청/응답 형식

### 3. 코드 품질
- ✅ 타입 힌트 추가
- ✅ 문서화 개선
- ✅ 에러 처리 강화
- ✅ 로깅 시스템 개선

### 4. 유지보수성
- ✅ 모듈화된 구조
- ✅ 설정 상수 중앙 관리
- ✅ 재사용 가능한 컴포넌트

## 📝 사용 예시

### 1. 전처리 블록 설정
```python
# POST /convert
{
    "stage": "pre",
    "dataset": "mnist_train.csv",
    "is_test": "false",
    "a": "80",
    "drop_na": "on",
    "split_xy": "on",
    "resize_n": "28",
    "normalize": "0-1"
}
```

### 2. 모델 설계 블록 설정
```python
# POST /convert
{
    "stage": "model",
    "input_w": "28",
    "input_h": "28",
    "input_c": "1",
    "conv1_filters": "32",
    "conv1_kernel": "3",
    "conv1_padding": "same",
    "conv1_activation": "relu",
    "pool1_type": "max",
    "pool1_size": "2",
    "dense_units": "128",
    "num_classes": "10"
}
```

### 3. 학습 실행
```python
# POST /run/train
# 응답: {"ok": true, "pid": 12345}

# GET /logs/stream?stage=train
# SSE로 실시간 로그 수신
```

## 🔍 디버깅 팁

### 로그 확인
```bash
# 실시간 로그 모니터링
tail -f logs/<uid>_<stage>.log

# 전체 로그 확인
cat logs/<uid>_pre.log
```

### 생성된 코드 확인
```bash
# 사용자 워크스페이스 확인
ls -la workspace/<uid>/

# 생성된 코드 보기
cat workspace/<uid>/preprocessing.py
```

### 데이터 확인
```python
# Python에서 저장된 데이터 확인
import torch
data = torch.load('workspace/<uid>/data/dataset.pt')
print(f"X_train shape: {data['X_train'].shape}")
print(f"y_train shape: {data['y_train'].shape}")
```

## 📚 추가 문서

- [API 명세서](api_spec.md) - 전체 API 엔드포인트 문서
- [코드 생성기 가이드](generators/README.md) - 코드 생성 모듈 개발 가이드
- [프론트엔드 가이드](static/README.md) - UI/UX 커스터마이징 가이드

## 🤝 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

Apache 2.0 License

## 👥 문의

- 이슈 트래커: GitHub Issues
- 이메일: your-email@example.com
