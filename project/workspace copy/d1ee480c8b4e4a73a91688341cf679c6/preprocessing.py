
# 자동 생성된 preprocessing.py
# 필요한 라이브러리 임포트
import pandas as pd
import torch, numpy as np
from PIL import Image
from torchvision import transforms

# ---- 로깅 유틸 -------------------------------------------------------------
import sys, time, datetime

def _ts():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def log(msg):
    # 표준출력으로 즉시 흘려보내기(로그 탭에 실시간 노출)
    sys.stdout.write(f"[pre][{_ts()}] {msg}\n")
    sys.stdout.flush()

log("=== PREPROCESSING START ===")


t0_data = time.perf_counter()
log('START: 데이터 선택/로딩')
# -----------------------------
# ---------[데이터 선택 블록]---------
# -----------------------------
train_df = pd.read_csv('dataset/mnist_test.csv')  # 'mnist_test.csv' 파일에서 학습용 데이터 로드

# 테스트 미지정 → 학습 데이터를 80% 사용, 나머지 20%를 테스트로 분할
test_df  = train_df.sample(frac=0.2, random_state=42)
train_df = train_df.drop(test_df.index).reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

log(f'END  : 데이터 선택/로딩 (elapsed={time.perf_counter()-t0_data:.3f}s)')

log('SKIP : 결측치 행 삭제')

log('SKIP : 잘못된 라벨 삭제')

t0_split = time.perf_counter()
log('START: 입력/라벨 분리(X/y)')
# -----------------------------
# ------[입력/라벨 분리 블록]------
# -----------------------------
import torch

# 1) 학습 데이터(X_train, y_train) 분리
X_train = train_df.iloc[:, 1:].values  # 학습용 입력 데이터 (NumPy 배열)
y_train = train_df.iloc[:, 0].values     # 학습용 라벨 데이터 (NumPy 배열)
y_train = torch.from_numpy(y_train).long()  # NumPy → LongTensor 변환

# 2) 테스트 데이터(X_test, y_test) 분리
X_test  = test_df.iloc[:, 1:].values   # 테스트용 입력 데이터 (NumPy 배열)
y_test  = test_df.iloc[:, 0].values     # 테스트용 라벨 데이터 (NumPy 배열)
y_test  = torch.from_numpy(y_test).long()   # NumPy → LongTensor 변환

log(f'END  : 입력/라벨 분리(X/y) (elapsed={time.perf_counter()-t0_split:.3f}s)')

t0_resize = time.perf_counter()
log('START: 이미지 크기 변경 -> 28x28')
# -----------------------------
# ----[이미지 크기 변경 블록]-----
# -----------------------------
import numpy as np
from torchvision import transforms

# 이미지 크기 변경: 28×28
# - X_train, X_test 은 현재 NumPy 배열(shape: N×784)
transform = transforms.Compose([
    transforms.ToPILImage(),              # NumPy 배열 → PIL 이미지
    transforms.Resize((28, 28)),  # 지정 크기로 리사이즈
    transforms.ToTensor()                 # PIL → Tensor (C×H×W), 값 0~1
])

# 1) 학습 데이터 리사이즈
images_2d = X_train.reshape(-1, 28, 28).astype(np.uint8)   # 1D→2D 전환
X_train = torch.stack([transform(img) for img in images_2d], dim=0)

# 2) 테스트 데이터 리사이즈
images_2d = X_test.reshape(-1, 28, 28).astype(np.uint8)
X_test  = torch.stack([transform(img) for img in images_2d], dim=0)

log(f'END  : 이미지 크기 변경 (elapsed={time.perf_counter()-t0_resize:.3f}s)')

log('SKIP : 이미지 증강')

log('SKIP : 정규화')


# =============================
# [전처리 결과 저장 블록]
# =============================
import os, torch
t0_save = time.perf_counter()
log('START: 전처리 결과 저장(dataset.pt)')

WORKDIR = os.environ.get("AIB_WORKDIR", ".")
os.makedirs(os.path.join(WORKDIR, "data"), exist_ok=True)
save_path = os.path.join(WORKDIR, "data", "dataset.pt")

# 기대 형태:
#   X_train/X_test: torch.Tensor [N, C, H, W]
#   y_train/y_test: torch.LongTensor [N]
torch.save(
    {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test},
    save_path
)
log(f'END  : 전처리 결과 저장(dataset.pt) (elapsed={time.perf_counter()-t0_save:.3f}s)')
log('=== PREPROCESSING DONE ===')
