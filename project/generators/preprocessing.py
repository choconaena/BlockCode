# generators/preprocessing.py
# 전처리 코드 생성기

from .base import BaseGenerator


class PreprocessingGenerator(BaseGenerator):
    """전처리 코드 생성기"""
    
    def generate(self, form_data):
        """전처리 코드 생성"""
        self.clear_blocks()
        
        # 헤더
        self._add_header()
        
        # 데이터 선택 (필수)
        self._add_data_selection(form_data)
        
        # 결측치 제거
        if self.is_checked(form_data, 'drop_na'):
            self._add_drop_na()
        else:
            self._add_skip_log("결측치 행 삭제")
        
        # 잘못된 라벨 제거
        if self.is_checked(form_data, 'drop_bad'):
            self._add_drop_bad_labels(form_data)
        else:
            self._add_skip_log("잘못된 라벨 삭제")
        
        # X/y 분리
        if self.is_checked(form_data, 'split_xy'):
            self._add_split_xy()
        else:
            self._add_skip_log("입력/라벨 분리(X/y)")
        
        # 이미지 리사이즈
        resize_n = self.get_form_value(form_data, 'resize_n')
        if resize_n:
            self._add_resize(int(resize_n))
        else:
            self._add_skip_log("이미지 크기 변경")
        
        # 이미지 증강
        augment_method = self.get_form_value(form_data, 'augment_method')
        augment_param = self.get_form_value(form_data, 'augment_param')
        if augment_method and augment_param:
            self._add_augmentation(augment_method, int(augment_param))
        else:
            self._add_skip_log("이미지 증강")
        
        # 정규화
        normalize = self.get_form_value(form_data, 'normalize')
        if normalize:
            self._add_normalization(normalize)
        else:
            self._add_skip_log("정규화")
        
        # 저장 (필수)
        self._add_save_data()
        
        return self.render_blocks()
    
    def _add_header(self):
        """헤더 추가"""
        code = '''# 자동 생성된 preprocessing.py
# AI 블록코딩 - 전처리 파이프라인

import pandas as pd
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import time
import datetime
import os

# 로깅 유틸
def _ts():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def log(msg):
    sys.stdout.write(f"[pre][{_ts()}] {msg}\\n")
    sys.stdout.flush()

log("=== PREPROCESSING START ===")
'''
        self.add_block("header", code, required=True)
    
    def _add_data_selection(self, form_data):
        """데이터 선택 블록"""
        dataset = self.get_form_value(form_data, 'dataset', 'mnist_train.csv')
        is_test = self.get_form_value(form_data, 'is_test', 'false')
        testdataset = self.get_form_value(form_data, 'testdataset', '')
        a = int(self.get_form_value(form_data, 'a', 80))
        
        code = f'''
# 데이터 선택 및 로드
t0_data = time.perf_counter()
log('START: 데이터 선택/로딩')

train_df = pd.read_csv('dataset/{dataset}')
'''
        
        if is_test == 'true' and testdataset:
            code += f'''test_df = pd.read_csv('dataset/{testdataset}')
'''
        else:
            test_ratio = (100 - a) / 100.0
            code += f'''
# 학습 데이터를 {a}%/{100-a}%로 분할
test_df = train_df.sample(frac={test_ratio}, random_state=42)
train_df = train_df.drop(test_df.index).reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
'''
        
        code += '''
log(f'END  : 데이터 선택/로딩 (elapsed={time.perf_counter()-t0_data:.3f}s)')
log(f'Train size: {len(train_df)}, Test size: {len(test_df)}')
'''
        self.add_block("data_selection", code, required=True)
    
    def _add_drop_na(self):
        """결측치 제거 블록"""
        code = '''
# 결측치 제거
t0_dropna = time.perf_counter()
log('START: 결측치 행 삭제')

before_train = len(train_df)
before_test = len(test_df)
train_df = train_df.dropna()
test_df = test_df.dropna()

log(f'Removed {before_train - len(train_df)} rows from train, {before_test - len(test_df)} rows from test')
log(f'END  : 결측치 행 삭제 (elapsed={time.perf_counter()-t0_dropna:.3f}s)')
'''
        self.add_block("drop_na", code)
    
    def _add_drop_bad_labels(self, form_data):
        """잘못된 라벨 제거 블록"""
        min_label = int(self.get_form_value(form_data, 'min_label', 0))
        max_label = int(self.get_form_value(form_data, 'max_label', 9))
        
        code = f'''
# 잘못된 라벨 제거
t0_bad = time.perf_counter()
log('START: 잘못된 라벨 삭제 (허용={min_label}~{max_label})')

before_train = len(train_df)
before_test = len(test_df)
train_df = train_df[train_df['label'].between({min_label}, {max_label})]
test_df = test_df[test_df['label'].between({min_label}, {max_label})]

log(f'Removed {{before_train - len(train_df)}} rows from train, {{before_test - len(test_df)}} rows from test')
log(f'END  : 잘못된 라벨 삭제 (elapsed={{time.perf_counter()-t0_bad:.3f}}s)')
'''
        self.add_block("drop_bad_labels", code)
    
    def _add_split_xy(self):
        """X/y 분리 블록"""
        code = '''
# 입력/라벨 분리
t0_split = time.perf_counter()
log('START: 입력/라벨 분리(X/y)')

# 학습 데이터
X_train = train_df.iloc[:, 1:].values  # 입력 (픽셀값)
y_train = train_df.iloc[:, 0].values   # 라벨
y_train = torch.from_numpy(y_train).long()

# 테스트 데이터
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values
y_test = torch.from_numpy(y_test).long()

log(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
log(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
log(f'END  : 입력/라벨 분리(X/y) (elapsed={time.perf_counter()-t0_split:.3f}s)')
'''
        self.add_block("split_xy", code)
    
    def _add_resize(self, size):
        """이미지 리사이즈 블록"""
        code = f'''
# 이미지 크기 변경
t0_resize = time.perf_counter()
log('START: 이미지 크기 변경 -> {size}x{size}')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(({size}, {size})),
    transforms.ToTensor()
])

# 학습 데이터 리사이즈
images_2d = X_train.reshape(-1, 28, 28).astype(np.uint8)
X_train = torch.stack([transform(img) for img in images_2d], dim=0)

# 테스트 데이터 리사이즈
images_2d = X_test.reshape(-1, 28, 28).astype(np.uint8)
X_test = torch.stack([transform(img) for img in images_2d], dim=0)

log(f'Resized to: X_train {{X_train.shape}}, X_test {{X_test.shape}}')
log(f'END  : 이미지 크기 변경 (elapsed={{time.perf_counter()-t0_resize:.3f}}s)')
'''
        self.add_block("resize", code)
    
    def _add_augmentation(self, method, param):
        """이미지 증강 블록"""
        transform_map = {
            'rotate': f'transforms.RandomRotation(({param}, {param}))',
            'hflip': 'transforms.RandomHorizontalFlip(p=1.0)',
            'vflip': 'transforms.RandomVerticalFlip(p=1.0)',
            'translate': f'transforms.RandomAffine(degrees=0, translate=({param/100}, {param/100}))'
        }
        
        transform_code = transform_map.get(method, transform_map['rotate'])
        
        code = f'''
# 이미지 증강
t0_aug = time.perf_counter()
log('START: 이미지 증강 (방법={method}, 파라미터={param})')

transform_aug = {transform_code}

# 학습 데이터 증강
aug_train = torch.stack([transform_aug(x) for x in X_train], dim=0)
X_train = torch.cat([X_train, aug_train], dim=0)
y_train = torch.cat([y_train, y_train], dim=0)

# 테스트 데이터 증강
aug_test = torch.stack([transform_aug(x) for x in X_test], dim=0)
X_test = torch.cat([X_test, aug_test], dim=0)
y_test = torch.cat([y_test, y_test], dim=0)

log(f'Augmented to: X_train {{X_train.shape}}, X_test {{X_test.shape}}')
log(f'END  : 이미지 증강 (elapsed={{time.perf_counter()-t0_aug:.3f}}s)')
'''
        self.add_block("augmentation", code)
    
    def _add_normalization(self, method):
        """정규화 블록"""
        if method == '0-1':
            norm_code = '''X_train = X_train / 255.0
X_test = X_test / 255.0'''
        else:  # -1-1
            norm_code = '''X_train = X_train / 127.5 - 1.0
X_test = X_test / 127.5 - 1.0'''
        
        code = f'''
# 픽셀 값 정규화
t0_norm = time.perf_counter()
log('START: 정규화 (방법={method})')

{norm_code}

log(f'Normalized: X_train min={{X_train.min():.2f}}, max={{X_train.max():.2f}}')
log(f'END  : 정규화 (elapsed={{time.perf_counter()-t0_norm:.3f}}s)')
'''
        self.add_block("normalization", code)
    
    def _add_skip_log(self, name):
        """스킵 로그"""
        code = f"log('SKIP : {name}')"
        self.add_block(f"skip_{name}", code)
    
    def _add_save_data(self):
        """데이터 저장 블록"""
        code = '''
# 전처리 결과 저장
t0_save = time.perf_counter()
log('START: 전처리 결과 저장(dataset.pt)')

WORKDIR = os.environ.get("AIB_WORKDIR", ".")
os.makedirs(os.path.join(WORKDIR, "data"), exist_ok=True)
save_path = os.path.join(WORKDIR, "data", "dataset.pt")

torch.save({
    "X_train": X_train,
    "y_train": y_train,
    "X_test": X_test,
    "y_test": y_test
}, save_path)

log(f'Saved to: {save_path}')
log(f'END  : 전처리 결과 저장 (elapsed={time.perf_counter()-t0_save:.3f}s)')
log('=== PREPROCESSING DONE ===')
'''
        self.add_block("save_data", code, required=True)