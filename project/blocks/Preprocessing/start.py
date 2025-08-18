# blocks/Preprocessing/start.py
# 각 전처리 블록을 조립하며, 실행 시 섹션별 시작/종료 로그가 나오도록 스니펫을 생성한다.

from .data_selection     import generate_data_selection_snippet
from .drop_na            import generate_drop_na_snippet
from .drop_bad_labels    import generate_drop_bad_labels_snippet
from .split_xy           import generate_split_xy_snippet
from .resize             import generate_resize_snippet
from .augment            import generate_augment_snippet
from .normalize          import generate_normalize_snippet

def generate_preprocessing_snippet(form):
    """
    form: request.form 딕셔너리
    → preprocessing 단계(1~7) 전체 코드를 조립하여 반환
      - 각 블록 앞뒤로 [START]/[END] 로그 출력
      - 선택되지 않은 블록은 [SKIP] 로그 출력
      - 마지막에 저장 블록도 [START]/[END] 포함
    """
    # 1) form에서 파라미터 꺼내기
    dataset       = form['dataset']
    is_test       = form['is_test']
    testdataset   = form.get('testdataset', '')
    a             = form.get('a', '100')
    drop_na_flag  = 'drop_na' in form
    drop_bad_flag = 'drop_bad' in form
    min_label     = form.get('min_label', '0')
    max_label     = form.get('max_label', '9')
    split_xy_flag = 'split_xy' in form
    resize_n      = form.get('resize_n', '')
    augment_m     = form.get('augment_method', '')
    augment_p     = form.get('augment_param', '')
    normalize_m   = form.get('normalize', '')

    # 2) 스니펫 조립(머리말 + 공통 유틸)
    head = r"""
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
"""

    lines = [head, ""]

    # 3) 데이터 불러오기 (필수)
    #    - 시작/종료 로그 + 경과 시간
    data_sel = generate_data_selection_snippet(dataset, is_test, testdataset, a)
    lines += [
        "t0_data = time.perf_counter()",
        "log('START: 데이터 선택/로딩')",
        data_sel,
        "log(f'END  : 데이터 선택/로딩 (elapsed={time.perf_counter()-t0_data:.3f}s)')",
        ""
    ]

    # 4) 각 블록별 조건적 삽입 + 로그
    # 4-1) 결측치 제거
    if drop_na_flag:
        lines += [
            "t0_dropna = time.perf_counter()",
            "log('START: 결측치 행 삭제')",
            generate_drop_na_snippet(),
            "log(f'END  : 결측치 행 삭제 (elapsed={time.perf_counter()-t0_dropna:.3f}s)')",
            ""
        ]
    else:
        lines += ["log('SKIP : 결측치 행 삭제')", ""]

    # 4-2) 잘못된 라벨 제거
    if drop_bad_flag:
        lines += [
            "t0_bad = time.perf_counter()",
            f"log('START: 잘못된 라벨 삭제 (허용={min_label}~{max_label})')",
            generate_drop_bad_labels_snippet(min_label, max_label),
            "log(f'END  : 잘못된 라벨 삭제 (elapsed={time.perf_counter()-t0_bad:.3f}s)')",
            ""
        ]
    else:
        lines += ["log('SKIP : 잘못된 라벨 삭제')", ""]

    # 4-3) X/y 분리
    if split_xy_flag:
        lines += [
            "t0_split = time.perf_counter()",
            "log('START: 입력/라벨 분리(X/y)')",
            generate_split_xy_snippet(),
            "log(f'END  : 입력/라벨 분리(X/y) (elapsed={time.perf_counter()-t0_split:.3f}s)')",
            ""
        ]
    else:
        lines += ["log('SKIP : 입력/라벨 분리(X/y)')", ""]

    # 4-4) 리사이즈
    if resize_n:
        try:
            resize_int = int(resize_n)
        except:
            resize_int = 28
        lines += [
            "t0_resize = time.perf_counter()",
            f"log('START: 이미지 크기 변경 -> {resize_int}x{resize_int}')",
            generate_resize_snippet(resize_int),
            "log(f'END  : 이미지 크기 변경 (elapsed={time.perf_counter()-t0_resize:.3f}s)')",
            ""
        ]
    else:
        lines += ["log('SKIP : 이미지 크기 변경')", ""]

    # 4-5) 증강
    if augment_m and augment_p:
        try:
            augment_p_int = int(augment_p)
        except:
            augment_p_int = 0
        lines += [
            "t0_aug = time.perf_counter()",
            f"log('START: 이미지 증강 (방법={augment_m}, 파라미터={augment_p_int})')",
            generate_augment_snippet(augment_m, augment_p_int),
            "log(f'END  : 이미지 증강 (elapsed={time.perf_counter()-t0_aug:.3f}s)')",
            ""
        ]
    else:
        lines += ["log('SKIP : 이미지 증강')", ""]

    # 4-6) 정규화
    if normalize_m:
        lines += [
            "t0_norm = time.perf_counter()",
            f"log('START: 정규화 (방법={normalize_m})')",
            generate_normalize_snippet(normalize_m),
            "log(f'END  : 정규화 (elapsed={time.perf_counter()-t0_norm:.3f}s)')",
            ""
        ]
    else:
        lines += ["log('SKIP : 정규화')", ""]

    # 5) 필수 저장 블록 (맨 마지막)
    saving_footer = r"""
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
"""
    lines.append(saving_footer)

    # 최종 문자열 반환
    return "\n".join(lines)