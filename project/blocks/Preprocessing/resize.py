# blocks/resize.py

def generate_resize_snippet(resize_n):
    """
    X_train,X_test (NumPy 배열) → (resize_n×resize_n) PyTorch 텐서로 변환하는 스니펫 생성
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# ----[이미지 크기 변경 블록]-----")
    lines.append("# -----------------------------")
    lines.append("import numpy as np")
    lines.append("from torchvision import transforms")
    lines.append("")
    lines.append(f"# 이미지 크기 변경: {resize_n}×{resize_n}")
    lines.append("# - X_train, X_test 은 현재 NumPy 배열(shape: N×784)")
    lines.append("transform = transforms.Compose([")
    lines.append("    transforms.ToPILImage(),              # NumPy 배열 → PIL 이미지")
    lines.append(f"    transforms.Resize(({resize_n}, {resize_n})),  # 지정 크기로 리사이즈")
    lines.append("    transforms.ToTensor()                 # PIL → Tensor (C×H×W), 값 0~1")
    lines.append("])")
    lines.append("")
    lines.append("# 1) 학습 데이터 리사이즈")
    lines.append("images_2d = X_train.reshape(-1, 28, 28).astype(np.uint8)   # 1D→2D 전환")
    lines.append("X_train = torch.stack([transform(img) for img in images_2d], dim=0)")
    lines.append("")
    lines.append("# 2) 테스트 데이터 리사이즈")
    lines.append("images_2d = X_test.reshape(-1, 28, 28).astype(np.uint8)")
    lines.append("X_test  = torch.stack([transform(img) for img in images_2d], dim=0)")
    lines.append("")
    return "\n".join(lines)
