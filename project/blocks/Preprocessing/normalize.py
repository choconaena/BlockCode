# blocks/normalize.py

def generate_normalize_snippet(method):
    """
    X_train/X_test 을 0~1 또는 -1~1 범위로 정규화
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# ---[픽셀 값 정규화 블록]---")
    lines.append("# -----------------------------")
    if method == '0-1':
        lines.append("# 0~1 범위로 스케일링")
        lines.append("X_train = X_train / 255.0")
        lines.append("X_test  = X_test  / 255.0")
    else:
        lines.append("# -1~1 범위로 스케일링")
        lines.append("X_train = X_train / 127.5 - 1.0")
        lines.append("X_test  = X_test  / 127.5 - 1.0")
    lines.append("")
    return "\n".join(lines)
