# blocks/Training/optimizer.py

def generate_optimizer_snippet(method, lr):
    """
    ⑥ Optimizer 블록: 옵티마이저 설정 스니펫 생성
    method: 'Adam', 'SGD' 또는 'RMSprop'
    lr: float (learning rate)
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# ---------[옵티마이저 블록]---------")
    lines.append("# -----------------------------")
    lines.append(f"# 선택된 옵티마이저: {method}, Learning Rate={lr}")
    if method == 'Adam':
        lines.append(f"optimizer = optim.Adam(model.parameters(), lr={lr})")
    elif method == 'SGD':
        lines.append(f"optimizer = optim.SGD(model.parameters(), lr={lr})")
    elif method == 'RMSprop':
        lines.append(f"optimizer = optim.RMSprop(model.parameters(), lr={lr})")
    lines.append("")
    return "\n".join(lines)
