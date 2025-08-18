# blocks/Training/loss.py

def generate_loss_snippet(loss_method):
    """
    ⑤ Loss 블록: 손실함수 선택 스니펫 생성
    loss_method: 'CrossEntropy' 또는 'MSE'
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# ---------[손실함수 블록]---------")
    lines.append("# -----------------------------")
    lines.append(f"# 선택된 손실함수: {loss_method}")
    if loss_method == 'CrossEntropy':
        lines.append("criterion = nn.CrossEntropyLoss()  # 크로스엔트로피 손실 함수")
    elif loss_method == 'MSE':
        lines.append("criterion = nn.MSELoss()  # Mean Squared Error 손실 함수")
    lines.append("")
    return "\n".join(lines)
