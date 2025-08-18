# blocks/Training/training_option.py

def generate_training_option_snippet(epochs, batch_size, patience):
    """
    ⑦ Training Option 블록: 학습 옵션 스니펫 생성
    epochs: int (반복횟수)
    batch_size: int (배치 크기)
    patience: int (조기 종료 대기 에폭)
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# -------[학습 옵션 블록]--------")
    lines.append("# -----------------------------")
    lines.append(f"# epochs={epochs}, batch_size={batch_size}, patience={patience}")
    lines.append(f"num_epochs = {epochs}        # 전체 학습 반복 횟수")
    lines.append(f"batch_size = {batch_size}     # 한 배치 크기")
    lines.append(f"patience = {patience}         # 조기 종료 전 대기 에폭 수")
    lines.append("")
    return "\n".join(lines)
