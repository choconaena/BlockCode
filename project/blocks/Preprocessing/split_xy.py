# blocks/split_xy.py

def generate_split_xy_snippet():
    """
    train_df / test_df 에서 입력(X)과 라벨(y)을 분리하고
    y를 Tensor로 변환하는 코드 블록 생성
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# ------[입력/라벨 분리 블록]------")
    lines.append("# -----------------------------")
    lines.append("import torch")
    lines.append("")
    lines.append("# 1) 학습 데이터(X_train, y_train) 분리")
    lines.append("X_train = train_df.iloc[:, 1:].values  # 학습용 입력 데이터 (NumPy 배열)")
    lines.append("y_train = train_df.iloc[:, 0].values     # 학습용 라벨 데이터 (NumPy 배열)")
    lines.append("y_train = torch.from_numpy(y_train).long()  # NumPy → LongTensor 변환")
    lines.append("")
    lines.append("# 2) 테스트 데이터(X_test, y_test) 분리")
    lines.append("X_test  = test_df.iloc[:, 1:].values   # 테스트용 입력 데이터 (NumPy 배열)")
    lines.append("y_test  = test_df.iloc[:, 0].values     # 테스트용 라벨 데이터 (NumPy 배열)")
    lines.append("y_test  = torch.from_numpy(y_test).long()   # NumPy → LongTensor 변환")
    lines.append("")
    return "\n".join(lines)
