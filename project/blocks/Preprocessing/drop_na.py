# blocks/drop_na.py

def generate_drop_na_snippet():
    """
    train_df/test_df 에서 결측치 있는 행 제거
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# ------[빈 데이터 삭제 블록]------")
    lines.append("# -----------------------------")
    lines.append("train_df = train_df.dropna()  # 학습 데이터에서 NaN 포함 행 제거")
    lines.append("test_df  = test_df.dropna()   # 테스트 데이터에서 NaN 포함 행 제거")
    lines.append("")
    return "\n".join(lines)
