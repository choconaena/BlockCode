# blocks/drop_bad_labels.py

def generate_drop_bad_labels_snippet(min_label, max_label):
    """
    train_df/test_df 에서 라벨이 지정한 범위(min_label~max_label) 밖에 있는 행을 제거하는 코드 블록 생성
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# --[잘못된 라벨 삭제 블록]--")
    lines.append("# -----------------------------")
    lines.append(f"# 라벨값 허용 범위: {min_label} ~ {max_label}")
    lines.append(f"train_df = train_df[train_df['label'].between({min_label}, {max_label})]  # 학습 데이터 필터링")
    lines.append(f"test_df  = test_df[test_df['label'].between({min_label}, {max_label})]   # 테스트 데이터 필터링")
    lines.append("")
    return "\n".join(lines)
