# blocks/Preprocessing/data_selection.py

def generate_data_selection_snippet(dataset, is_test, testdataset, a):
    """
    train_df / test_df 로드 및 분할 스니펫 생성
    - dataset 디렉터리 사용
    - is_test='true'면 testdataset도 dataset/에서 읽음
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# ---------[데이터 선택 블록]---------")
    lines.append("# -----------------------------")
    lines.append(f"train_df = pd.read_csv('dataset/{dataset}')  # '{dataset}' 파일에서 학습용 데이터 로드")
    lines.append("")
    if is_test == 'true' and testdataset:
        lines.append(f"test_df  = pd.read_csv('dataset/{testdataset}')  # 사용자가 지정한 테스트 데이터 로드")
    else:
        a_int = int(a) if str(a).isdigit() else 80
        lines.append(f"# 테스트 미지정 → 학습 데이터를 {a_int}% 사용, 나머지 {100-a_int}%를 테스트로 분할")
        lines.append(f"test_df  = train_df.sample(frac={(100-a_int)/100.0}, random_state=42)")
        lines.append("train_df = train_df.drop(test_df.index).reset_index(drop=True)")
        lines.append("test_df  = test_df.reset_index(drop=True)")
    lines.append("")
    return "\n".join(lines)