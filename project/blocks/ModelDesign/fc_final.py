# conv2d.py
from dataclasses import dataclass

@dataclass
class OutputLayer:
    # 정답 예측층 (Output) 구조체
    num_classes: int  # 클래스 수
    dense_output_size: int

def generate_fc_final_code(output_info: dict) -> str:
    """
    Convolution 2D 블록 생성
    """
    lines = []
    lines.append(f"\t\t\t" + f"nn.Linear({output_info['dense_output_size']}, {output_info['num_classes']})                             # 최종 FC: {output_info['dense_output_size']} → {output_info['num_classes']} 클래스")
    lines.append("\t\t" + ")")

    return "\n".join(lines) 