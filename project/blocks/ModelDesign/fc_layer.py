# conv2d.py
from dataclasses import dataclass

@dataclass
class FCLayer:
    # 분류 준비하기 (Dense) 구조체
    dense_input_size: int
    dense_output_size: int 

def generate_fc_layer_code(fc_info: dict) -> str:
    """
    Convolution 2D 블록 생성
    """
    lines = []
    lines.append("\t\t" + ")")
    lines.append("")
    lines.append("\t\t" + "# 완전 연결 계층 블록: 분류기")
    lines.append("\t\t" + "self.fc_layers = nn.Sequential(")
    lines.append("\t\t\t" + "nn.Flatten(),                                  # 다차원 텐서를 1차원으로 평탄화")
    lines.append(f"\t\t\t" + f"nn.Linear({fc_info['dense_input_size']}, {fc_info['dense_output_size']}),                    # FC")
    
    return "\n".join(lines) 