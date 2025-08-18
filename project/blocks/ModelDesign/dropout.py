# conv2d.py
from dataclasses import dataclass

@dataclass
class DropoutLayer:
    # 헷갈림 방지하기 (Dropout) 구조체
    p: float  # 드롭아웃 비율 (0.1 ~ 0.5)

def generate_dropout_code(dropout_info: dict) -> str:
    """
    Convolution 2D 블록 생성
    """
    lines = []
    lines.append(f"\t\t\t" + f"nn.Dropout(p={dropout_info['p']}),                             # 드롭아웃 ({dropout_info['p']})")
    
    return "\n".join(lines) 