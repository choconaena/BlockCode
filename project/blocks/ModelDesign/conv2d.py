# conv2d.py

from dataclasses import dataclass

@dataclass
class Conv2DLayer:
    # 입력층 설정 구조체 (첫 번째 계층에만 필요)
    in_channels: int  # 입력 채널 수 (1: 흑백, 3: 컬러)
    # 이미지 특징 찾기 (Conv2D) 구조체
    out_channels: int  
    kernel_size: int  
    padding: int  

def generate_conv2d_code(conv_info: dict, is_first: bool = False) -> str:
    """
    Convolution 2D 블록 생성
    """
    lines = []
    layer_desc = "첫 번째 합성곱" if is_first else "두 번째 합성곱"
    
    lines.append(f"\t\t\t" + f"nn.Conv2d(                                     # {layer_desc}")
    lines.append(f"\t\t\t\t" + f"in_channels={conv_info['in_channels']},                            # 입력 채널")
    lines.append(f"\t\t\t\t" + f"out_channels={conv_info['out_channels']},                           # 출력 채널")
    lines.append(f"\t\t\t\t" + f"kernel_size={conv_info['kernel_size']},                             # 커널 크기")
    lines.append(f"\t\t\t\t" + f"padding={conv_info['padding']}                                  # 패딩")
    lines.append("\t\t\t\t" + "),")
    
    return "\n".join(lines) 