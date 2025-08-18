# conv2d.py
from dataclasses import dataclass

@dataclass
class ActivationLayer:
    # 활성화 함수 구조체
    activation_type: str  # 'ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'Step'

def get_activation_function(activation_type: str) -> str:
    activation_map = {
        'ReLU': 'nn.ReLU()',
        'Sigmoid': 'nn.Sigmoid()',
        'Tanh': 'nn.Tanh()',
        'Softmax': 'nn.Softmax(dim=1)',
        'Step': 'nn.Hardtanh(min_val=0, max_val=1)'  
    }
    return activation_map.get(activation_type, 'nn.ReLU()') 

def generate_activation_code(activation_info: dict) -> str:
    """
    Convolution 2D 블록 생성
    """
    lines = []
    activation_type = activation_info['activation_type']
    activation_func = get_activation_function(activation_type)
    lines.append(f"\t\t\t" + f"{activation_func},                                      # {activation_type} 활성화 적용")
    
    return "\n".join(lines) 