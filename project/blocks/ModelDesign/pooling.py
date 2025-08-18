# conv2d.py
from dataclasses import dataclass

@dataclass
class PoolingLayer:
    # 특징 크기 줄이기 (Pooling) 구조체
    pool_type: str  # Max 또는 Avg
    size: int       # 풀링 크기

def generate_pooling_code(pool_info: dict) -> str:
    """
    Convolution 2D 블록 생성
    """
    lines = []
    lines.append(f"\t\t\t" + f"nn.{pool_info['pool_type']}Pool2d({pool_info['size']}, {pool_info['size']}),                        # {pool_info['pool_type']} 풀링")
    
    return "\n".join(lines) 