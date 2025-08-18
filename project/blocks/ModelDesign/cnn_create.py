# cnn_create.py

def generate_cnn_class_code():
    """
    CNN 클래스 생성
    """
    lines = []
    lines.append("# 혼동 행렬 보기")
    lines.append("import torch.nn as nn")
    lines.append("")
    lines.append("# CNN 모델 정의: nn.Module 상속")
    lines.append("class CNN(nn.Module):                                      # 사용자 정의 CNN 클래스")
    lines.append("\t" + "def __init__(self):                                    # 생성자 함수")
    lines.append("\t\t" + "super(CNN, self).__init__()                        # 부모 클래스 초기화")
    lines.append("")
    lines.append("\t\t" + "# 합성곱 계층 블록: 특성 맵 추출")
    lines.append("\t\t" + "self.conv_layers = nn.Sequential(")
    
    return "\n".join(lines) 