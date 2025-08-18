# confusion_matrix.py
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ConfusionMatrix:
    figure_size: Tuple[int, int] = (10, 8)  # 그래프 크기
    color_map: str = 'Blues'  # 색상 맵 ('Blues', 'Reds', 'Greens' 등)
    show_numbers: bool = True  # 숫자 표시 여부

def generate_confusion_matrix_code(cm_block: ConfusionMatrix) -> str:
    lines = []
    
    lines.append("# 혼동 행렬 보기")
    lines.append("from sklearn.metrics import confusion_matrix")
    lines.append("import matplotlib.pyplot as plt")
    lines.append("import seaborn as sns")
    lines.append("")
    lines.append("cm = confusion_matrix(y_true, y_pred)")
    lines.append(f"plt.figure(figsize={cm_block.figure_size})")
    
    if cm_block.show_numbers:
        lines.append(f"sns.heatmap(cm, annot=True, fmt='d', cmap='{cm_block.color_map}')")
    else:
        lines.append(f"sns.heatmap(cm, annot=False, cmap='{cm_block.color_map}')")
    
    lines.append("plt.xlabel('Predicted')")
    lines.append("plt.ylabel('Actual')")
    lines.append("plt.title('Confusion Matrix (PyTorch)')")
    lines.append("plt.show()")
    
    return "\n".join(lines) 