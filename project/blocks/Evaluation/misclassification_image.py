# misclassified_images.py
from dataclasses import dataclass
from typing import Tuple

@dataclass
class MisclassifiedImages:
    max_show: int = 5  # 최대 보여줄 이미지 개수
    figure_size_per_image: int = 3  # 이미지당 그래프 크기
    image_size: Tuple[int, int] = (28, 28)  # 이미지 크기
    color_map: str = 'gray'  # 색상 맵
    title: str = "잘못 예측한 이미지"  # 제목

def generate_misclassified_images_code(mis_block: MisclassifiedImages) -> str:
    lines = []
    
    lines.append("# 잘못 예측한 이미지 모아보기")
    lines.append("import numpy as np")
    lines.append("import matplotlib.pyplot as plt")
    lines.append("")
    lines.append("mis_idx = np.where(np.array(y_true) != np.array(y_pred))[0]")
    lines.append("")
    lines.append("if len(mis_idx) == 0:")
    lines.append("    print(\"잘못 예측한 이미지가 없습니다.\")")
    lines.append("else:")
    lines.append(f"    n_show = min({mis_block.max_show}, len(mis_idx))")
    lines.append("    pick = np.random.choice(mis_idx, n_show, replace=False)")
    lines.append("")
    lines.append(f"    fig, axes = plt.subplots(1, n_show, figsize=({mis_block.figure_size_per_image} * n_show, {mis_block.figure_size_per_image}))")
    lines.append("    for ax, idx in zip(axes, pick):")
    lines.append(f"        ax.imshow(X_test_tensor[idx].reshape{mis_block.image_size}, cmap='{mis_block.color_map}')")
    lines.append("        ax.set_title(f\"GT:{y_test_tensor[idx].item()}  /  P:{y_pred[idx]}\")")
    lines.append("        ax.axis('off')")
    lines.append(f"    plt.suptitle(\"{mis_block.title}\")")
    lines.append("    plt.tight_layout()")
    lines.append("    plt.show()")
    
    return "\n".join(lines) 