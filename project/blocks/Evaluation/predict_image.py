# prediction_images.py
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PredictionImages:
    num_samples: int = 10  # 보여줄 이미지 개수
    figure_size: Tuple[int, int] = (12, 5)  # 그래프 크기
    image_size: Tuple[int, int] = (28, 28)  # 이미지 크기
    color_map: str = 'gray'  # 색상 맵
    grid_rows: int = 2  # 격자 행 수
    grid_cols: int = 5  # 격자 열 수

def generate_prediction_images_code(pred_block: PredictionImages) -> str:
    lines = []
    
    lines.append("# 예측 결과 이미지 보기")
    lines.append("import random")
    lines.append("import matplotlib.pyplot as plt")
    lines.append("")
    lines.append(f"samples = random.sample(range(len(X_test_tensor)), {pred_block.num_samples})")
    lines.append("X_samples = X_test_tensor[samples].to(device)")
    lines.append("")
    lines.append("with torch.no_grad():")
    lines.append("    pred_samples = model(X_samples).argmax(dim=1).cpu().numpy()")
    lines.append("")
    lines.append(f"plt.figure(figsize={pred_block.figure_size})")
    lines.append("for i, idx in enumerate(samples):")
    lines.append(f"    plt.subplot({pred_block.grid_rows}, {pred_block.grid_cols}, i + 1)")
    lines.append(f"    plt.imshow(X_test_tensor[idx].reshape{pred_block.image_size}, cmap='{pred_block.color_map}')")
    lines.append("    plt.title(f\"실제: {y_test_tensor[idx].item()} / 예측: {pred_samples[i]}\")")
    lines.append("    plt.axis('off')")
    lines.append("")
    lines.append("plt.tight_layout()")
    lines.append("plt.show()")
    
    return "\n".join(lines) 