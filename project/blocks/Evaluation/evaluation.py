# performance_evaluation.py
from dataclasses import dataclass
from typing import List

@dataclass
class PerformanceEvaluation:
    metrics: List[str]  # ['정확도', '손실값', 'Precision', 'Recall', 'F1-score']
    model_path: str = 'best_model.pth'  # 모델 파일 경로

def generate_performance_evaluation_code(eval_block: PerformanceEvaluation) -> str:
    lines = []
    
    lines.append("# 성능 평가하기")
    lines.append("from sklearn.metrics import classification_report")
    lines.append("")
    lines.append(f"model.load_state_dict(torch.load('{eval_block.model_path}'))")
    lines.append("model.eval()")
    lines.append("")
    lines.append("y_true, y_pred = [], []")
    lines.append("")
    lines.append("with torch.no_grad():")
    lines.append("    for X_batch, y_batch in test_loader:")
    lines.append("        X_batch = X_batch.to(device)")
    lines.append("        outputs = model(X_batch)")
    lines.append("        preds = outputs.argmax(dim=1).cpu().numpy()")
    lines.append("        y_pred.extend(preds)")
    lines.append("        y_true.extend(y_batch.numpy())")
    lines.append("")
    
    if '정확도' in eval_block.metrics or 'Precision' in eval_block.metrics or 'Recall' in eval_block.metrics or 'F1-score' in eval_block.metrics:
        lines.append("print(classification_report(y_true, y_pred))")
    
    return "\n".join(lines) 