# -----------------------------
# --------- [평가하기 블록] ---------
# -----------------------------
import os, io, json, math, torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import build_model, load_model

# 워크디렉토리/아티팩트 경로
WORKDIR = os.environ.get('AIB_WORKDIR', '.')
DATA_PATH = os.path.join(WORKDIR, 'data', 'dataset.pt')
CKPT_PATH = os.path.join(WORKDIR, 'artifacts', 'best_model.pth')
os.makedirs(os.path.join(WORKDIR, 'artifacts'), exist_ok=True)

# 장치 선택
device = 'cpu' if False else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[device] {device}')

# 데이터 로드
bundle = torch.load(DATA_PATH, map_location='cpu')
X_test, y_test = bundle['X_test'], bundle['y_test']
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)

# 모델 로드
model = build_model()
model = load_model(model, CKPT_PATH, map_location=device)
model.to(device).eval()

# 추론: y_true / y_pred / y_proba(optional) 수집
all_true, all_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu()
        all_pred.extend(pred.tolist())
        all_true.extend(yb.tolist())

import numpy as np
y_true = np.array(all_true, dtype=int)
y_pred = np.array(all_pred, dtype=int)
class_names = [str(i) for i in range(10)]
print('[eval] num_samples=', len(y_true))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
acc = accuracy_score(y_true, y_pred)
print(f'Accuracy: {acc:.4f}')
print('[classification report]')
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
cm = confusion_matrix(y_true, y_pred)
print('[confusion matrix]\n', cm)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(6,5))
    mat = cm.astype('float')
    if False:
        mat = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    im = ax.imshow(mat, cmap='Blues')
    ax.set_title('Confusion Matrix'+(' (normalized)' if False else ''))
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat[i,j]:.2f}' if False else int(cm[i,j]),
                    ha='center', va='center', color='black', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_png = os.path.join(WORKDIR, 'artifacts', 'confusion_matrix.png')
    plt.tight_layout(); fig.savefig(out_png); plt.close(fig)
    print(f'[saved] confusion_matrix -> {out_png}')
except Exception as e:
    print('[warn] confusion matrix image save failed:', e)
try:
    import matplotlib.pyplot as plt
    import math
    # 예측 샘플 그리드
    N = int(10)
    if N > 0:
        cols = min(10, max(1, int(math.sqrt(N))))
        rows = math.ceil(N/cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        axes = np.array(axes).reshape(-1)
        take = min(N, len(X_test))
        for i in range(take):
            img = X_test[i].cpu().numpy()
            if img.ndim == 3 and img.shape[0] in (1,3):
                # (C,H,W) -> (H,W) 또는 (H,W,C)
                img = np.transpose(img, (1,2,0)) if img.shape[0]==3 else img[0]
            axes[i].imshow(img, cmap='gray' if img.ndim==2 else None)
            axes[i].set_title(f't:{y_true[i]} p:{y_pred[i]}', fontsize=9)
            axes[i].axis('off')
        for j in range(take, len(axes)): axes[j].axis('off')
        out_png = os.path.join(WORKDIR, 'artifacts', 'samples.png')
        plt.tight_layout(); fig.savefig(out_png); plt.close(fig)
        print(f'[saved] samples -> {out_png}')

    # 오분류 샘플 그리드
    M = int(5)
    if M > 0:
        wrong_idx = np.where(y_true != y_pred)[0].tolist()
        take = min(M, len(wrong_idx))
        if take > 0:
            cols = min(10, max(1, int(math.sqrt(take))))
            rows = math.ceil(take/cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
            axes = np.array(axes).reshape(-1)
            for i in range(take):
                idx = wrong_idx[i]
                img = X_test[idx].cpu().numpy()
                if img.ndim == 3 and img.shape[0] in (1,3):
                    img = np.transpose(img, (1,2,0)) if img.shape[0]==3 else img[0]
                axes[i].imshow(img, cmap='gray' if img.ndim==2 else None)
                axes[i].set_title(f't:{y_true[idx]} p:{y_pred[idx]}', fontsize=9)
                axes[i].axis('off')
            for j in range(take, len(axes)): axes[j].axis('off')
            out_png = os.path.join(WORKDIR, 'artifacts', 'misclassified.png')
            plt.tight_layout(); fig.savefig(out_png); plt.close(fig)
            print(f'[saved] misclassified -> {out_png}')
        else:
            print('[viz] 오분류 샘플 없음')
except Exception as e:
    print('[warn] visualization failed:', e)
print('[eval] done')