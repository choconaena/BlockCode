# blocks/Training/start.py

from .loss            import generate_loss_snippet
from .optimizer       import generate_optimizer_snippet
from .training_option import generate_training_option_snippet

def generate_training_snippet(form):
    """
    form: request.form 딕셔너리
    → ⑤~⑦ 학습하기 단계 전체 스니펫 생성
    """
    # 1) form에서 파라미터 추출
    loss_method      = form.get('loss_method', '')
    optimizer_method = form.get('optimizer_method', '')
    learning_rate    = form.get('learning_rate', '')
    epochs           = form.get('epochs', '')
    batch_size       = form.get('batch_size', '')
    patience         = form.get('patience', '')

    lines = []
    # 학습 모듈 import
    lines.append("# -----------------------------")
    lines.append("# --------- [학습하기 블록] ---------")
    lines.append("# -----------------------------")
    lines.append("import torch.nn as nn")
    lines.append("import torch.optim as optim")
    lines.append("")

    # ⑤ 손실함수
    if loss_method:
        lines.append(generate_loss_snippet(loss_method))

    # ⑥ 옵티마이저
    if optimizer_method and learning_rate:
        lr = float(learning_rate)
        lines.append(generate_optimizer_snippet(optimizer_method, lr))

    # ⑦ 학습 옵션
    if epochs and batch_size and patience:
        e = int(epochs)
        bs = int(batch_size)
        p = int(patience)
        lines.append(generate_training_option_snippet(e, bs, p))
    # ==== 실행 가능 파트 추가 ====
    lines += [r"""
# ==== [데이터 로드/로더 구성] ====
import os, torch
from torch.utils.data import TensorDataset, DataLoader

WORKDIR = os.environ.get("AIB_WORKDIR", ".")
os.makedirs(os.path.join(WORKDIR, "artifacts"), exist_ok=True)
data_path = os.path.join(WORKDIR, "data", "dataset.pt")
bundle = torch.load(data_path, map_location="cpu")
X_train, y_train = bundle["X_train"], bundle["y_train"]
X_test,  y_test  = bundle["X_test"],  bundle["y_test"]

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# ==== [훈련 루프 + 얼리 스탑 + 체크포인트] ====
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

# model.py 의 build_model() 사용
from model import build_model
model = build_model().to(device)

best_val = float("inf")
es_cnt   = 0
ckpt_path = os.path.join(WORKDIR, "artifacts", "best_model.pth")

for epoch in range(1, num_epochs + 1):
    # ---- train ----
    model.train()
    running = 0.0
    for xb, yb in tqdm(train_loader, desc=f"train {epoch}/{num_epochs}", ncols=80):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running += loss.item()
    train_loss = running / max(1, len(train_loader))

    # ---- val ----
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc=f"valid {epoch}/{num_epochs}", ncols=80):
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total   += yb.size(0)
    val_loss /= max(1, len(test_loader))
    val_acc   = correct / max(1, total)

    print(f"[epoch {epoch}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        es_cnt = 0
        torch.save(model.state_dict(), ckpt_path)
        print(f"[checkpoint] saved -> {ckpt_path}")
    else:
        es_cnt += 1
        print(f"[early-stopping] no improve ({es_cnt}/{patience})")
        if es_cnt >= patience:
            print("[early-stopping] stop training")
            break

print("[train] done")
"""]
    
    return "\n".join(lines)
