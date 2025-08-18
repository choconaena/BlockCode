# -----------------------------
# --------- [학습하기 블록] ---------
# -----------------------------
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# ---------[손실함수 블록]---------
# -----------------------------
# 선택된 손실함수: CrossEntropy
criterion = nn.CrossEntropyLoss()  # 크로스엔트로피 손실 함수

# -----------------------------
# ---------[옵티마이저 블록]---------
# -----------------------------
# 선택된 옵티마이저: Adam, Learning Rate=1e-05
optimizer = optim.Adam(model.parameters(), lr=1e-05)

# -----------------------------
# -------[학습 옵션 블록]--------
# -----------------------------
# epochs=10, batch_size=64, patience=3
num_epochs = 10        # 전체 학습 반복 횟수
batch_size = 64     # 한 배치 크기
patience = 3         # 조기 종료 전 대기 에폭 수


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
