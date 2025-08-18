# blocks/Evaluation/start.py

def generate_evaluation_snippet(form):
    """
    평가 스니펫을 폼 파라미터에 따라 '능동적으로' 생성한다.
    - 데이터/모델 로드
    - 선택 메트릭(accuracy/precision/recall/f1/top-k/auc)
    - 선택 리포트/혼동행렬
    - 선택 시각화(예측 샘플/오분류 샘플)
    """
    # ---------- 폼 파라미터 수집 ----------
    get = form.get
    getlist = form.getlist

    metrics = set([m.strip().lower() for m in getlist("metrics")])  # {'accuracy','precision',...}
    show_report = bool(get("show_classification_report"))
    show_conf   = bool(get("show_confusion_matrix"))
    cm_normalize = (get("cm_normalize","").lower() in ("1","true","yes","on"))

    # 평균 방식 (precision/recall/f1)
    average = (get("average","macro") or "macro").lower()
    if average not in ("macro","micro","weighted","binary"):
        average = "macro"

    # top-k
    try:
        topk_k = int(get("topk_k","3"))
    except:
        topk_k = 3

    # 배치 크기
    try:
        eval_batch = int(get("eval_batch","128"))
    except:
        eval_batch = 128

    # 시각화 개수
    try:
        viz_samples = int(get("viz_samples","0") or 0)
    except:
        viz_samples = 0
    try:
        viz_mis = int(get("viz_mis","0") or 0)
    except:
        viz_mis = 0

    # 클래스/라벨 이름
    try:
        num_classes = int(get("num_classes","10"))
    except:
        num_classes = 10
    class_names_raw = (get("class_names","") or "").strip()
    has_class_names = bool(class_names_raw)
    # CPU 강제
    force_cpu = bool(get("force_cpu"))

    # ---------- 코드 조립 ----------
    L = []
    L += [
        "# -----------------------------",
        "# --------- [평가하기 블록] ---------",
        "# -----------------------------",
        "import os, io, json, math, torch",
        "import numpy as np",
        "from torch.utils.data import TensorDataset, DataLoader",
        "from model import build_model, load_model",
        "",
        "# 워크디렉토리/아티팩트 경로",
        "WORKDIR = os.environ.get('AIB_WORKDIR', '.')",
        "DATA_PATH = os.path.join(WORKDIR, 'data', 'dataset.pt')",
        "CKPT_PATH = os.path.join(WORKDIR, 'artifacts', 'best_model.pth')",
        "os.makedirs(os.path.join(WORKDIR, 'artifacts'), exist_ok=True)",
        "",
        "# 장치 선택",
        f"device = 'cpu' if {str(force_cpu)} else ('cuda' if torch.cuda.is_available() else 'cpu')",
        "print(f'[device] {device}')",
        "",
        "# 데이터 로드",
        "bundle = torch.load(DATA_PATH, map_location='cpu')",
        "X_test, y_test = bundle['X_test'], bundle['y_test']",
        "test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size="+str(eval_batch)+", shuffle=False)",
        "",
        "# 모델 로드",
        "model = build_model()",
        "model = load_model(model, CKPT_PATH, map_location=device)",
        "model.to(device).eval()",
        "",
        "# 추론: y_true / y_pred / y_proba(optional) 수집",
        "all_true, all_pred = [], []",
    ]

    # AUC / top-k가 있으면 확률/로짓 보관
    need_proba = ("auc" in metrics) or ("topk" in metrics)
    if need_proba:
        L += ["all_proba = []  # softmax 확률 보관"]

    L += [
        "with torch.no_grad():",
        "    for xb, yb in test_loader:",
        "        xb = xb.to(device)",
        "        logits = model(xb)",
        "        pred = logits.argmax(dim=1).cpu()",
        "        all_pred.extend(pred.tolist())",
        "        all_true.extend(yb.tolist())",
    ]
    if need_proba:
        L += [
            "        # 확률(소프트맥스)",
            "        prob = torch.softmax(logits, dim=1).cpu()",
            "        all_proba.extend(prob.numpy().tolist())",
        ]
    L += [
        "",
        "import numpy as np",
        "y_true = np.array(all_true, dtype=int)",
        "y_pred = np.array(all_pred, dtype=int)",
    ]
    if need_proba:
        L += ["y_proba = np.array(all_proba, dtype=float)"]

    # 클래스 이름 준비
    if has_class_names:
        L += [f"class_names = [s.strip() for s in '''{class_names_raw}'''.split(',') if s.strip()]"]
        L += ["if len(class_names) != "+str(num_classes)+":",
              "    print('[warn] class_names 개수가 num_classes와 다릅니다. 숫자 라벨로 대체합니다.')",
              "    class_names = [str(i) for i in range("+str(num_classes)+")]"]
    else:
        L += [f"class_names = [str(i) for i in range({num_classes})]"]

    # ------- 메트릭들 -------
    L += ["print('[eval] num_samples=', len(y_true))", ""]

    # accuracy/precision/recall/f1
    use_sklearn = any(m in metrics for m in ("accuracy","precision","recall","f1","auc")) or show_report or show_conf
    if use_sklearn:
        L += [
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score",
            "from sklearn.metrics import classification_report, confusion_matrix",
        ]

    if "accuracy" in metrics:
        L += [
            "acc = accuracy_score(y_true, y_pred)",
            "print(f'Accuracy: {acc:.4f}')"
        ]
    if "precision" in metrics:
        L += [
            f"prec = precision_score(y_true, y_pred, average='{average}', zero_division=0)",
            f"print(f'Precision({average}): {{prec:.4f}}')"
        ]
    if "recall" in metrics:
        L += [
            f"rec = recall_score(y_true, y_pred, average='{average}', zero_division=0)",
            f"print(f'Recall({average}): {{rec:.4f}}')"
        ]
    if "f1" in metrics:
        L += [
            f"f1 = f1_score(y_true, y_pred, average='{average}', zero_division=0)",
            f"print(f'F1({average}): {{f1:.4f}}')"
        ]

    # Top-K
    if "topk" in metrics:
        L += [
            "if len(y_pred) and "+("True" if need_proba else "False")+":",
            "    k = max(1, int("+str(topk_k)+"))",
            "    topk_correct = 0",
            "    for i in range(len(y_true)):",
            "        topk = np.argpartition(-y_proba[i], k-1)[:k]",
            "        if y_true[i] in topk:",
            "            topk_correct += 1",
            "    topk_acc = topk_correct / max(1, len(y_true))",
            "    print(f'Top-{"+str(topk_k)+"} Accuracy: {topk_acc:.4f}')",
        ]

    # AUC (이진)
    if "auc" in metrics:
        L += [
            "from sklearn.metrics import roc_auc_score",
            "try:",
            "    # 이진 분류 가정: 포지티브 클래스 1의 확률 사용",
            "    if y_proba.shape[1] == 2:",
            "        auc = roc_auc_score(y_true, y_proba[:,1])",
            "        print(f'AUC: {auc:.4f}')",
            "    else:",
            "        print('[warn] AUC는 이진 분류에서만 기본 지원합니다. 다중 분류는 one-vs-rest로 확장하세요.')",
            "except Exception as e:",
            "    print('[warn] AUC 계산 실패:', e)",
        ]

    # 리포트
    if show_report:
        L += [
            "print('[classification report]')",
            "print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))",
        ]

    # 혼동행렬
    if show_conf:
        L += [
            "cm = confusion_matrix(y_true, y_pred)",
            "print('[confusion matrix]\\n', cm)",
        ]
        # 이미지로 저장(선택)
        L += [
            "try:",
            "    import matplotlib.pyplot as plt",
            "    import numpy as np",
            "    fig, ax = plt.subplots(figsize=(6,5))",
            "    mat = cm.astype('float')",
            f"    if {str(cm_normalize)}:",
            "        mat = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)",
            "    im = ax.imshow(mat, cmap='Blues')",
            "    ax.set_title('Confusion Matrix'+(' (normalized)' if "+str(cm_normalize)+" else ''))",
            "    ax.set_xlabel('Predicted'); ax.set_ylabel('True')",
            "    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha='right')",
            "    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)",
            "    for i in range(mat.shape[0]):",
            "        for j in range(mat.shape[1]):",
            "            ax.text(j, i, f'{mat[i,j]:.2f}' if "+str(cm_normalize)+" else int(cm[i,j]),",
            "                    ha='center', va='center', color='black', fontsize=8)",
            "    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)",
            "    out_png = os.path.join(WORKDIR, 'artifacts', 'confusion_matrix.png')",
            "    plt.tight_layout(); fig.savefig(out_png); plt.close(fig)",
            "    print(f'[saved] confusion_matrix -> {out_png}')",
            "except Exception as e:",
            "    print('[warn] confusion matrix image save failed:', e)",
        ]

    # 시각화 (예측 샘플 / 오분류 샘플)
    viz_needed = (viz_samples > 0) or (viz_mis > 0)
    if viz_needed:
        L += [
            "try:",
            "    import matplotlib.pyplot as plt",
            "    import math",
            "    # 예측 샘플 그리드",
            f"    N = int({viz_samples})",
            "    if N > 0:",
            "        cols = min(10, max(1, int(math.sqrt(N))))",
            "        rows = math.ceil(N/cols)",
            "        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))",
            "        axes = np.array(axes).reshape(-1)",
            "        take = min(N, len(X_test))",
            "        for i in range(take):",
            "            img = X_test[i].cpu().numpy()",
            "            if img.ndim == 3 and img.shape[0] in (1,3):",
            "                # (C,H,W) -> (H,W) 또는 (H,W,C)",
            "                img = np.transpose(img, (1,2,0)) if img.shape[0]==3 else img[0]",
            "            axes[i].imshow(img, cmap='gray' if img.ndim==2 else None)",
            "            axes[i].set_title(f't:{y_true[i]} p:{y_pred[i]}', fontsize=9)",
            "            axes[i].axis('off')",
            "        for j in range(take, len(axes)): axes[j].axis('off')",
            "        out_png = os.path.join(WORKDIR, 'artifacts', 'samples.png')",
            "        plt.tight_layout(); fig.savefig(out_png); plt.close(fig)",
            "        print(f'[saved] samples -> {out_png}')",
            "",
            "    # 오분류 샘플 그리드",
            f"    M = int({viz_mis})",
            "    if M > 0:",
            "        wrong_idx = np.where(y_true != y_pred)[0].tolist()",
            "        take = min(M, len(wrong_idx))",
            "        if take > 0:",
            "            cols = min(10, max(1, int(math.sqrt(take))))",
            "            rows = math.ceil(take/cols)",
            "            fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))",
            "            axes = np.array(axes).reshape(-1)",
            "            for i in range(take):",
            "                idx = wrong_idx[i]",
            "                img = X_test[idx].cpu().numpy()",
            "                if img.ndim == 3 and img.shape[0] in (1,3):",
            "                    img = np.transpose(img, (1,2,0)) if img.shape[0]==3 else img[0]",
            "                axes[i].imshow(img, cmap='gray' if img.ndim==2 else None)",
            "                axes[i].set_title(f't:{y_true[idx]} p:{y_pred[idx]}', fontsize=9)",
            "                axes[i].axis('off')",
            "            for j in range(take, len(axes)): axes[j].axis('off')",
            "            out_png = os.path.join(WORKDIR, 'artifacts', 'misclassified.png')",
            "            plt.tight_layout(); fig.savefig(out_png); plt.close(fig)",
            "            print(f'[saved] misclassified -> {out_png}')",
            "        else:",
            "            print('[viz] 오분류 샘플 없음')",
            "except Exception as e:",
            "    print('[warn] visualization failed:', e)",
        ]

    L += ["print('[eval] done')"]
    return "\n".join(L)