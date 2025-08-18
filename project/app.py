# app.py
# -----------------------------------------------------------------------------
# 이 파일은 Flask 웹앱의 진입점이다.
# - 좌측 블록 UI에서 전달된 폼 값을 받아 단계별 코드 스니펫을 생성한다.
# - 사용자별(workspace/<uid>) 작업 폴더에 코드와 입력 상태(JSON)를 저장한다.
# - dataset/ 폴더의 CSV를 읽어 데이터 요약/샘플/이미지 미리보기를 제공한다.
# - 각 단계 스크립트를 서버에서 실행하고, 로그를 SSE로 실시간 전송한다.
# -----------------------------------------------------------------------------

import os
import io
import json
import base64
import subprocess, threading, time          # ← 실행/로그용 추가
from uuid import uuid4                      # 사용자별 구분을 위한 랜덤 uid 발급
from pathlib import Path                    # 경로 연산을 안전하게 하기 위한 pathlib
from flask import (
    Flask, render_template, request, send_from_directory, jsonify,
    make_response, redirect, url_for, Response              # ← SSE용 Response 추가
)
import pandas as pd                         # CSV 로드/가공용
import numpy as np                          # (여기선 주로 이미지 처리/배열 변환 보조)
from PIL import Image                       # base64 PNG 생성용 (미리보기 이미지)

# ---- 블록 스니펫 생성기 임포트 ----------------------------------------------
# 각 단계별(start.py)에 정의된 "코드 생성 함수"를 불러온다.
# - generate_*_snippet(form): request.form(사용자 선택)을 바탕으로 코드 문자열 생성
# - calculate_connections: 전처리/입력 크기 등을 바탕으로 모델 차원 계산 보조
from blocks.Preprocessing.start import generate_preprocessing_snippet
from blocks.ModelDesign.start   import generate_modeldesign_snippet, calculate_connections
from blocks.Training.start      import generate_training_snippet
from blocks.Evaluation.start    import generate_evaluation_snippet

# ---- 경로 상수 ---------------------------------------------------------------
# BASE_DIR     : app.py가 있는 프로젝트 루트
# DATASET_DIR  : CSV 데이터셋을 두는 전용 폴더 (예: project/dataset/*.csv)
# WORKSPACE_DIR: 사용자별 작업 파일(.py / inputs_*.json)을 저장하는 폴더
BASE_DIR      = Path(__file__).resolve().parent
DATASET_DIR   = BASE_DIR / "dataset"
WORKSPACE_DIR = BASE_DIR / "workspace"
LOGS_DIR      = BASE_DIR / "logs"           # ← 사용자별 실행 로그 저장
LOGS_DIR.mkdir(exist_ok=True)

# Flask 앱 인스턴스 생성
# - template_folder="templates": Jinja 템플릿 위치
# - static_folder="static": 정적 파일(js/css/img) 위치
app = Flask(__name__, template_folder="templates", static_folder="static")

# -----------------------------------------------------------------------------
# 유틸: UID 쿠키/작업폴더 확보
# -----------------------------------------------------------------------------
def ensure_uid_and_workspace(resp=None):
    """
    현재 요청의 쿠키에서 uid를 읽어 사용자별 작업 폴더(workspace/<uid>)를 보장한다.
    - uid가 없으면 새로 생성(uuid4 hex)하고, README.txt를 1회 기록한다.
    - resp(Flask Response)가 주어지면, 여기에 uid 쿠키를 설정한다.
    반환: (uid, wdir Path)
    """
    uid = request.cookies.get("uid")   # 클라이언트 쿠키에서 uid 조회
    created = False                    # 새로 만들었는지 플래그

    if not uid:
        uid = uuid4().hex              # 랜덤 uid 발급
        created = True

    wdir = WORKSPACE_DIR / uid         # 사용자 전용 폴더
    wdir.mkdir(parents=True, exist_ok=True)

    # 최초 생성 시 안내 파일 기록(선택 사항)
    if created:
        (wdir / "README.txt").write_text(
            "이 폴더는 사용자별로 생성되는 작업 공간입니다.\n"
            "preprocessing.py / model.py / training.py / evaluation.py 가 여기에 순차 생성됩니다.\n",
            encoding="utf-8",
        )

    # 응답 객체가 있으면 uid 쿠키를 세팅한다(첫 방문자).
    # - httponly=True: JS에서 쿠키 접근 불가(보안 ↑)
    # - samesite="Lax": 크로스 사이트 요청에서 기본적으로 전송되지 않음
    if created and resp is not None:
        resp.set_cookie("uid", uid, httponly=True, samesite="Lax")

    return uid, wdir

def list_datasets():
    """
    dataset/ 폴더에 있는 *.csv 파일 목록을 정렬해 반환한다.
    - 폴더가 없다면 생성만 해두고 빈 리스트를 돌려준다.
    """
    DATASET_DIR.mkdir(exist_ok=True)
    return sorted([f.name for f in DATASET_DIR.glob("*.csv")])

# 단계 키 목록: 전처리(pre), 모델(model), 학습(train), 평가(eval)
STAGES = ("pre", "model", "train", "eval")

def _read_json(p: Path):
    """
    주어진 경로 p의 JSON 파일을 읽어 dict로 반환한다.
    - 파일이 없거나 파싱 실패 시 빈 dict 반환(에러 삼킴).
    """
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def load_all_form_state(uid: str) -> dict:
    """
    사용자 uid의 workspace에서 inputs_pre/model/train/eval.json을 모두 읽어
    하나의 dict로 병합해 반환한다.
    - 뒤에 읽는(나중 단계) 값이 동일 키를 덮어쓴다.
    - 템플릿에서 폼 초기값 복원에 사용.
    """
    wdir = WORKSPACE_DIR / uid
    merged = {}
    for st in STAGES:
        merged.update(_read_json(wdir / f"inputs_{st}.json"))
    return merged

def save_stage(uid, stage, code, form):
    """
    특정 stage에 대해:
    1) 생성된 코드 문자열을 workspace/<uid>/<stage>.py에 저장
    2) 현재 request.form의 값을 inputs_<stage>.json으로 저장(체크박스 등 복수값 보존)
       - form 매개변수는 현재 사용하지 않지만, 시그니처 호환성상 둠.
    """
    wdir = WORKSPACE_DIR / uid
    wdir.mkdir(exist_ok=True)

    # 단계별 파일명 매핑 후 코드 저장
    (wdir / {"pre":"preprocessing.py","model":"model.py","train":"training.py","eval":"evaluation.py"}[stage]) \
        .write_text(code or "", encoding="utf-8")

    # request.form.lists()를 사용해 같은 name의 복수값(체크박스/멀티셀렉트)을 보존
    form_dict = {}
    for k, vals in request.form.lists():
        # 값이 1개면 단일값으로, 2개 이상이면 리스트로 저장한다.
        form_dict[k] = vals if len(vals) > 1 else vals[0]

    # 사용자 입력을 JSON으로 직렬화하여 해당 단계 파일로 저장
    (wdir / f"inputs_{stage}.json").write_text(
        json.dumps(form_dict, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def read_snippet_files(uid):
    """
    현재까지 생성된 각 단계 코드 파일을 읽어 템플릿에 전달하기 위한 dict로 반환.
    - 파일이 없으면 빈 문자열("")을 넣는다.
    """
    wdir = WORKSPACE_DIR / uid

    def r(name):
        p = wdir / name
        return p.read_text(encoding="utf-8") if p.exists() else ""

    return {
        "snippet_pre":   r("preprocessing.py"),
        "snippet_model": r("model.py"),
        "snippet_train": r("training.py"),
        "snippet_eval":  r("evaluation.py"),
    }

# -----------------------------------------------------------------------------
# 라우팅
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """
    루트 접근 시 /app으로 리다이렉트한다.
    - 이때 ensure_uid_and_workspace를 호출하여 uid 쿠키와 작업폴더를 준비한다.
    """
    resp = make_response(redirect(url_for("index")))
    ensure_uid_and_workspace(resp)
    return resp

@app.route("/app", methods=["GET"])
def index():
    """
    메인 페이지 렌더링(GET 전용).
    - uid/workspace 보장
    - 이미 생성된 스니펫(각 단계 .py) 읽어서 우측 코드 탭에 표시
    - dataset/의 CSV 목록을 좌측 select에 표시
    - 사용자 입력 상태(여러 inputs_*.json)를 병합하여 폼 초기값 복원에 사용
    """
    # 쿠키/워크스페이스 보장 + 현재 스니펫 로딩
    resp = make_response()
    uid, _ = ensure_uid_and_workspace(resp)

    snippets   = read_snippet_files(uid)   # 각 단계 .py 내용
    options    = list_datasets()           # dataset/*.csv 파일명들
    form_state = load_all_form_state(uid)  # 이전 입력 값 병합 상태

    # 렌더 후 쿠키 포함 응답
    resp.set_data(render_template(
        "index.html",
        options=options,
        form_state=form_state,             # 템플릿에서 자바스크립트로 복원
        **snippets
    ))
    return resp

@app.route("/convert", methods=["POST"])
def convert_stage():
    """
    '변환하기' 요청 처리(POST).
    - stage 파라미터에 따라 지정된 단계만 코드 생성/저장한다. ("pre" | "model" | "train" | "eval" | "all")
    - 저장 후 최신 스니펫과 입력 상태를 다시 렌더링한다.
    """
    # 어떤 단계 변환? (폼의 버튼 name="stage" value=...)
    stage = request.form.get("stage")  # pre / model / train / eval / all

    # 사용자 uid/워크스페이스 확보
    uid, _ = ensure_uid_and_workspace()

    # 전처리 단계에서 리사이즈 정보를 모델 설계 계산에 활용하기 위한 보조 dict
    # - resize_n 폼 입력이 없으면 기본 28 사용
    preprocessing_info = {
        "resize_n": int(request.form.get("resize_n", 28)) if request.form.get("resize_n") else 28
    }

    # 단계별 코드 문자열 기본값(빈 문자열) 준비
    snippet_pre = snippet_model = snippet_train = snippet_eval = ""

    try:
        # 전처리 코드 생성/저장
        if stage in ("pre", "all"):
            snippet_pre = generate_preprocessing_snippet(request.form)
            save_stage(uid, "pre", snippet_pre, request.form)

        # 모델 설계 코드 생성/저장
        if stage in ("model", "all"):
            # 입력/전처리 정보를 바탕으로 채널/FC 입력크기 등을 계산(선택적)
            conn = calculate_connections(request.form, preprocessing_info)
            snippet_model = generate_modeldesign_snippet(request.form, conn)
            save_stage(uid, "model", snippet_model, request.form)

        # 학습 코드 생성/저장
        if stage in ("train", "all"):
            snippet_train = generate_training_snippet(request.form)
            save_stage(uid, "train", snippet_train, request.form)

        # 평가 코드 생성/저장
        if stage in ("eval", "all"):
            snippet_eval = generate_evaluation_snippet(request.form)
            save_stage(uid, "eval", snippet_eval, request.form)

    except Exception as e:
        # 생성 과정에서 예외가 발생하면 서버 콘솔에 출력(사용자 화면에는 조용히 실패)
        # TIP: traceback.print_exc()를 함께 쓰면 원인 분석에 더 도움된다.
        print(f"[ERROR] /convert: {e}", flush=True)

    # 변환 후 화면에 표시할 최신 스니펫/옵션/폼 상태를 다시 읽어 템플릿 렌더
    snippets   = read_snippet_files(uid)
    options    = list_datasets()
    form_state = load_all_form_state(uid)   # 저장 직후 상태로 복원

    return render_template("index.html", options=options, form_state=form_state, **snippets)

@app.route("/download/<stage>", methods=["GET"])
def download_stage(stage):
    """
    단계별 코드 파일 또는 전체 ZIP 다운로드 엔드포인트.
    - /download/pre   -> preprocessing.py
      /download/model -> model.py
      /download/train -> training.py
      /download/eval  -> evaluation.py
      /download/all   -> 위 파일들과 inputs_*.json, README.txt를 묶은 zip
    """
    uid, wdir = ensure_uid_and_workspace()

    # 단일 파일 매핑 테이블
    mapping = {
        "pre": "preprocessing.py",
        "model": "model.py",
        "train": "training.py",
        "eval": "evaluation.py",
        "all":  None
    }
    fname = mapping.get(stage)

    # 특정 단계 파일 다운로드
    if fname:
        return send_from_directory(wdir, fname, as_attachment=True)

    # stage=="all": zip 묶음 생성 후 다운로드
    import zipfile, time
    zipname = f"workspace_{uid}_{int(time.time())}.zip"
    zpath = wdir / zipname

    with zipfile.ZipFile(zpath, "w") as z:
        # 각 단계 코드 파일이 존재하면 zip에 포함
        for nm in ("preprocessing.py", "model.py", "training.py", "evaluation.py"):
            p = wdir / nm
            if p.exists():
                z.write(p, arcname=nm)

        # 단계별 입력 상태 JSON도 포함(있을 때만)
        for nm in ("inputs_pre.json","inputs_model.json","inputs_train.json","inputs_eval.json"):
            p = wdir / nm
            if p.exists():
                z.write(p, arcname=nm)

        # README 포함(있을 때만)
        readme = wdir / "README.txt"
        if readme.exists():
            z.write(readme, arcname="README.txt")

    return send_from_directory(wdir, zipname, as_attachment=True)

# -----------------------------------------------------------------------------
# 실행 & 로그 스트리밍 (SSE)
# -----------------------------------------------------------------------------
def _stage_file(stage: str) -> str:
    """스테이지 키를 실행 대상 파이썬 파일명으로 매핑"""
    return {
        "pre":   "preprocessing.py",
        "model": "model.py",
        "train": "training.py",
        "eval":  "evaluation.py",
    }.get(stage, "")

@app.route("/run/<stage>", methods=["POST"])
def run_stage(stage):
    """
    각 단계 스크립트를 서브프로세스로 실행한다.
    - stdout/stderr를 logs/<uid>_<stage>.log 에 실시간 기록
    - 스크립트에는 환경변수 AIB_WORKDIR로 사용자 워크스페이스 경로 제공
    """
    uid, wdir = ensure_uid_and_workspace()
    pyfile = _stage_file(stage)
    if not pyfile:
        return jsonify(error="unknown stage"), 400
    target = wdir / pyfile
    if not target.exists():
        return jsonify(error=f"{pyfile} not found"), 404

    log_path = LOGS_DIR / f"{uid}_{stage}.log"
    try:
        log_path.unlink()  # 이전 로그 제거
    except FileNotFoundError:
        pass

    env = os.environ.copy()
    env["AIB_WORKDIR"] = str(wdir)   # 스크립트가 저장/불러오기 경로로 사용

    # -u: unbuffered (버퍼링 없이 실시간 출력)
    proc = subprocess.Popen(
        ["python", "-u", str(target)],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )

    # stdout을 백그라운드에서 로그 파일로 흘려쓰기
    def pump():
        with open(log_path, "a", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
        proc.wait()

    threading.Thread(target=pump, daemon=True).start()
    return jsonify(ok=True)

@app.route("/logs/stream")
def logs_stream():
    """
    Server-Sent Events로 현재 사용자/스테이지의 로그를 스트리밍한다.
    쿼리: ?stage=pre|model|train|eval  (기본 train)
    """
    uid, _ = ensure_uid_and_workspace()
    stage = request.args.get("stage", "train")
    log_path = LOGS_DIR / f"{uid}_{stage}.log"

    def generate():
        last_size = 0
        while True:
            try:
                if log_path.exists():
                    with open(log_path, "r", encoding="utf-8") as f:
                        f.seek(last_size)
                        chunk = f.read()
                        if chunk:
                            # 파일 포인터 이동 기준(바이트) 누적
                            last_size += len(chunk.encode("utf-8"))
                            for line in chunk.splitlines():
                                yield f"data: {line}\n\n"
                time.sleep(0.3)
            except GeneratorExit:
                break
            except Exception as e:
                yield f"data: [stream-error] {e}\n\n"
                time.sleep(1)

    return Response(generate(), mimetype="text/event-stream")

# -----------------------------------------------------------------------------
# 데이터 요약/이미지 미리보기 엔드포인트
# -----------------------------------------------------------------------------
@app.route("/data-info", methods=["GET"])
def data_info():
    """
    dataset/<file> CSV를 읽고, type에 따라 간단한 정보를 JSON으로 반환한다.
    쿼리:
      - file: CSV 파일명(예: 'mnist_train.csv')  ※ dataset/ 내부만 허용
      - type: 'shape' | 'structure' | 'sample' | 'images'
      - n   : sample/images에서 보여줄 개수(정수). shape/structure에서는 무시.
    반환(JSON):
      - shape     -> {"rows": int, "cols": int}
      - structure -> {"columns": [{"name": str, "dtype": str}, ...]}
      - sample    -> {"columns": [col1, col2, ...], "sample": [[...], [...], ...]}
      - images    -> {"images": ["<base64 png>", ...]}
    """
    file = request.args.get("file","").strip()         # 클라이언트가 고른 CSV 파일명
    info_type = request.args.get("type","shape")       # 리턴할 정보 유형
    try:
        n = int(request.args.get("n","5"))             # 개수 파라미터(n) 파싱
    except:
        n = 5

    path = DATASET_DIR / file                          # dataset/<file> 경로 조합

    # 파일이 실제로 존재하지 않으면 404 JSON 응답
    if not path.exists():
        return jsonify(error="dataset file not found"), 404

    # CSV를 DataFrame으로 로드
    df = pd.read_csv(path)

    # 1) shape: (행, 열) 정보만 응답
    if info_type == "shape":
        return jsonify(rows=df.shape[0], cols=df.shape[1])

    # 2) structure: 각 컬럼의 이름/자료형(dtype 문자열)을 응답
    if info_type == "structure":
        cols = [{"name":c, "dtype":str(df[c].dtype)} for c in df.columns]
        return jsonify(columns=cols)

    # 3) sample: 상위 n행을 리스트로 변환하여 컬럼명과 함께 응답
    if info_type == "sample":
        sample = df.head(n).values.tolist()           # [[row], [row], ...] 형태
        return jsonify(columns=list(df.columns), sample=sample)

    # 4) images: MNIST 유사 포맷 가정(첫 컬럼이 라벨, 나머지가 28*28 픽셀(0~255))
    #    - 상위 n개 행에서 픽셀을 28x28로 reshape하여 PNG로 인코딩한 뒤 base64 문자열로 반환
    if info_type == "images":
        # 첫 열이 라벨, 나머지가 28x28 픽셀이라는 가정 (MNIST 형태)
        images = []
        for _, row in df.head(n).iterrows():
            # row.values: numpy 배열. [label, p0, p1, ..., p783]
            # values[1:]: 픽셀 값들만 추출 -> int -> 28x28 -> uint8
            arr = row.values[1:].astype(int).reshape(28,28).astype("uint8")

            # PIL Image로 만든 뒤 메모리 버퍼에 PNG 저장 -> base64 인코딩
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            images.append(base64.b64encode(buf.getvalue()).decode())

        return jsonify(images=images)

    # type이 위 네 가지에 해당하지 않는 경우 빈 JSON
    return jsonify({})

# -----------------------------------------------------------------------------
# 개발 서버 구동
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 필수 디렉터리 보장(없으면 생성)
    BASE_DIR.mkdir(exist_ok=True)
    DATASET_DIR.mkdir(exist_ok=True)
    WORKSPACE_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # 로컬 개발 서버 실행
    # - host="127.0.0.1": 로컬호스트에서만 접근 가능(외부 접속 불가)
    # - port=9098: 기존 프로젝트에서 쓰던 포트를 유지
    # - debug=True: 코드 변경 시 자동 리로드/디버그 핀(프로덕션에선 False 권장)
    # - use_reloader=False: 일부 환경에서 이중 실행/중복 로그를 막고 싶을 때 False
    app.run(host="127.0.0.1", port=9000, debug=True, use_reloader=False)