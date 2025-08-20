# app.py - CORS 지원 추가
# -----------------------------------------------------------------------------
# AI 블록코딩 Flask 애플리케이션
# - 명확한 라우팅 구조
# - 체계적인 에러 처리
# - 클린한 함수 분리
# - CORS 지원 추가
# -----------------------------------------------------------------------------

import os
import json
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4
from flask import (
    Flask, render_template, request, send_from_directory, 
    jsonify, make_response, redirect, url_for, Response
)
from flask_cors import CORS  # 🔹 CORS 추가

# ===== 설정 상수 =====
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
WORKSPACE_DIR = BASE_DIR / "workspace"
LOGS_DIR = BASE_DIR / "logs"

# Flask 앱 초기화
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 업로드 제한

# 🔹 CORS 설정
CORS(app, 
     origins=["http://localhost:5173", 
              "https://4th-security-cube-ai-fe.vercel.app", 
              "http://localhost:5174", 
              "http://localhost:9022", 
              "http://localhost:9000"],
     supports_credentials=True,  # 🔹 쿠키/인증 헤더 허용
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)

# ===== 코드 생성 모듈 임포트 =====
from generators.preprocessing import PreprocessingGenerator
from generators.model import ModelGenerator
from generators.training import TrainingGenerator
from generators.evaluation import EvaluationGenerator

# ===== 유틸리티 함수 =====
# [기존 클래스들은 동일하므로 생략...]

class WorkspaceManager:
    """사용자 워크스페이스 관리"""
    
    @staticmethod
    def get_or_create_uid(response=None):
        """UID 가져오기 또는 생성"""
        # 🔹 쿠키 대신 요청 파라미터에서 user_id 가져오기
        uid = request.form.get("user_id") or request.args.get("user_id")
        created = False
        
        if not uid:
            # 🔹 아이디가 없으면 기본값 설정 (또는 에러 처리)
            uid = "anonymous"
            created = True
        
        # 🔹 안전한 파일명으로 변환 (특수문자 제거)
        import re
        uid = re.sub(r'[^a-zA-Z0-9_-]', '_', uid)[:50]  # 최대 50자, 안전한 문자만
        
        workspace_path = WORKSPACE_DIR / uid
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        if created:
            readme_path = workspace_path / "README.txt"
            readme_path.write_text(
                "AI 블록코딩 워크스페이스\n"
                f"생성일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"User ID: {uid}\n",
                encoding="utf-8"
            )
            
            (workspace_path / "data").mkdir(exist_ok=True)
            (workspace_path / "artifacts").mkdir(exist_ok=True)
            (workspace_path / "logs").mkdir(exist_ok=True)
            
            # 🔹 쿠키 설정 제거 (더 이상 필요 없음)
            # if response:
            #     response.set_cookie("uid", uid, httponly=True, samesite="Lax")
        
        return uid, workspace_path
    
    @staticmethod
    def save_code(uid, stage, code):
        """생성된 코드 저장"""
        workspace_path = WORKSPACE_DIR / uid
        
        filename_map = {
            "pre": "preprocessing.py",
            "model": "model.py",
            "train": "training.py",
            "eval": "evaluation.py"
        }
        
        if stage in filename_map:
            file_path = workspace_path / filename_map[stage]
            file_path.write_text(code, encoding="utf-8")
    
    @staticmethod
    def save_inputs(uid, stage, form_data):
        """폼 입력값 저장"""
        workspace_path = WORKSPACE_DIR / uid
        
        inputs = {}
        for key in form_data.keys():
            values = form_data.getlist(key)
            inputs[key] = values if len(values) > 1 else values[0] if values else ""
        
        json_path = workspace_path / f"inputs_{stage}.json"
        json_path.write_text(
            json.dumps(inputs, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    @staticmethod
    def load_inputs(uid, stage=None):
        """저장된 입력값 로드"""
        workspace_path = WORKSPACE_DIR / uid
        
        if stage:
            json_path = workspace_path / f"inputs_{stage}.json"
            if json_path.exists():
                try:
                    return json.loads(json_path.read_text(encoding="utf-8"))
                except:
                    return {}
        else:
            merged = {}
            for s in ["pre", "model", "train", "eval"]:
                json_path = workspace_path / f"inputs_{s}.json"
                if json_path.exists():
                    try:
                        data = json.loads(json_path.read_text(encoding="utf-8"))
                        merged.update(data)
                    except:
                        pass
            return merged
    
    @staticmethod
    def load_snippets(uid):
        """모든 코드 스니펫 로드"""
        workspace_path = WORKSPACE_DIR / uid
        
        snippets = {}
        file_map = {
            "snippet_pre": "preprocessing.py",
            "snippet_model": "model.py",
            "snippet_train": "training.py",
            "snippet_eval": "evaluation.py"
        }
        
        for key, filename in file_map.items():
            file_path = workspace_path / filename
            snippets[key] = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
        
        return snippets


class DatasetManager:
    """데이터셋 관리"""
    
    @staticmethod
    def list_datasets():
        """사용 가능한 데이터셋 목록"""
        DATASET_DIR.mkdir(exist_ok=True)
        return sorted([f.name for f in DATASET_DIR.glob("*.csv")])
    
    @staticmethod
    def get_dataset_info(filename, info_type="shape", n=5):
        """데이터셋 정보 조회"""
        import pandas as pd
        import numpy as np
        from PIL import Image
        import io
        import base64
        
        file_path = DATASET_DIR / filename
        if not file_path.exists():
            return None
        
        df = pd.read_csv(file_path)
        
        if info_type == "shape":
            return {"rows": df.shape[0], "cols": df.shape[1]}
        
        elif info_type == "structure":
            columns = [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]
            return {"columns": columns}
        
        elif info_type == "sample":
            sample_data = df.head(n).values.tolist()
            return {"columns": list(df.columns), "sample": sample_data}
        
        elif info_type == "images":
            images = []
            for _, row in df.head(n).iterrows():
                pixels = row.values[1:].astype(int).reshape(28, 28).astype("uint8")
                
                img = Image.fromarray(pixels)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                images.append(img_base64)
            
            return {"images": images}
        
        return {}


class CodeGenerator:
    """코드 생성 관리"""
    
    def __init__(self):
        self.generators = {
            "pre": PreprocessingGenerator(),
            "model": ModelGenerator(),
            "train": TrainingGenerator(),
            "eval": EvaluationGenerator()
        }
    
    def generate(self, stage, form_data):
        """특정 스테이지의 코드 생성"""
        if stage not in self.generators:
            raise ValueError(f"Unknown stage: {stage}")
        
        return self.generators[stage].generate(form_data)


class ProcessManager:
    """프로세스 실행 관리"""
    
    @staticmethod
    def run_script(uid, stage):
        """스크립트 실행"""
        workspace_path = WORKSPACE_DIR / uid
        
        script_map = {
            "pre": "preprocessing.py",
            "model": "model.py",
            "train": "training.py",
            "eval": "evaluation.py"
        }
        
        if stage not in script_map:
            return {"error": "Unknown stage"}
        
        script_path = workspace_path / script_map[stage]
        if not script_path.exists():
            return {"error": f"{script_map[stage]} not found"}
        
        log_path = LOGS_DIR / f"{uid}_{stage}.log"
        log_path.unlink(missing_ok=True)
        
        env = os.environ.copy()
        env["AIB_WORKDIR"] = str(workspace_path)
        
        proc = subprocess.Popen(
            ["python", "-u", str(script_path)],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1
        )
        
        def stream_logs():
            with open(log_path, "a", encoding="utf-8") as f:
                for line in proc.stdout:
                    f.write(line)
                    f.flush()
            proc.wait()
        
        threading.Thread(target=stream_logs, daemon=True).start()
        
        return {"ok": True, "pid": proc.pid}
    
    @staticmethod
    def stream_logs(uid, stage):
        """로그 스트리밍 (SSE)"""
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
                                last_size += len(chunk.encode("utf-8"))
                                for line in chunk.splitlines():
                                    yield f"data: {line}\n\n"
                    time.sleep(0.3)
                except GeneratorExit:
                    break
                except Exception as e:
                    yield f"data: [error] {e}\n\n"
                    time.sleep(1)
        
        response = Response(generate(), mimetype="text/event-stream")
        
        # 🔹 SSE에 CORS 헤더 수동 추가
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        
        return response


# ===== 라우트 =====

@app.route("/")
def home():
    """홈 페이지 - /app으로 리다이렉트"""
    # resp = make_response(redirect(url_for("main_app")))
    # WorkspaceManager.get_or_create_uid(resp)
    # return resp
    return redirect(url_for("main_app"))

'''
@app.route("/app")
def main_app():
    """메인 애플리케이션 페이지"""
    resp = make_response()
    uid, _ = WorkspaceManager.get_or_create_uid(resp)
    
    context = {
        "options": DatasetManager.list_datasets(),
        "form_state": WorkspaceManager.load_inputs(uid),
        **WorkspaceManager.load_snippets(uid)
    }
    
    resp.set_data(render_template("index.html", **context))
    return resp
'''

@app.route("/app")
def main_app():
    """메인 애플리케이션 페이지"""
    # 🔹 user_id 파라미터로 UID 가져오기
    user_id = request.args.get("user_id", "anonymous")
    
    # 안전한 파일명으로 변환
    import re
    uid = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)[:50]
    
    workspace_path = WORKSPACE_DIR / uid
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    context = {
        "options": DatasetManager.list_datasets(),
        "form_state": WorkspaceManager.load_inputs(uid),
        "current_user_id": user_id,  # 🔹 템플릿에 전달
        **WorkspaceManager.load_snippets(uid)
    }
    
    return render_template("index.html", **context)

'''
@app.route("/convert", methods=["POST"])
def convert():
    """코드 변환 API - 텍스트 응답 + CORS 지원"""
    uid, _ = WorkspaceManager.get_or_create_uid()
    stage = request.form.get("stage", "all")
    
    code_gen = CodeGenerator()
    
    try:
        if stage == "all":
            all_codes = {}
            for s in ["pre", "model", "train", "eval"]:
                code = code_gen.generate(s, request.form)
                WorkspaceManager.save_code(uid, s, code)
                WorkspaceManager.save_inputs(uid, s, request.form)
                all_codes[s] = code
            
            combined_code = ""
            stage_names = {
                "pre": "전처리 (preprocessing.py)",
                "model": "모델 설계 (model.py)", 
                "train": "학습 (training.py)",
                "eval": "평가 (evaluation.py)"
            }
            
            for s in ["pre", "model", "train", "eval"]:
                combined_code += f"# ========== {stage_names[s]} ==========\n\n"
                combined_code += all_codes[s]
                combined_code += "\n\n"
            
            response = make_response(combined_code)
            response.headers['Content-Type'] = 'text/plain; charset=utf-8'
            
            # 🔹 수동 CORS 헤더 추가 (보완)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
            
            return response
            
        else:
            code = code_gen.generate(stage, request.form)
            WorkspaceManager.save_code(uid, stage, code)
            WorkspaceManager.save_inputs(uid, stage, request.form)
            
            response = make_response(code)
            response.headers['Content-Type'] = 'text/plain; charset=utf-8'
            
            # 🔹 수동 CORS 헤더 추가 (보완)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
            
            return response
    
    except Exception as e:
        print(f"[ERROR] /convert: {e}")
        error_msg = f"코드 생성 중 오류 발생: {str(e)}"
        response = make_response(error_msg, 500)
        response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        
        # 🔹 에러 응답에도 CORS 헤더 추가
        response.headers['Access-Control-Allow-Origin'] = '*'
        
        return response
'''

@app.route("/convert", methods=["POST"])
def convert():
    """코드 변환 API - 사용자 ID 기반"""
    uid, _ = WorkspaceManager.get_or_create_uid()
    stage = request.form.get("stage", "all")
    
    # 🔹 사용자 ID 검증
    if uid == "anonymous" and not request.form.get("user_id"):
        error_msg = "사용자 ID가 필요합니다."
        response = make_response(error_msg, 400)
        response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    
    print(f"[INFO] /convert 요청: stage={stage}, user_id={uid}")
    
    code_gen = CodeGenerator()
    
    try:
        if stage == "all":
            all_codes = {}
            for s in ["pre", "model", "train", "eval"]:
                code = code_gen.generate(s, request.form)
                WorkspaceManager.save_code(uid, s, code)
                WorkspaceManager.save_inputs(uid, s, request.form)
                all_codes[s] = code
            
            combined_code = ""
            stage_names = {
                "pre": "전처리 (preprocessing.py)",
                "model": "모델 설계 (model.py)", 
                "train": "학습 (training.py)",
                "eval": "평가 (evaluation.py)"
            }
            
            for s in ["pre", "model", "train", "eval"]:
                combined_code += f"# ========== {stage_names[s]} ==========\n\n"
                combined_code += all_codes[s]
                combined_code += "\n\n"
            
            response = make_response(combined_code)
            response.headers['Content-Type'] = 'text/plain; charset=utf-8'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
            
            return response
            
        else:
            code = code_gen.generate(stage, request.form)
            WorkspaceManager.save_code(uid, stage, code)
            WorkspaceManager.save_inputs(uid, stage, request.form)
            
            response = make_response(code)
            response.headers['Content-Type'] = 'text/plain; charset=utf-8'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
            
            return response
    
    except Exception as e:
        print(f"[ERROR] /convert: {e}")
        error_msg = f"코드 생성 중 오류 발생: {str(e)}"
        response = make_response(error_msg, 500)
        response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        response.headers['Access-Control-Allow-Origin'] = '*'
        
        return response

# 🔹 OPTIONS 메서드 수동 처리 (preflight)
@app.route("/convert", methods=["OPTIONS"])
def convert_options():
    """CORS preflight 요청 처리"""
    response = make_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '86400'  # 24시간 캐시
    return response

@app.route("/run/<stage>", methods=["POST"])
def run_stage(stage):
    """코드 실행 API - 사용자 ID 기반"""
    # 🔹 여러 방식으로 사용자 ID 받기
    user_id = None
    
    # 1. POST form-data에서
    if request.form.get("user_id"):
        user_id = request.form.get("user_id")
    # 2. JSON에서
    elif request.is_json and request.json and request.json.get("user_id"):
        user_id = request.json.get("user_id")
    # 3. 쿼리 파라미터에서
    elif request.args.get("user_id"):
        user_id = request.args.get("user_id")
    
    print(f"[DEBUG] /run/{stage} - 받은 user_id: '{user_id}'")
    print(f"[DEBUG] request.form: {dict(request.form)}")
    print(f"[DEBUG] request.args: {dict(request.args)}")
    if request.is_json:
        print(f"[DEBUG] request.json: {request.json}")
    
    if not user_id or user_id.strip() == "":
        return jsonify({"error": "user_id가 필요합니다."}), 400
    
    # 안전한 파일명으로 변환
    import re
    uid = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id.strip())[:50]
    
    print(f"[DEBUG] 변환된 uid: '{uid}'")
    
    result = ProcessManager.run_script(uid, stage)
    
    if "error" in result:
        return jsonify(result), 400
    
    return jsonify(result)


@app.route("/logs/stream")
def logs_stream():
    """로그 스트리밍 API (SSE) - 사용자 ID 기반"""
    # 🔹 쿼리 파라미터에서 사용자 ID 받기
    user_id = request.args.get("user_id")
    stage = request.args.get("stage", "train")
    
    if not user_id:
        def error_generator():
            yield "data: [error] user_id가 필요합니다.\n\n"
        
        response = Response(error_generator(), mimetype="text/event-stream")
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    
    # 안전한 파일명으로 변환
    import re
    uid = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)[:50]
    
    return ProcessManager.stream_logs(uid, stage)


@app.route("/download/<stage>")
def download(stage):
    """생성된 코드 다운로드 - 사용자 ID 기반"""
    # 🔹 쿼리 파라미터에서 사용자 ID 받기
    user_id = request.args.get("user_id")
    
    if not user_id:
        return jsonify({"error": "user_id가 필요합니다."}), 400
    
    # 안전한 파일명으로 변환
    import re
    uid = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)[:50]
    
    workspace_path = WORKSPACE_DIR / uid
    
    file_map = {
        "pre": "preprocessing.py",
        "model": "model.py",
        "train": "training.py",
        "eval": "evaluation.py"
    }
    
    if stage in file_map:
        file_path = workspace_path / file_map[stage]
        if not file_path.exists():
            return jsonify({"error": f"{file_map[stage]} 파일이 없습니다. 먼저 코드를 생성해주세요."}), 404
        
        return send_from_directory(workspace_path, file_map[stage], as_attachment=True)
    
    elif stage == "all":
        # ZIP 파일 생성
        import zipfile
        zip_name = f"workspace_{uid}_{int(time.time())}.zip"
        zip_path = workspace_path / zip_name
        
        with zipfile.ZipFile(zip_path, "w") as zf:
            for filename in file_map.values():
                file_path = workspace_path / filename
                if file_path.exists():
                    zf.write(file_path, arcname=filename)
            
            # inputs 파일들도 포함
            for s in ["pre", "model", "train", "eval"]:
                input_file = workspace_path / f"inputs_{s}.json"
                if input_file.exists():
                    zf.write(input_file, arcname=input_file.name)
        
        return send_from_directory(workspace_path, zip_name, as_attachment=True)
    
    return jsonify({"error": "Unknown stage"}), 400



@app.route("/app/<user_id>")
def main_app_with_user(user_id):
    """사용자 ID가 포함된 URL 지원"""
    return redirect(url_for("main_app", user_id=user_id))


# ===== 에러 핸들러 =====

@app.errorhandler(404)
def not_found(e):
    """404 에러 처리"""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    """500 에러 처리"""
    return jsonify({"error": "Internal server error"}), 500


# ===== 초기화 =====

def initialize_directories():
    """필수 디렉터리 생성"""
    for directory in [BASE_DIR, DATASET_DIR, WORKSPACE_DIR, LOGS_DIR]:
        directory.mkdir(exist_ok=True)


# ===== 메인 실행 =====

if __name__ == "__main__":
    initialize_directories()
    
    app.run(
        host="127.0.0.1",
        port=9000,
        debug=True,
        use_reloader=False
    )