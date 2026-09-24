
import os
import sys
import cv2
import time
import threading
import requests
import numpy as np
from pathlib import Path
from flask import Flask, Response, jsonify, request, render_template, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_path, ".."))
yolo_src = os.path.join(project_root, "YOLOv13-main")
sys.path.insert(0, yolo_src)

from ultralytics import YOLO
from scoring import BilliardScoring

try:
    import torch
    HAS_CUDA = bool(torch.cuda.is_available())
except Exception:
    HAS_CUDA = False

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "billiard_system_secret_key_2026"
CORS(app)

DEFAULT_VIDEO_SOURCE = os.path.join(base_path, "videos", "test1.mp4")
MODEL_PATH = os.path.join(base_path, "models", "best.pt")
UPLOAD_DIR = os.path.join(base_path, "uploads")
RECORD_DIR = os.path.join(base_path, "recordings")
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FRAME_SIZE = (1280, 720)
JPEG_QUALITY = 78
TRACK_CONF = 0.10
TRACK_IOU = 0.45
STREAM_FPS_FALLBACK = 20.0
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:1.5b")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RECORD_DIR, exist_ok=True)

LOGIN_USER = "admin"
LOGIN_PASS = "123456"

video_state = {
    "is_paused": True,
    "need_restart": False,
    "seek_to": -1,
    "current_frame": 0,
    "total_frames": 1
}

source_state = {
    "mode": "none",
    "camera_index": 0,
    "need_reopen": False
}

current_video_state = {
    "path": DEFAULT_VIDEO_SOURCE,
    "name": os.path.basename(DEFAULT_VIDEO_SOURCE)
}

recording_state = {
    "enabled": False,
    "writer": None,
    "path": "",
    "filename": ""
}

state_lock = threading.Lock()
frame_lock = threading.Lock()

latest_jpeg = None
worker_started = False
video_preview_cache = None
model_warmed_up = False

print(f"🚀 Loading Model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
scorer = BilliardScoring()


def is_logged_in():
    return bool(session.get("logged_in", False))


def make_text_frame(text, width=FRAME_SIZE[0], height=FRAME_SIZE[1]):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(img, text, (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 255), 3, cv2.LINE_AA)
    return img


def normalize_frame(frame):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if (w, h) == FRAME_SIZE:
        return frame
    return cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)


def encode_jpeg(frame):
    return cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])


def put_preview_frame(frame, label="LOCAL VIDEO READY"):
    global latest_jpeg
    show = normalize_frame(frame).copy()
    cv2.putText(show, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    ok, buf = encode_jpeg(show)
    if ok:
        with frame_lock:
            latest_jpeg = buf.tobytes()


def open_camera_with_fallback(index: int):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap is not None and cap.isOpened():
                print(f"✅ 摄像头打开成功 [Backend: {backend}] index={index}")
                return cap
            if cap is not None:
                cap.release()
        except Exception:
            pass
    print(f"❌ 摄像头 {index} 打开失败")
    return None


def open_video_with_fallback(path: str):
    backends = [cv2.CAP_FFMPEG, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        try:
            cap = cv2.VideoCapture(path, backend)
            if cap is not None and cap.isOpened():
                print(f"✅ 本地视频打开成功 [Backend: {backend}] path={path}")
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            if cap is not None:
                cap.release()
        except Exception:
            pass
    print(f"❌ 本地视频打开失败: {path}")
    return None


def get_current_video_source():
    with state_lock:
        return current_video_state["path"], current_video_state["name"]


def set_current_video_source(path: str):
    global video_preview_cache
    with state_lock:
        current_video_state["path"] = path
        current_video_state["name"] = os.path.basename(path)
        video_preview_cache = None


def cache_local_video_preview(force=False):
    global video_preview_cache
    if video_preview_cache is not None and not force:
        return

    video_path, _ = get_current_video_source()
    cap = open_video_with_fallback(video_path)
    if cap is None:
        return

    try:
        success, frame = cap.read()
        if success and frame is not None:
            video_preview_cache = normalize_frame(frame)
            print("✅ 已缓存本地视频首帧预览")
    finally:
        cap.release()


def warm_up_model():
    global model_warmed_up
    if model_warmed_up:
        return

    try:
        print("🔥 开始预热 YOLO 模型与跟踪器...")
        dummy = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        model.track(
            dummy,
            persist=True,
            conf=TRACK_CONF,
            iou=TRACK_IOU,
            tracker="bytetrack.yaml",
            imgsz=FRAME_SIZE[0],
            verbose=False
        )
        model_warmed_up = True
        print("✅ YOLO 模型预热完成")
    except Exception as e:
        print(f"⚠️ 模型预热失败，但系统仍可继续运行: {e}")


def background_prepare_local_video():
    cache_local_video_preview(force=True)
    warm_up_model()


def probe_cameras():
    return [
        {"label": "默认摄像头 (设备 0)", "value": 0},
        {"label": "外接摄像头 (设备 1)", "value": 1},
        {"label": "虚拟摄像头 (设备 2)", "value": 2}
    ]


def build_coach_system_prompt(mode: str, score_solid: int, score_stripe: int, foul: bool):
    mode_name = "本地视频回放" if mode == "file" else "实时摄像头"
    foul_text = "是" if foul else "否"
    return f"""
你是“专属桌球教练”，服务于一个中文桌球裁判与分析系统。
请始终使用中文回答，语气专业、清晰、实用。
你的回答原则：
1. 优先给出可执行的桌球建议，而不是空泛解释。
2. 当用户问策略时，优先从进攻选择、母球走位、防守风险、下一杆衔接来分析。
3. 信息不足时要明确说明你是在基于当前已知信息推断。
4. 不要编造你没有看到的具体球型细节。
5. 当用户询问规则时，直接解释黑八规则、犯规判定、常见争议点。

当前系统状态：
- 当前模式：{mode_name}
- 实色球比分：{score_solid}
- 花色球比分：{score_stripe}
- 当前是否犯规：{foul_text}

请以“专属桌球教练”的身份回答。
"""


def sanitize_history(history):
    messages = []
    if not isinstance(history, list):
        return messages
    for item in history[-8:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content[:2000]})
    return messages


def open_capture_by_state(mode, cam_idx):
    if mode == "none":
        return None

    if mode == "camera":
        print(f"📷 尝试打开摄像头: {cam_idx}")
        cap = open_camera_with_fallback(cam_idx)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        return None

    video_path, video_name = get_current_video_source()
    print(f"🎬 尝试打开本地视频: {video_name} -> {video_path}")
    return open_video_with_fallback(video_path)


def stop_recording():
    with state_lock:
        writer = recording_state["writer"]
        recording_state["writer"] = None
        recording_state["enabled"] = False
        path = recording_state["path"]
        filename = recording_state["filename"]
        recording_state["path"] = ""
        recording_state["filename"] = ""
    if writer is not None:
        try:
            writer.release()
        except Exception:
            pass
    return path, filename


def start_recording():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"billiard_record_{timestamp}.mp4"
    save_path = os.path.join(RECORD_DIR, filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, 20.0, FRAME_SIZE)
    if not writer.isOpened():
        raise RuntimeError("录制文件创建失败，请检查 recordings 目录权限")

    with state_lock:
        old_writer = recording_state["writer"]
        recording_state["writer"] = writer
        recording_state["enabled"] = True
        recording_state["path"] = save_path
        recording_state["filename"] = filename

    if old_writer is not None:
        try:
            old_writer.release()
        except Exception:
            pass
    return filename


def run_inference(frame):
    frame = normalize_frame(frame)
    kwargs = dict(
        persist=True,
        conf=TRACK_CONF,
        iou=TRACK_IOU,
        tracker="bytetrack.yaml",
        imgsz=FRAME_SIZE[0],
        verbose=False
    )
    if HAS_CUDA:
        kwargs["device"] = 0
        kwargs["half"] = True
    results = model.track(frame, **kwargs)
    return frame, results


def video_worker():
    global latest_jpeg
    cap = None
    current_mode = None
    current_cam = None

    while True:
        try:
            loop_start = time.time()

            with state_lock:
                mode = source_state["mode"]
                cam_idx = int(source_state["camera_index"])
                need_reopen = source_state["need_reopen"]
                is_paused = video_state["is_paused"]
                source_changed = (mode != current_mode) or (cam_idx != current_cam)

            if cap is None or need_reopen or source_changed:
                if cap is not None:
                    cap.release()
                    cap = None
                    time.sleep(0.08)

                if mode == "none":
                    ok, buf = encode_jpeg(make_text_frame("System Standby. Please select a mode."))
                    if ok:
                        with frame_lock:
                            latest_jpeg = buf.tobytes()
                elif mode == "camera":
                    ok, buf = encode_jpeg(make_text_frame(f"Starting Camera {cam_idx}... Please wait."))
                    if ok:
                        with frame_lock:
                            latest_jpeg = buf.tobytes()
                else:
                    if video_preview_cache is not None:
                        put_preview_frame(video_preview_cache, "LOCAL VIDEO READY")
                    else:
                        ok, buf = encode_jpeg(make_text_frame("Loading Local Video..."))
                        if ok:
                            with frame_lock:
                                latest_jpeg = buf.tobytes()

                cap = open_capture_by_state(mode, cam_idx)

                with state_lock:
                    current_mode = mode
                    current_cam = cam_idx
                    source_state["need_reopen"] = False
                    scorer.reset_game()
                    video_state["is_paused"] = False
                    video_state["need_restart"] = False
                    video_state["current_frame"] = 0
                    video_state["total_frames"] = 1
                    is_paused = False

                if current_mode != "camera":
                    stop_recording()

                if cap is not None and mode == "file":
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    with state_lock:
                        video_state["total_frames"] = total_frames if total_frames > 0 else 1

            if is_paused:
                time.sleep(0.05)
                continue

            if cap is None or (not cap.isOpened()):
                tip = "System Standby..." if current_mode == "none" else "Video source unavailable. Check connection."
                ok, buf = encode_jpeg(make_text_frame(tip))
                if ok:
                    with frame_lock:
                        latest_jpeg = buf.tobytes()
                time.sleep(0.4)
                continue

            with state_lock:
                mode_now = source_state["mode"]
                seek_to = video_state["seek_to"]
                need_restart = video_state["need_restart"]
                recording_enabled = recording_state["enabled"]
                recording_writer = recording_state["writer"]

            if seek_to >= 0 and mode_now == "file":
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
                scorer.reset_game()
                with state_lock:
                    video_state["seek_to"] = -1

            if need_restart:
                if mode_now == "file":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                scorer.reset_game()
                with state_lock:
                    video_state["need_restart"] = False
                    video_state["is_paused"] = False

            success, frame = cap.read()

            if not success:
                if mode_now == "file":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    scorer.reset_game()
                    time.sleep(0.01)
                    continue
                else:
                    ok, buf = encode_jpeg(make_text_frame("Camera frame drop. Reconnecting..."))
                    if ok:
                        with frame_lock:
                            latest_jpeg = buf.tobytes()
                    time.sleep(0.1)
                    continue

            frame, results = run_inference(frame)

            if current_mode == "file":
                with state_lock:
                    video_state["current_frame"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            scorer.update(results[0])
            annotated = results[0].plot()

            for px, py in scorer.pockets:
                cv2.circle(annotated, (int(px), int(py)), scorer.pocket_radius, (0, 255, 0), 2)

            mode_label = "LOCAL VIDEO" if current_mode == "file" else f"CAMERA {current_cam}"
            cv2.putText(annotated, mode_label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

            if current_mode == "camera" and recording_enabled and recording_writer is not None:
                recording_writer.write(annotated)

            ok, buf = encode_jpeg(annotated)
            if ok:
                with frame_lock:
                    latest_jpeg = buf.tobytes()

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = STREAM_FPS_FALLBACK
            target_delay = 1.0 / min(fps, 20.0 if current_mode == "file" else 25.0)
            cost = time.time() - loop_start
            if target_delay - cost > 0:
                time.sleep(target_delay - cost)

        except Exception as e:
            print(f"🔥 视频处理线程异常: {e}")
            time.sleep(0.5)


def ensure_worker_started():
    global worker_started
    with state_lock:
        if not worker_started:
            t = threading.Thread(target=video_worker, daemon=True)
            t.start()
            worker_started = True


def is_allowed_video_file(filename: str):
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def build_safe_video_filename(filename: str):
    safe_name = secure_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    stem = Path(safe_name).stem[:60] or "video"
    return f"{int(time.time())}_{stem}{suffix}"


def generate_stream():
    ensure_worker_started()
    while True:
        with frame_lock:
            frame_data = latest_jpeg
        if frame_data is None:
            time.sleep(0.03)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.02)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    if not is_logged_in():
        return Response("Unauthorized", status=401)
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json(force=True)
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()
    if username == LOGIN_USER and password == LOGIN_PASS:
        session["logged_in"] = True
        return jsonify({"ok": True, "msg": "登录成功"})
    return jsonify({"ok": False, "msg": "用户名或密码错误"}), 401


@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    stop_recording()
    return jsonify({"ok": True})


@app.route('/api/cameras')
def api_cameras():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    return jsonify({"cameras": probe_cameras()})


@app.route('/api/source')
def api_source():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    with state_lock:
        return jsonify({
            "mode": source_state["mode"],
            "camera_index": source_state["camera_index"],
            "video_name": current_video_state["name"]
        })


@app.route('/api/video/current')
def api_video_current():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    video_path, video_name = get_current_video_source()
    return jsonify({"ok": True, "video_name": video_name, "video_path": video_path})


@app.route('/api/video/upload', methods=['POST'])
def api_video_upload():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401

    if "video" not in request.files:
        return jsonify({"msg": "未检测到视频文件"}), 400

    file = request.files["video"]
    if file is None or not file.filename:
        return jsonify({"msg": "请选择要上传的视频文件"}), 400

    if not is_allowed_video_file(file.filename):
        return jsonify({"msg": "仅支持 mp4 / avi / mov / mkv / webm 格式"}), 400

    save_name = build_safe_video_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, save_name)
    file.save(save_path)

    set_current_video_source(save_path)
    cache_local_video_preview(force=True)

    with state_lock:
        video_state["need_restart"] = True
        video_state["seek_to"] = -1
        video_state["current_frame"] = 0
        video_state["total_frames"] = 1
        if source_state["mode"] == "file":
            source_state["need_reopen"] = True
            video_state["is_paused"] = False

    return jsonify({"ok": True, "msg": "本地视频已更新", "video_name": save_name})


@app.route('/api/source/select', methods=['POST'])
def api_source_select():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    data = request.get_json(force=True)
    mode = str(data.get("mode", "file")).strip()
    cam_idx = int(data.get("camera_index", 0))

    with state_lock:
        source_state["mode"] = mode
        source_state["camera_index"] = cam_idx
        source_state["need_reopen"] = True
        video_state["is_paused"] = False
        video_state["need_restart"] = False

    ensure_worker_started()
    return jsonify({"ok": True, "msg": "视频源已切换"})


@app.route('/api/source/stop', methods=['POST'])
def api_source_stop():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    with state_lock:
        source_state["mode"] = "none"
        source_state["need_reopen"] = True
        video_state["is_paused"] = True
    stop_recording()
    print("🛑 收到前端返回指令，已释放所有摄像头/视频硬件资源")
    return jsonify({"ok": True, "msg": "Source stopped"})


@app.route('/api/status')
def api_status():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    status = scorer.get_status()
    with state_lock:
        status["current_frame"] = video_state["current_frame"]
        status["total_frames"] = video_state["total_frames"]
        status["is_paused"] = video_state["is_paused"]
        status["video_name"] = current_video_state["name"]
        status["recording"] = recording_state["enabled"]
        status["recording_filename"] = recording_state["filename"]
        status["mode"] = source_state["mode"]
    return jsonify(status)


@app.route('/api/control/pause', methods=['POST'])
def api_pause():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    with state_lock:
        video_state["is_paused"] = True
    return jsonify({"msg": "Paused"})


@app.route('/api/control/resume', methods=['POST'])
def api_resume():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    with state_lock:
        video_state["is_paused"] = False
    return jsonify({"msg": "Resumed"})


@app.route('/api/control/restart', methods=['POST'])
def api_restart():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    with state_lock:
        video_state["need_restart"] = True
    return jsonify({"msg": "Restarted"})


@app.route('/api/control/seek', methods=['POST'])
def api_seek():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    data = request.get_json(force=True)
    prog = data.get("progress", 0)
    with state_lock:
        total_frames = video_state["total_frames"]
        video_state["seek_to"] = int((prog / 100.0) * total_frames)
    return jsonify({"msg": "Seeking"})


@app.route('/api/recording/start', methods=['POST'])
def api_recording_start():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    with state_lock:
        current_mode = source_state["mode"]
        already = recording_state["enabled"]
    if current_mode != "camera":
        return jsonify({"msg": "仅实时摄像头模式支持录制"}), 400
    if already:
        return jsonify({"ok": True, "msg": "录制已在进行中", "filename": recording_state["filename"]})
    try:
        filename = start_recording()
        return jsonify({"ok": True, "msg": "已开始录制", "filename": filename})
    except Exception as e:
        return jsonify({"msg": str(e)}), 500


@app.route('/api/recording/stop', methods=['POST'])
def api_recording_stop():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    _, filename = stop_recording()
    return jsonify({"ok": True, "msg": "录制已停止", "filename": filename})




@app.route('/api/coach/chat', methods=['POST'])
def api_coach_chat():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()
    mode = str(data.get("mode", "file")).strip()
    score_solid = int(data.get("scoreSolid", 0))
    score_stripe = int(data.get("scoreStripe", 0))
    foul = bool(data.get("foul", False))
    history = sanitize_history(data.get("history", []))

    if not user_message:
        return jsonify({"msg": "请输入问题"}), 400

    messages = [{"role": "system", "content": build_coach_system_prompt(mode, score_solid, score_stripe, foul)}]
    messages.extend(history)
    if not history or history[-1].get("content") != user_message:
        messages.append({"role": "user", "content": user_message})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.4
        }
    }

    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        result = resp.json()
        answer = (result.get("message", {}) or {}).get("content") or result.get("response") or "我暂时没有生成回答。"
        return jsonify({
            "ok": True,
            "answer": answer.strip()
        })
    except requests.exceptions.RequestException as e:
        return jsonify({
            "msg": f"专属桌球教练不可用，请检查本地 Ollama 服务和模型配置。详细信息：{str(e)}"
        }), 500

@app.route('/api/adjust', methods=['POST'])
def api_adjust():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    data = request.get_json(force=True)
    team = data.get("team")
    delta = int(data.get("delta", 0))
    if team == "solid":
        scorer.score_solid += delta
    elif team == "stripe":
        scorer.score_stripe += delta
    return jsonify(scorer.get_status())


@app.route('/api/clear_foul')
def api_clear_foul():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    scorer.reset_foul()
    return jsonify({"msg": "Foul Cleared"})


@app.route('/api/reset')
def api_reset():
    if not is_logged_in():
        return jsonify({"msg": "Unauthorized"}), 401
    scorer.reset_game()
    return jsonify({"msg": "Game Reset"})


threading.Thread(target=background_prepare_local_video, daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
