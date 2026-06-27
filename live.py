import time
import json
import cv2
import threading
import signal
import sys
from datetime import datetime
import ssl

from flask import Flask, Response, jsonify
from ultralytics import YOLO
import paho.mqtt.client as mqtt


# =========================
# Load Config
# =========================

try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json tidak ditemukan.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: config.json tidak valid. Detail: {e}")
    sys.exit(1)


MQTT_BROKER = config["mqtt_broker"]
MQTT_PORT = int(config.get("mqtt_port", 8883))
MQTT_TOPIC = config["mqtt_topic"]
MQTT_USERNAME = config["mqtt_username"]
MQTT_PASSWORD = config["mqtt_password"]
MQTT_TLS = bool(config.get("mqtt_tls", True))


DEVICE_ID = config.get("device_id", "raspi-plant-01")
DEVICE_TOKEN = config["device_token"]

INTERVAL_SCAN = int(config.get("interval_scan_seconds", 1800))

CAMERA_INDEX = int(config.get("camera_index", 0))
STREAM_HOST = config.get("stream_host", "0.0.0.0")
STREAM_PORT = int(config.get("stream_port", 5001))

STREAM_WIDTH = int(config.get("stream_width", 640))
STREAM_HEIGHT = int(config.get("stream_height", 480))
STREAM_FPS = int(config.get("stream_fps", 10))
JPEG_QUALITY = int(config.get("jpeg_quality", 60))

YOLO_MODEL_PATH = config.get("yolo_model", "best.pt")
YOLO_IMGSZ = int(config.get("yolo_imgsz", 416))
YOLO_CONF = float(config.get("yolo_conf", 0.4))

TORCH_NUM_THREADS = int(config.get("torch_num_threads", 2))


# =========================
# Torch Optimization
# =========================

try:
    import torch
    torch.set_num_threads(TORCH_NUM_THREADS)
    print(f"Torch CPU threads dibatasi: {TORCH_NUM_THREADS}")
except Exception as e:
    print(f"Torch setting dilewati: {e}")


# =========================
# Global State
# =========================

app = Flask(__name__)

latest_frame = None
latest_jpeg = None
latest_detection_jpeg = None

frame_id = 0
frame_lock = threading.Lock()
frame_condition = threading.Condition(frame_lock)

running = True
mqtt_connected = False
camera_online = False
model_ready = False


# =========================
# Load YOLO Model
# =========================

try:
    print("Loading YOLO model...")
    model_yolo = YOLO(YOLO_MODEL_PATH)
    model_ready = True
    print("YOLO model loaded.")
except Exception as e:
    model_ready = False
    print(f"Error loading YOLO model: {e}")
    sys.exit(1)


# =========================
# MQTT Setup
# =========================

def on_connect(client, userdata, flags, rc, *extra):
    global mqtt_connected

    if rc == 0:
        mqtt_connected = True
        print("MQTT connected.")
    else:
        mqtt_connected = False
        print(f"MQTT gagal connect. RC={rc}")


def on_disconnect(client, userdata, rc, *extra):
    global mqtt_connected

    mqtt_connected = False
    print(f"MQTT disconnected. RC={rc}")


mqtt_client = mqtt.Client(
    client_id=DEVICE_ID,
    clean_session=True,
    protocol=mqtt.MQTTv311
)

# ThingsBoard menggunakan device token sebagai username
mqtt_client.username_pw_set(
    username=MQTT_USERNAME,
    password=MQTT_PASSWORD
)

if MQTT_TLS:
    mqtt_client.tls_set(
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLS_CLIENT
    )
    mqtt_client.tls_insecure_set(False)

mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect

mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
mqtt_client.max_inflight_messages_set(20)


def start_mqtt():
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"MQTT connecting to {MQTT_BROKER}:{MQTT_PORT}")
    except Exception as e:
        print(f"MQTT initial connect error: {e}")


def publish_mqtt(payload):
    try:
        data_json = json.dumps(payload)

        result = mqtt_client.publish(
            MQTT_TOPIC,
            data_json,
            qos=1,
            retain=False
        )

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            result.wait_for_publish(timeout=5)
            print("Data MQTT berhasil dikirim.")
        else:
            print(f"Gagal publish MQTT. RC={result.rc}")

    except Exception as e:
        print(f"Error publish MQTT: {e}")


# =========================
# Camera Loop
# =========================

def camera_loop():
    global latest_frame, latest_jpeg, frame_id, running, camera_online

    cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    # Untuk USB webcam, MJPG biasanya lebih ringan di Raspberry Pi
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
    cam.set(cv2.CAP_PROP_FPS, STREAM_FPS)

    # Supaya buffer kamera tidak numpuk frame lama
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cam.isOpened():
        camera_online = False
        running = False
        print("Error: kamera tidak bisa dibuka.")
        return

    camera_online = True

    print("Kamera aktif.")
    print(f"Live MJPEG: http://<IP_RASPBERRY_PI>:{STREAM_PORT}/video_feed")
    print(f"Snapshot:    http://<IP_RASPBERRY_PI>:{STREAM_PORT}/snapshot.jpg")
    print(f"Health:      http://<IP_RASPBERRY_PI>:{STREAM_PORT}/health")

    delay = 1.0 / max(STREAM_FPS, 1)

    while running:
        start = time.monotonic()

        ret, frame = cam.read()

        if not ret:
            camera_online = False
            print("Warning: gagal membaca frame kamera.")
            time.sleep(0.2)
            continue

        camera_online = True

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )

        if success:
            with frame_condition:
                latest_frame = frame.copy()
                latest_jpeg = buffer.tobytes()
                frame_id += 1
                frame_condition.notify_all()

        elapsed = time.monotonic() - start
        sleep_time = max(0, delay - elapsed)
        time.sleep(sleep_time)

    cam.release()
    camera_online = False
    print("Kamera ditutup.")


# =========================
# MJPEG Stream
# =========================

def generate_mjpeg():
    last_sent_id = -1

    while running:
        with frame_condition:
            frame_condition.wait_for(
                lambda: frame_id != last_sent_id or not running,
                timeout=2
            )

            if not running:
                break

            frame_bytes = latest_jpeg
            current_id = frame_id

        if frame_bytes is None:
            time.sleep(0.05)
            continue

        last_sent_id = current_id

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/snapshot.jpg")
def snapshot():
    with frame_lock:
        frame_bytes = latest_jpeg

    if frame_bytes is None:
        return "Frame belum tersedia", 503

    return Response(frame_bytes, mimetype="image/jpeg")


@app.route("/latest_detection.jpg")
def latest_detection():
    with frame_lock:
        frame_bytes = latest_detection_jpeg

    if frame_bytes is None:
        return "Gambar deteksi belum tersedia", 503

    return Response(frame_bytes, mimetype="image/jpeg")


@app.route("/health")
def health():
    with frame_lock:
        has_frame = latest_jpeg is not None
        has_detection_image = latest_detection_jpeg is not None

    return jsonify({
        "device_id": DEVICE_ID,
        "camera_online": camera_online,
        "camera_frame_available": has_frame,
        "stream_status": "running" if running and camera_online else "stopped",
        "model_status": "ready" if model_ready else "not_ready",
        "mqtt_connected": mqtt_connected,
        "stream_width": STREAM_WIDTH,
        "stream_height": STREAM_HEIGHT,
        "stream_fps": STREAM_FPS,
        "latest_detection_image_available": has_detection_image,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


# =========================
# Detection
# =========================

def ambil_frame_terbaru():
    with frame_lock:
        if latest_frame is None:
            return None
        return latest_frame.copy()


def deteksi_daun(frame):
    results = model_yolo.predict(
        source=frame,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        verbose=False
    )

    kondisi_daun = "Tidak Terdeteksi"
    keyakinan = 0.0
    annotated_frame = frame.copy()

    for result in results:
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            confidences = boxes.conf.cpu().numpy()
            best_index = int(confidences.argmax())

            class_index = int(boxes.cls[best_index])
            kondisi_daun = model_yolo.names[class_index]
            keyakinan = float(boxes.conf[best_index]) * 100

            annotated_frame = result.plot()
            break

    return kondisi_daun, keyakinan, annotated_frame


def buat_payload_deteksi(kondisi_daun, keyakinan, waktu_proses):
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if kondisi_daun != "Tidak Terdeteksi":
        status_deteksi = "terdeteksi"
    else:
        status_deteksi = "tidak_terdeteksi"

    payload = {
        "device_id": DEVICE_ID,

        "kondisi_daun": kondisi_daun,
        "tingkat_keyakinan": round(keyakinan, 2),
        "status_deteksi": status_deteksi,

        "camera_status": "online" if camera_online else "offline",
        "stream_status": "running" if running and camera_online else "stopped",
        "model_status": "ready" if model_ready else "not_ready",

        "waktu_proses_deteksi_detik": round(waktu_proses, 2),
        "detected_at": detected_at
    }

    return payload


def detection_loop():
    global latest_detection_jpeg

    print(f"Detection aktif. Interval: {INTERVAL_SCAN} detik.")

    # Tunggu kamera siap
    time.sleep(3)

    while running:
        start_time = time.time()

        frame = ambil_frame_terbaru()

        if frame is None:
            print("Frame belum tersedia. Menunggu kamera...")
            time.sleep(2)
            continue

        print("Menjalankan deteksi YOLO...")

        try:
            kondisi_daun, keyakinan, annotated_frame = deteksi_daun(frame)
        except Exception as e:
            kondisi_daun = "Error Deteksi"
            keyakinan = 0.0
            annotated_frame = frame.copy()
            print(f"Error saat deteksi YOLO: {e}")

        elapsed = time.time() - start_time

        payload = buat_payload_deteksi(
            kondisi_daun=kondisi_daun,
            keyakinan=keyakinan,
            waktu_proses=elapsed
        )

        success, buffer = cv2.imencode(
            ".jpg",
            annotated_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )

        if success:
            with frame_lock:
                latest_detection_jpeg = buffer.tobytes()
        else:
            print("Warning: gagal encode gambar hasil deteksi.")

        publish_mqtt(payload)

        print("Payload MQTT:")
        print(payload)

        time.sleep(INTERVAL_SCAN)


# =========================
# Graceful Shutdown
# =========================

def stop_program(signum=None, frame=None):
    global running

    if not running:
        return

    print("Menghentikan sistem...")
    running = False

    try:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    except Exception:
        pass

    with frame_condition:
        frame_condition.notify_all()

    time.sleep(1)
    sys.exit(0)


signal.signal(signal.SIGINT, stop_program)
signal.signal(signal.SIGTERM, stop_program)


# =========================
# Main
# =========================

if __name__ == "__main__":
    start_mqtt()

    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    detection_thread = threading.Thread(target=detection_loop, daemon=True)

    camera_thread.start()
    detection_thread.start()

    app.run(
        host=STREAM_HOST,
        port=STREAM_PORT,
        debug=False,
        threaded=True,
        use_reloader=False
    )