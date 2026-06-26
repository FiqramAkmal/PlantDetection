import time
import json
import cv2
from ultralytics import YOLO
import paho.mqtt.client as mqtt

# --- configuration ---
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    MQTT_BROKER = config["mqtt_broker"]
    MQTT_TOPIC = config["mqtt_topic"]
    DEVICE_TOKEN = config["device_token"]
    THRESHOLD_MOISTURE = config["threshold_moisture"]
    INTERVAL_SCAN = config["interval_scan_seconds"]
except FileNotFoundError:
    print("Error: File config.json tidak ditemukan! Pastikan file berada di folder yang sama.")
    exit()

# load model ai
model_yolo = YOLO('best.pt')

def baca_sensor_fisik():
    # dummy simulasi:
    return {"suhu": 29.0, "hum_udara": 60.0, "soil_moisture": 42.0, "tangki": 80.0}

def hitung_estimasi_kritis(suhu, hum_udara):
    menit = (100 - hum_udara) * 5 / (suhu / 25)
    return round(menit)

# setup MQTT
client = mqtt.Client()
client.username_pw_set(DEVICE_TOKEN)
client.connect(MQTT_BROKER, 1883, 60)
client.loop_start()

try:
    while True:
        data_sensor = baca_sensor_fisik()
        
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        if ret:
            cv2.imwrite('daun_aktual.jpg', frame)
        cam.release()
        
        hasil_yolo = model_yolo('daun_aktual.jpg')
        kondisi_daun = "Tidak Terdeteksi"
        keyakinan = 0.0
        
        for r in hasil_yolo:
            if len(r.boxes) > 0:
                index_kelas = int(r.boxes.cls[0])
                kondisi_daun = model_yolo.names[index_kelas]
                keyakinan = float(r.boxes.conf[0]) * 100
        
        estimasi_menit = hitung_estimasi_kritis(data_sensor["suhu"], data_sensor["hum_udara"])
        
        status_pompa = "OFF"
        if data_sensor["soil_moisture"] < THRESHOLD_MOISTURE and data_sensor["tangki"] > 10.0:
            status_pompa = "ON"
            
        payload = {
            "suhu_udara": data_sensor["suhu"],
            "kelembapan_udara": data_sensor["hum_udara"],
            "kelembapan_tanah": data_sensor["soil_moisture"],
            "level_air_tangki": data_sensor["tangki"],
            "status_pompa": status_pompa,
            "kondisi_daun": kondisi_daun,
            "tingkat_keyakinan": round(keyakinan, 2),
            "estimasi_waktu_kritis_menit": estimasi_menit
        }
        
        client.publish(MQTT_TOPIC, json.dumps(payload))
        print(f"Data Berhasil Dikirim: {payload}")
        
        time.sleep(INTERVAL_SCAN)

except KeyboardInterrupt:
    print("Sistem dihentikan.")
    client.loop_stop()
    client.disconnect()