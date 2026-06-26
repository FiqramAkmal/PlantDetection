import time
import json
import cv2
import random
from ultralytics import YOLO

MODEL_PATH = "best.pt"
FOTO_SAMPEL = "foto_daun_tes.jpg"  

print("=== MEMULAI SIMULASI R-SMART IRRIGATION ===")
print("Memuat Model AI...")
try:
    model_yolo = YOLO(MODEL_PATH)
    print("Model AI Berhasil Dimuat.")
except Exception as e:
    print(f"Gagal memuat model: {e}. Pastikan file 'best.pt' ada di folder ini.")
    exit()

def baca_sensor_fisik_simulasi():
    suhu = round(random.uniform(25.0, 33.0), 1)
    kelembapan_udara = round(random.uniform(55.0, 75.0), 1)
    kelembapan_tanah = round(random.uniform(35.0, 60.0), 1) # Di bawah 45% pompa harusnya ON
    level_tangki = round(random.uniform(70.0, 95.0), 1)
    return {
        "suhu": suhu,
        "hum_udara": kelembapan_udara,
        "soil_moisture": kelembapan_tanah,
        "tangki": level_tangki
    }

def hitung_estimasi_kritis(suhu, hum_udara):
    menit = (100 - hum_udara) * 5 / (suhu / 25)
    return round(menit)

try:
    siklus = 1
    while True:
        print(f"\n--- MENJALANKAN SIKLUS KE-{siklus} ---")
        
        data_sensor = baca_sensor_fisik_simulasi()
        print(f"[Sensor] Suhu: {data_sensor['suhu']}°C, Kelembapan Tanah: {data_sensor['soil_moisture']}%")
        
        print(f"[Kamera] Mengambil snapshot daun dari file: {FOTO_SAMPEL}...")
        frame = cv2.imread(FOTO_SAMPEL)
        
        if frame is None:
            print(f"Error: File '{FOTO_SAMPEL}' tidak ditemukan! Taruh foto daun di folder ini.")
            time.sleep(5)
            continue
            
        print("[AI] Menganalisis kondisi kesehatan daun...")
        hasil_yolo = model_yolo(FOTO_SAMPEL, verbose=False) 
        
        kondisi_daun = "Tidak Terdeteksi"
        keyakinan = 0.0
        
        for r in hasil_yolo:
            if len(r.boxes) > 0:
                index_kelas = int(r.boxes.cls[0])
                kondisi_daun = model_yolo.names[index_kelas]
                keyakinan = float(r.boxes.conf[0]) * 100
        
        print(f"[AI Hasil] Daun terdeteksi: {kondisi_daun} ({keyakinan:.2f}%)")
        
        estimasi_menit = hitung_estimasi_kritis(data_sensor["suhu"], data_sensor["hum_udara"])
        
        status_pompa = "OFF"
        if data_sensor["soil_moisture"] < 45.0:
            status_pompa = "ON (Pompa Menyala Otomatis)"
        
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
        
        print("\n[MQTT Payload JSON yang siap dikirim]:")
        print(json.dumps(payload, indent=2))
        
        print("\nMenunggu 5 detik untuk siklus berikutnya... (Tekan Ctrl+C untuk berhenti)")
        time.sleep(5)
        siklus += 1

except KeyboardInterrupt:
    print("\nSimulasi dihentikan oleh pengguna.")