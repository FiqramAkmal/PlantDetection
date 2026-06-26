from ultralytics import YOLO

# 1. Load model hasil training
model = YOLO('best.pt')

print("\nDAFTAR KELAS YANG DIKENALI")
print(model.names)

# 3. Lakukan deteksi pada foto sampel
foto_tes = 'foto_sampel.jpg'
print(f"\nMenganalisis foto: {foto_tes} ...")
results = model(foto_tes)

# 4. Ekstraksi hasil
for r in results:
    if len(r.boxes) > 0:
        akurasi = float(r.boxes.conf[0]) * 100
        index_kelas = int(r.boxes.cls[0])
        nama_penyakit = model.names[index_kelas]

        print(f"\nHASIL PREDIKSI")
        print(f"Status Daun : {nama_penyakit}")
        print(f"Akurasi     : {akurasi:.2f}%")
    else:
        print("\nTidak ada daun atau penyakit yang terdeteksi di foto ini.")