import torch
import cv2
import numpy as np
from ANNModel import ANNModel  # Pastikan ini adalah class model ANN yang sama
from datasetLoader import ImageTensorDataset  # Fungsi transformasi gambar yang digunakan sebelumnya

# *Konfigurasi Perangkat (CPU/GPU)*
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *Memuat Dataset untuk Mendapatkan Nama Folder (Label Asli)*
train_dataset = ImageTensorDataset("DATASET/TRAINING")  # Pastikan path sesuai dengan dataset training
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}  # Konversi index ke nama folder

# *Memuat Model ANN*
input_size = 224 * 224 * 3  # Sesuai dengan model yang dilatih
num_classes = len(idx_to_class)  # Pastikan sesuai dengan jumlah kelas dalam dataset
model = ANNModel(input_size, num_classes).to(device)
model.load_state_dict(torch.load("trained_model.pth", map_location=device))
model.eval()  # Set model ke mode evaluasi

# *Inisialisasi Face Detector (Haar Cascade)*
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# *Inisialisasi Kamera*
cap = cv2.VideoCapture(0)  # 0 untuk kamera utama

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # *Konversi frame ke grayscale untuk deteksi wajah*
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # *Potong wajah dari frame*
        face_crop = frame[y:y + h, x:x + w]

        # *Preprocessing Wajah untuk Model*
        try:
            img = cv2.resize(face_crop, (224, 224))  # Resize ke ukuran yang sesuai
            img = img.astype(np.float32) / 255.0  # Normalisasi 0-1
            img = np.transpose(img, (2, 0, 1))  # Ubah ke format (C, H, W)
            img_tensor = torch.tensor(img).unsqueeze(0).to(device)  # Tambahkan batch dimensi & kirim ke GPU/CPU

            # *Prediksi Kelas Wajah*
            with torch.no_grad():
                output = model(img_tensor.reshape(-1, input_size))
                _, predicted = torch.max(output, 1)
                label_predicted = idx_to_class.get(predicted.item(), "Unknown")  # Ambil nama kelas asli

            # *Gambar Bounding Box & Label pada Wajah*
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{label_predicted}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error processing face: {e}")

    # *Tampilkan Video dengan Bounding Box*
    cv2.imshow("Face Recognition with ANN", frame)

    # *Tekan 'q' untuk keluar*
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# *Tutup Kamera*
cap.release()
cv2.destroyAllWindows()