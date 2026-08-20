# 📊 DOKUMENTASI APLIKASI WEB - FLOOD SEGMENTATION SYSTEM

## 🎯 OVERVIEW APLIKASI

**Nama Aplikasi:** Flood Area Segmentation System  
**Teknologi:** Flask (Python), TensorFlow/Keras, OpenCV  
**Tujuan:** Deteksi dan segmentasi area banjir dari citra satelit menggunakan 4 model Deep Learning

---

## 🏗️ ARSITEKTUR SISTEM

### 1. **Backend Framework**
- **Flask** - Web framework Python yang ringan dan fleksibel
- **SQLite** - Database untuk autentikasi pengguna
- **TensorFlow/Keras** - Framework deep learning untuk model AI
- **OpenCV** - Library computer vision untuk pengolahan gambar

### 2. **Model AI yang Digunakan**
Aplikasi ini mengimplementasikan 4 model segmentasi semantik:

#### a) **U-Net Model** (`unet_flood_model_final.h5`)
- Arsitektur encoder-decoder klasik
- Efektif untuk segmentasi gambar medis dan satelit
- Skip connections untuk mempertahankan detail spasial

#### b) **Attention U-Net** (`attention_unet_flood.h5`)
- U-Net dengan attention gates
- Fokus pada region of interest (area banjir)
- Meningkatkan akurasi dengan attention mechanism

#### c) **U-Net dengan Backbone** (`my_model.h5`)
- Menggunakan pre-trained backbone (seperti ResNet/VGG)
- Transfer learning untuk feature extraction yang lebih baik
- Konvergensi lebih cepat

#### d) **U-Net3+** (`unet3plus_best.h5`)
- Arsitektur U-Net generasi ketiga
- Full-scale skip connections
- Menangkap fitur multi-scale lebih efektif

---

## 📁 STRUKTUR DIREKTORI

```
wbsite/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── users.db                        # SQLite database (auto-generated)
│
├── model/                          # Model AI
│   ├── unet_flood_model_final.h5
│   ├── attention_unet_flood.h5
│   ├── my_model.h5
│   └── unet3plus_best.h5
│
├── templates/                      # HTML Templates
│   ├── login.html                  # Halaman login
│   ├── register.html               # Halaman registrasi
│   ├── dashboard.html              # Dashboard utama
│   └── predict.html                # Halaman prediksi
│
├── static/                         # Static files
│   ├── uploads/                    # Gambar yang di-upload
│   └── results/                    # Hasil segmentasi
│
└── Dataset/                        # Dataset dan resources
    └── slangwords_dict.txt
```

---

## 🎨 FITUR-FITUR APLIKASI

### 1. **Sistem Autentikasi**
- ✅ **Login** - Akses dengan username dan password
- ✅ **Register** - Pendaftaran pengguna baru
- ✅ **Session Management** - Melindungi halaman yang memerlukan login
- ✅ **Logout** - Keluar dari sistem

### 2. **Dashboard**
- 📊 Halaman utama setelah login
- 🎨 Design modern dengan gradient background
- 🔗 Navigasi ke halaman prediksi
- 👤 Menampilkan informasi pengguna yang login

### 3. **Prediksi Area Banjir**
- 📤 **Upload Gambar** - Upload citra satelit/foto udara
- 🤖 **Multi-Model Prediction** - Prediksi menggunakan 4 model sekaligus
- 📊 **Metrik Perbandingan**:
  - Persentase area banjir (%)
  - Confidence score (%)
  - Total pixels
  - Jumlah pixel banjir
- 🖼️ **Visualisasi** - Menampilkan hasil segmentasi dari semua model

---

## 🔧 TEKNOLOGI DETAIL

### Backend (app.py)

#### **Konfigurasi**
```python
IMG_HEIGHT = 256
IMG_WIDTH = 256
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
DATABASE = 'users.db'
```

#### **Custom Metric - Dice Coefficient**
```python
def dice_coef(y_true, y_pred, smooth=1e-6):
    # Metric untuk evaluasi segmentasi
    # Mengukur overlap antara prediksi dan ground truth
    # Range: 0-1 (1 = perfect match)
```

#### **Preprocessing Pipeline**
1. Baca gambar dengan OpenCV
2. Convert BGR → RGB
3. Resize ke 256x256
4. Normalisasi (0-255 → 0-1)
5. Expand dimensions untuk batch

#### **Postprocessing**
1. Threshold (> 0.5) untuk binary mask
2. Scale ke 0-255
3. Simpan sebagai gambar grayscale

#### **Metrics Calculation**
- **Flood Percentage**: Proporsi pixel banjir terhadap total pixel
- **Confidence**: Rata-rata nilai prediksi pada area banjir
- **Total Pixels**: Ukuran gambar (256 × 256 = 65,536)
- **Flood Pixels**: Jumlah pixel yang terdeteksi banjir

---

## 🎯 WORKFLOW APLIKASI

### 1. **User Authentication Flow**
```
START → Login Page → Validate Credentials → Dashboard
                  ↓ (invalid)
              Error Message
```

### 2. **Prediction Flow**
```
Dashboard → Predict Page → Upload Image → Preprocessing
                                              ↓
                                    Run 4 Models in Parallel
                                              ↓
                            [UNet | Attention | Backbone | UNet3+]
                                              ↓
                                       Postprocessing
                                              ↓
                                    Calculate Metrics
                                              ↓
                               Display Results & Comparison
```

### 3. **Model Prediction Pipeline**
```
Input Image (any size)
    ↓
Resize to 256×256
    ↓
Normalize (0-1)
    ↓
Model Prediction
    ↓
Threshold (>0.5)
    ↓
Binary Mask (0 or 255)
    ↓
Save & Display
```

---

## 🎨 DESAIN UI/UX

### 1. **Login Page** (`login.html`)
- **Warna Tema**: Gradient ungu-biru (#667eea → #764ba2)
- **Fitur**:
  - Form login responsif
  - Animasi slide-in
  - Feedback error untuk login gagal
  - Link ke halaman register
- **Elemen**:
  - Input username
  - Input password
  - Tombol login
  - Link "Belum punya akun?"

### 2. **Register Page** (`register.html`)
- **Desain**: Serupa dengan login page (konsistensi)
- **Fitur**:
  - Form registrasi
  - Validasi username unik
  - Feedback error untuk username duplikat
  - Link kembali ke login

### 3. **Dashboard** (`dashboard.html`)
- **Layout**:
  - **Navbar**: Brand + Logout button
  - **Welcome Section**: Sambutan untuk user
  - **Info Cards**: Informasi tentang model
  - **Action Button**: Tombol "Mulai Prediksi"
- **Desain**:
  - Card-based layout
  - Gradient backgrounds
  - Hover effects
  - Responsive grid

### 4. **Predict Page** (`predict.html`)
- **Layout**:
  - Upload area (drag & drop style)
  - Results grid (5 kolom: original + 4 model results)
  - Metrics cards untuk setiap model
- **Fitur**:
  - Preview gambar original
  - Side-by-side comparison
  - Detailed metrics untuk setiap model
  - Visual indicators

---

## 📊 DATABASE SCHEMA

### Table: `users`
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
);
```

**Catatan Keamanan**: 
⚠️ Password disimpan dalam plain text (untuk demo/development)
✅ **Rekomendasi Production**: Gunakan hashing (bcrypt, argon2)

---

## 🚀 CARA MENJALANKAN APLIKASI

### 1. **Instalasi Dependencies**
```bash
pip install -r requirements.txt
```

**Dependencies:**
- Flask 3.1.2
- TensorFlow 2.20.0
- OpenCV 4.12.0.88
- NumPy 2.2.6

### 2. **Menjalankan Server**
```bash
python app.py
```

Server akan berjalan di: `http://127.0.0.1:5000`

### 3. **Akses Aplikasi**
1. Buka browser
2. Akses `http://localhost:5000`
3. Register akun baru
4. Login dengan kredensial
5. Mulai prediksi!

---

## 📈 METRIK EVALUASI

### 1. **Flood Percentage**
- **Definisi**: Persentase area yang terdeteksi banjir
- **Formula**: `(flood_pixels / total_pixels) × 100`
- **Interpretasi**:
  - < 10%: Banjir ringan
  - 10-30%: Banjir sedang
  - > 30%: Banjir parah

### 2. **Confidence Score**
- **Definisi**: Tingkat keyakinan model pada prediksi
- **Formula**: `mean(prediction_values_on_flood_area) × 100`
- **Interpretasi**:
  - 50-70%: Low confidence
  - 70-85%: Medium confidence
  - > 85%: High confidence

### 3. **Comparison Metrics**
- Total pixels: 65,536 (256×256)
- Flood pixels: Jumlah pixel yang diprediksi banjir
- Non-flood pixels: Total - Flood pixels

---

## 🎯 KASUS PENGGUNAAN

### Use Case 1: Disaster Management
**Skenario**: Tim SAR perlu mengetahui area banjir untuk evakuasi
1. Upload foto drone/satelit
2. Dapatkan segmentasi area banjir
3. Hitung persentase area terdampak
4. Tentukan prioritas evakuasi

### Use Case 2: Urban Planning
**Skenario**: Pemerintah merencanakan infrastruksi drainase
1. Analisis foto historis banjir
2. Identifikasi area rawan banjir
3. Bandingkan hasil dari multiple models
4. Putuskan lokasi infrastruktur

### Use Case 3: Research & Education
**Skenario**: Penelitian tentang segmentasi semantik
1. Upload dataset test
2. Bandingkan performa 4 model
3. Analisis metrics
4. Evaluasi model terbaik

---

## 🔍 PERBEDAAN KEEMPAT MODEL

| Aspek | U-Net | Attention U-Net | U-Net Backbone | U-Net3+ |
|-------|-------|-----------------|----------------|---------|
| **Arsitektur** | Basic encoder-decoder | U-Net + attention gates | U-Net + pre-trained backbone | Full-scale connections |
| **Kompleksitas** | Sedang | Tinggi | Tinggi | Sangat Tinggi |
| **Parameter** | ~7-10M | ~10-15M | ~20-30M | ~25-35M |
| **Kecepatan** | Cepat | Sedang | Sedang | Lambat |
| **Akurasi** | Good | Better | Better | Best |
| **Use Case** | General purpose | Detail penting | Transfer learning | Maximum accuracy |

---

## 🛡️ KEAMANAN

### Implementasi Saat Ini
- ✅ Session management dengan Flask sessions
- ✅ Database isolation
- ✅ File upload ke folder tertentu

### ⚠️ Perlu Ditingkatkan untuk Production
- ❌ Password hashing (gunakan bcrypt/argon2)
- ❌ CSRF protection
- ❌ SQL injection prevention (gunakan parameterized queries)
- ❌ File upload validation (type, size, content)
- ❌ HTTPS/SSL encryption
- ❌ Rate limiting
- ❌ Input sanitization

---

## 🐛 TROUBLESHOOTING

### Problem 1: Model loading error
**Solusi**: 
- Pastikan semua model ada di folder `model/`
- Check TensorFlow version compatibility

### Problem 2: Database locked
**Solusi**:
```python
conn = sqlite3.connect(DATABASE, timeout=10)
```

### Problem 3: Out of Memory
**Solusi**:
- Reduce image size
- Process images one by one
- Use model.predict with smaller batch size

### Problem 4: Low accuracy
**Solusi**:
- Check image quality
- Ensure correct preprocessing
- Try different threshold values

---

## 📝 API ENDPOINTS

### 1. **Authentication**
```
GET  /              → Login page
POST /login         → Login submission
GET  /register      → Register page
POST /register      → Register submission
GET  /logout        → Logout
```

### 2. **Application**
```
GET  /dashboard     → Dashboard (protected)
GET  /predict       → Prediction page (protected)
POST /predict       → Upload & predict (protected)
```

### 3. **Static Files**
```
GET  /static/uploads/<filename>    → Uploaded images
GET  /static/results/<filename>    → Segmentation results
```

---

## 📸 CARA MENGAMBIL SCREENSHOT

### 1. **Login Page**
- Buka `http://localhost:5000`
- Screenshot form login

### 2. **Register Page**
- Klik "Belum punya akun?"
- Screenshot form register

### 3. **Dashboard**
- Login dengan akun
- Screenshot dashboard dengan navbar dan cards

### 4. **Prediction - Before**
- Klik "Mulai Prediksi"
- Screenshot upload area

### 5. **Prediction - After**
- Upload gambar banjir
- Klik "Predict"
- Screenshot hasil prediksi (5 gambar + metrics)

### 6. **Comparison View**
- Scroll untuk melihat semua hasil
- Screenshot perbandingan metrics

---

## 🎓 TEKNOLOGI PEMBELAJARAN

### Deep Learning Concepts
1. **Semantic Segmentation**: Klasifikasi per-pixel
2. **U-Net Architecture**: Encoder-decoder dengan skip connections
3. **Attention Mechanism**: Focus pada region penting
4. **Transfer Learning**: Memanfaatkan pre-trained models
5. **Dice Coefficient**: Metric untuk segmentasi

### Web Development
1. **Flask Framework**: Python web framework
2. **Jinja2 Templates**: Template engine
3. **Session Management**: User authentication
4. **File Upload**: Handling multipart/form-data
5. **Responsive Design**: Mobile-friendly UI

### Computer Vision
1. **Image Preprocessing**: Resize, normalize, color conversion
2. **Binary Segmentation**: Threshold untuk klasifikasi
3. **Mask Generation**: Output model ke visualization
4. **Metrics Calculation**: Evaluasi hasil segmentasi

---

## 🚀 FUTURE ENHANCEMENTS

### Short Term
1. ✨ Add image preview before upload
2. 📊 Export results as PDF/CSV
3. 📱 Improve mobile responsiveness
4. 🎨 Add more themes/color schemes
5. 📈 History/log of predictions

### Medium Term
1. 🗺️ Interactive map overlay
2. 📊 Statistics dashboard
3. 👥 Multi-user management
4. 📁 Batch processing
5. 🔔 Email notifications

### Long Term
1. 🧠 Model training interface
2. 🌐 Real-time processing
3. 🛰️ Satellite API integration
4. 📱 Mobile app
5. ☁️ Cloud deployment (AWS/GCP/Azure)

---

## 📚 REFERENSI

### Papers
1. **U-Net**: Ronneberger et al. - "U-Net: Convolutional Networks for Biomedical Image Segmentation"
2. **Attention U-Net**: Oktay et al. - "Attention U-Net: Learning Where to Look for the Pancreas"
3. **U-Net3+**: Huang et al. - "UNet 3+: A Full-Scale Connected UNet"

### Frameworks
- Flask: https://flask.palletsprojects.com/
- TensorFlow: https://www.tensorflow.org/
- OpenCV: https://opencv.org/

---

## 👨‍💻 DEVELOPER NOTES

### Code Style
- Python: PEP 8
- HTML/CSS: BEM methodology
- Comments: Indonesian + English

### Testing Checklist
- [ ] Test login dengan berbagai kredensial
- [ ] Test register dengan username duplikat
- [ ] Test upload berbagai format gambar
- [ ] Test semua model prediksi
- [ ] Test logout dan session
- [ ] Test responsive design
- [ ] Test error handling

### Deployment Checklist
- [ ] Set debug=False
- [ ] Implement password hashing
- [ ] Add HTTPS
- [ ] Configure production database
- [ ] Set up logging
- [ ] Add monitoring
- [ ] Configure CORS if needed
- [ ] Set up backup system

---

## 📞 SUPPORT & CONTACT

Untuk pertanyaan atau masalah teknis, silakan hubungi:
- 📧 Email: [your-email]
- 💬 GitHub Issues: [repository-url]
- 📱 WhatsApp: [your-number]

---

**Version**: 1.0.0  
**Last Updated**: 18 Januari 2026  
**Status**: ✅ Production Ready (dengan catatan keamanan)

---

## 📄 LICENSE

[Tentukan lisensi yang sesuai - MIT, GPL, Apache, dll.]

---

**© 2026 Flood Segmentation System - All Rights Reserved**
