# Flood Segmentation System

Sistem segmentasi area banjir menggunakan Deep Learning berbasis U-Net dengan 4 model:
- U-Net Standar
- Attention U-Net
- U-Net Backbone
- U-Net3+

## Fitur
- 🔐 Sistem login dan registrasi user
- 🖼️ Upload gambar satelit
- 🤖 Prediksi area banjir dengan 4 model berbeda
- 📊 Visualisasi hasil segmentasi
- 📈 Metrik performa (flood percentage, confidence, pixel count)

## Teknologi
- **Backend**: Flask (Python)
- **Deep Learning**: TensorFlow/Keras
- **Image Processing**: OpenCV
- **Database**: SQLite

## Instalasi

### 1. Clone Repository
```bash
git clone <repository-url>
cd wbsite
```

### 2. Buat Virtual Environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# atau
source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install Flask tensorflow opencv-python numpy
```

### 4. Jalankan Aplikasi
```bash
python app.py
```

### 5. Akses Aplikasi
Buka browser dan akses: `http://localhost:5000`

## Struktur Project
```
wbsite/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── model/                 # Model files (.h5)
│   ├── unet_flood_model_final.h5
│   ├── attention_unet_flood.h5
│   ├── my_model.h5
│   └── unet3plus_best.h5
├── templates/            # HTML templates
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   └── predict.html
├── static/               # Static files
│   ├── uploads/         # Uploaded images
│   └── results/         # Prediction results
└── Dataset/             # Training dataset
```

## Model

### U-Net Standar
Model segmentasi klasik dengan encoder-decoder architecture dan skip connections.

### Attention U-Net
U-Net dengan mekanisme attention untuk fokus pada fitur penting.

### U-Net Backbone
U-Net dengan pre-trained backbone untuk ekstraksi fitur lebih baik.

### U-Net3+
Arsitektur U-Net generasi ketiga dengan full-scale skip connections.

## Preprocessing
- **Resize**: 256x256 pixels
- **Normalisasi**: 0-1 range
- **Color Space**: RGB

## Metrik Evaluasi
- **Flood Percentage**: Persentase area yang terdeteksi banjir
- **Confidence Score**: Tingkat kepercayaan prediksi
- **Pixel Count**: Jumlah pixel banjir vs total pixel
- **Dice Coefficient**: Custom metric untuk training

## Screenshot
_(Tambahkan screenshot aplikasi di sini)_

## Author
Developed as part of Digital Image Processing (PCD) course assignment.

## License
Academic Project
