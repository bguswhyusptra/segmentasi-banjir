import os
import time
import cv2
import sqlite3
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename

# Import TensorFlow dengan error handling
try:
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    print(f"Warning: TensorFlow import failed: {e}")
    print("Starting in limited mode - prediction features disabled")
    TENSORFLOW_AVAILABLE = False
    K = None
    load_model = None

# =========================
# KONFIGURASI
# =========================
IMG_HEIGHT = 256
IMG_WIDTH = 256

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
DATABASE = 'users.db'

UNET_MODEL_PATH = 'model/unet_flood_model_final.h5'
ATT_MODEL_PATH  = 'model/attention_unet_flood.h5'
BACKBONE_MODEL_PATH = 'model/my_model.h5'
UNET3PLUS_MODEL_PATH = 'model/unet3plus_best.h5'

MODEL_PATHS = {
    'U-Net': UNET_MODEL_PATH,
    'Attention U-Net': ATT_MODEL_PATH,
    'U-Net Backbone': BACKBONE_MODEL_PATH,
    'U-Net3+': UNET3PLUS_MODEL_PATH,
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder='Menu')
app.secret_key = "flood_secret_key"

# =========================
# DATABASE INIT
# =========================
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# =========================
# CUSTOM METRIC
# =========================
def dice_coef(y_true, y_pred, smooth=1e-6):
    if not TENSORFLOW_AVAILABLE:
        return 0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# =========================
# LOAD MODEL
# =========================
if TENSORFLOW_AVAILABLE:
    try:
        unet_model = load_model(
            UNET_MODEL_PATH,
            custom_objects={'dice_coef': dice_coef}
        )

        attention_model = load_model(
            ATT_MODEL_PATH,
            custom_objects={'dice_coef': dice_coef}
        )

        backbone_model = load_model(
            BACKBONE_MODEL_PATH,
            custom_objects={'dice_coef': dice_coef}
        )

        unet3plus_model = load_model(
            UNET3PLUS_MODEL_PATH,
            custom_objects={'dice_coef': dice_coef},
            compile=False
        )

        print("✓ UNet, Attention UNet, UNet Backbone & UNet3+ loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        TENSORFLOW_AVAILABLE = False
else:
    print("✗ Models not loaded - TensorFlow unavailable")
    unet_model = None
    attention_model = None
    backbone_model = None
    unet3plus_model = None

# =========================
# IMAGE UTILS
# =========================
def preprocess_image(path):
    """Prepare an uploaded image for the segmentation models."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError("File bukan gambar yang valid atau tidak dapat dibaca")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def postprocess_mask(pred):
    mask = pred[0, :, :, 0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

def calculate_metrics(pred):
    """Hitung metrik dari hasil prediksi"""
    mask = pred[0, :, :, 0]
    binary_mask = (mask > 0.5).astype(np.float32)
    
    # Persentase area banjir
    flood_percentage = np.mean(binary_mask) * 100
    
    # Confidence score (rata-rata nilai prediksi pada area banjir)
    flood_pixels = mask[binary_mask == 1]
    confidence = np.mean(flood_pixels) * 100 if len(flood_pixels) > 0 else 0
    
    return {
        'flood_percentage': round(flood_percentage, 2),
        'confidence': round(confidence, 2),
        'total_pixels': mask.shape[0] * mask.shape[1],
        'flood_pixels': int(np.sum(binary_mask))
    }


def run_prediction(model, img):
    start_time = time.perf_counter()
    pred = model.predict(img, verbose=0)
    inference_seconds = time.perf_counter() - start_time
    mask = postprocess_mask(pred)
    metrics = calculate_metrics(pred)
    metrics['inference_seconds'] = round(inference_seconds, 4)
    return pred, mask, metrics

# =========================
# LOGIN
# =========================
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            return render_template('login.html', error="Username dan Password harus diisi")

        conn = sqlite3.connect(DATABASE, timeout=10)
        try:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username=? AND password=?",
                      (username, password))
            user = c.fetchone()

            if user:
                session['user'] = username
                return redirect(url_for('dashboard'))
            else:
                return render_template('login.html', error="Username / Password salah")
        finally:
            conn.close()

    return render_template('login.html')

# =========================
# REGISTER
# =========================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            return render_template('register.html', error="Username dan Password harus diisi")

        conn = sqlite3.connect(DATABASE, timeout=10)
        try:
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?,?)",
                      (username, password))
            conn.commit()
            return render_template('register.html', success="Registrasi berhasil! Silakan login.")
        except sqlite3.IntegrityError:
            return render_template('register.html', error="Username sudah ada")
        finally:
            conn.close()

    return render_template('register.html')

# =========================
# LOGOUT
# =========================
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# =========================
# DASHBOARD
# =========================
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

# =========================
# PREDICT (2 MODEL)
# =========================
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if not TENSORFLOW_AVAILABLE:
        return render_template('predict.html', 
                             error="Prediksi tidak tersedia - TensorFlow tidak dapat dimuat")

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('predict.html', error="No file")

        file = request.files['image']
        filename = secure_filename(file.filename)
        if not filename:
            return render_template('predict.html', error="Nama file tidak valid")

        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        try:
            img = preprocess_image(upload_path)
        except (ValueError, cv2.error) as e:
            return render_template('predict.html', error=str(e))

        # UNET
        unet_pred, unet_mask, unet_metrics = run_prediction(unet_model, img)
        unet_path = os.path.join(RESULT_FOLDER, f"unet_{filename}")
        cv2.imwrite(unet_path, unet_mask)

        # ATTENTION UNET
        att_pred, att_mask, att_metrics = run_prediction(attention_model, img)
        att_path = os.path.join(RESULT_FOLDER, f"attention_{filename}")
        cv2.imwrite(att_path, att_mask)

        # UNET BACKBONE
        backbone_pred, backbone_mask, backbone_metrics = run_prediction(backbone_model, img)
        backbone_path = os.path.join(RESULT_FOLDER, f"backbone_{filename}")
        cv2.imwrite(backbone_path, backbone_mask)

        # UNET3PLUS
        unet3plus_pred, unet3plus_mask, unet3plus_metrics = run_prediction(unet3plus_model, img)
        unet3plus_path = os.path.join(RESULT_FOLDER, f"unet3plus_{filename}")
        cv2.imwrite(unet3plus_path, unet3plus_mask)

        comparison_rows = [
            {
                'model': 'U-Net',
                'model_size_mb': round(os.path.getsize(MODEL_PATHS['U-Net']) / (1024 ** 2), 2),
                'parameters': unet_model.count_params(),
                'inference_seconds': unet_metrics['inference_seconds'],
                'flood_percentage': unet_metrics['flood_percentage'],
                'confidence': unet_metrics['confidence'],
            },
            {
                'model': 'Attention U-Net',
                'model_size_mb': round(os.path.getsize(MODEL_PATHS['Attention U-Net']) / (1024 ** 2), 2),
                'parameters': attention_model.count_params(),
                'inference_seconds': att_metrics['inference_seconds'],
                'flood_percentage': att_metrics['flood_percentage'],
                'confidence': att_metrics['confidence'],
            },
            {
                'model': 'U-Net Backbone',
                'model_size_mb': round(os.path.getsize(MODEL_PATHS['U-Net Backbone']) / (1024 ** 2), 2),
                'parameters': backbone_model.count_params(),
                'inference_seconds': backbone_metrics['inference_seconds'],
                'flood_percentage': backbone_metrics['flood_percentage'],
                'confidence': backbone_metrics['confidence'],
            },
            {
                'model': 'U-Net3+',
                'model_size_mb': round(os.path.getsize(MODEL_PATHS['U-Net3+']) / (1024 ** 2), 2),
                'parameters': unet3plus_model.count_params(),
                'inference_seconds': unet3plus_metrics['inference_seconds'],
                'flood_percentage': unet3plus_metrics['flood_percentage'],
                'confidence': unet3plus_metrics['confidence'],
            },
        ]

        return render_template(
            'predict.html',
            original_image=f"/{UPLOAD_FOLDER}/{filename}",
            unet_result=f"/{unet_path}",
            attention_result=f"/{att_path}",
            backbone_result=f"/{backbone_path}",
            unet3plus_result=f"/{unet3plus_path}",
            unet_metrics=unet_metrics,
            att_metrics=att_metrics,
            backbone_metrics=backbone_metrics,
            unet3plus_metrics=unet3plus_metrics,
            comparison_rows=comparison_rows
        )

    return render_template('predict.html')

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=False)  # Set False untuk startup lebih cepat
