import os
import cv2
import sqlite3
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

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

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
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
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# =========================
# LOAD MODEL
# =========================
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

print("UNet, Attention UNet, UNet Backbone & UNet3+ loaded")

# =========================
# IMAGE UTILS
# =========================
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
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

# =========================
# LOGIN
# =========================
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

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
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DATABASE, timeout=10)
        try:
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?,?)",
                      (username, password))
            conn.commit()
            return redirect(url_for('login'))
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

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('predict.html', error="No file")

        file = request.files['image']
        filename = file.filename

        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        img = preprocess_image(upload_path)

        # UNET
        unet_pred = unet_model.predict(img)
        unet_mask = postprocess_mask(unet_pred)
        unet_path = os.path.join(RESULT_FOLDER, f"unet_{filename}")
        cv2.imwrite(unet_path, unet_mask)
        unet_metrics = calculate_metrics(unet_pred)

        # ATTENTION UNET
        att_pred = attention_model.predict(img)
        att_mask = postprocess_mask(att_pred)
        att_path = os.path.join(RESULT_FOLDER, f"attention_{filename}")
        cv2.imwrite(att_path, att_mask)
        att_metrics = calculate_metrics(att_pred)

        # UNET BACKBONE
        backbone_pred = backbone_model.predict(img)
        backbone_mask = postprocess_mask(backbone_pred)
        backbone_path = os.path.join(RESULT_FOLDER, f"backbone_{filename}")
        cv2.imwrite(backbone_path, backbone_mask)
        backbone_metrics = calculate_metrics(backbone_pred)

        # UNET3PLUS
        unet3plus_pred = unet3plus_model.predict(img)
        unet3plus_mask = postprocess_mask(unet3plus_pred)
        unet3plus_path = os.path.join(RESULT_FOLDER, f"unet3plus_{filename}")
        cv2.imwrite(unet3plus_path, unet3plus_mask)
        unet3plus_metrics = calculate_metrics(unet3plus_pred)

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
            unet3plus_metrics=unet3plus_metrics
        )

    return render_template('predict.html')

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=False)  # Set False untuk startup lebih cepat
