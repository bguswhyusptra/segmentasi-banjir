import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session

# =========================
# KONFIGURASI
# =========================
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
DATABASE = 'users.db'

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
# PREDICT (PLACEHOLDER)
# =========================
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Lazy import TensorFlow hanya saat predict dipanggil
    try:
        import cv2
        import numpy as np
        from tensorflow.keras import backend as K
        from tensorflow.keras.models import load_model
        
        # Import berhasil, load models dan lakukan prediksi
        # (Kode prediksi lengkap ada di app_with_models.py)
        return render_template('predict.html', 
                             error="Fitur prediksi belum dimuat. Gunakan app_with_models.py untuk prediksi.")
    except Exception as e:
        return render_template('predict.html', 
                             error=f"TensorFlow error: {str(e)[:100]}... Silakan install ulang TensorFlow.")

    return render_template('predict.html')

# =========================
# RUN
# =========================
if __name__ == '__main__':
    print("="*60)
    print("🚀 APLIKASI FLOOD SEGMENTATION - MODE LITE")
    print("="*60)
    print("✓ Login dan Register: AKTIF")
    print("✗ Prediksi Model: NONAKTIF (TensorFlow issue)")
    print("\n💡 Untuk testing login/register, aplikasi siap digunakan")
    print("💡 Untuk prediksi lengkap, perbaiki instalasi TensorFlow")
    print("="*60)
    app.run(debug=True, port=5000)
