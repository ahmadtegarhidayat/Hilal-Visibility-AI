import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import requests
import math
import os
import warnings
from datetime import datetime
from skyfield.api import load, wgs84
from skyfield import almanac
from IPython.display import display, clear_output

# Machine Learning Libs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# ==============================================================================
# ⚙️ KONFIGURASI SISTEM (DEFAULT: UIN WALISONGO)
# ==============================================================================
FILENAME_CSV   = 'dataset.csv'
FILENAME_MODEL = 'model_klasifikasi_terbaiK.joblib'

# Default Location: Observatorium UIN Walisongo Semarang
DEF_NAME = "Observatorium UIN Walisongo"
DEF_LAT  = -6.995347
DEF_LON  = 110.347949
DEF_DATE = "2026-06-16"

# ==============================================================================
# 🧠 MODUL 1: TRAINING ENGINE (LOGISTIC REGRESSION)
# ==============================================================================
def train_model():
    print("\n[ SISTEM ] MEMULAI TRAINING MODEL BARU...")
    print("-" * 50)
    
    if not os.path.exists(FILENAME_CSV):
        print(f"❌ Error: File '{FILENAME_CSV}' tidak ditemukan.")
        return

    try:
        df = pd.read_csv(FILENAME_CSV, sep=';', encoding='latin1')
        if df.shape[1] < 2: df = pd.read_csv(FILENAME_CSV, sep=',', encoding='latin1')
        print(f"✅ Data dimuat: {len(df)} baris.")
    except Exception as e:
        print(f"❌ Gagal baca CSV: {e}"); return

    # 1. Standarisasi Nama Kolom (Pembersihan Data)
    rename_map = {
        'lluminasi (%)': 'Illuminasi', 'Illuminasi (%)': 'Illuminasi',
        'Suhu_Atmosfer_C': 'Suhu_Atmosfer_C', 'Kelembapan_Pct': 'Kelembapan_Pct',
        'Kecepatan_Angin_ms': 'Kecepatan_Angin_ms', 'Kondisi_Awan_Pct': 'Kondisi_Awan_Pct',
        'Transparansi_Index': 'Transparansi_Index'
    }
    df = df.rename(columns=rename_map)

    # 2. Definisikan Target
    col_target = 'Visibility'
    if col_target not in df.columns:
        print(f"❌ Kolom '{col_target}' tidak ada."); return
        
    df['Target'] = df[col_target].apply(lambda x: 1 if str(x) in ['1', '1.0', 'Terlihat', 'Visible'] else 0)

    # 3. Filter Fitur (Hanya Ambil Data Teknis)
    blacklist = ['No', 'ï»¿No', 'Latitude', 'longtitude', 'longitude', 'Lokasi', 'Kota', 
                 'Tanggal', 'Perukyat', 'Visibility', 'Target', 'Keterangan']
    
    X = df.drop(columns=[c for c in blacklist if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=['number']).fillna(0)
    y = df['Target']

    print(f"✅ Fitur Training ({len(X.columns)}): {list(X.columns)}")

    # 4. Arsitektur Model: StandardScaler + Logistic Regression
    # Menggunakan LogReg karena lebih baik dalam ekstrapolasi (data tinggi) dibanding Random Forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('model', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline.fit(X_train, y_train)

    # 5. Evaluasi & Simpan
    y_pred = pipeline.predict(X_test)
    print("\n📊 HASIL EVALUASI:")
    print(confusion_matrix(y_test, y_pred))
    
    joblib.dump(pipeline, FILENAME_MODEL)
    print(f"\n💾 Model Cerdas Berhasil Disimpan: {FILENAME_MODEL}")

# ==============================================================================
# 🔭 MODUL 2: ENGINE ASTRONOMI & CUACA
# ==============================================================================
def hitung_astronomi(lat, lon, tgl_str):
    print("🔭 Menghitung Hisab Posisi Bulan...")
    try:
        eph = load('de421.bsp')
    except:
        print("📥 Mendownload Ephemeris (de421.bsp)...")
        eph = load('de421.bsp')
        
    ts = load.timescale()
    topos = wgs84.latlon(lat, lon)
    sun, moon, earth = eph['sun'], eph['moon'], eph['earth']
    
    try:
        tgl_dt = datetime.strptime(tgl_str, "%Y-%m-%d")
        t_start = ts.utc(tgl_dt.year, tgl_dt.month, tgl_dt.day, 6) 
        t_end = ts.utc(tgl_dt.year, tgl_dt.month, tgl_dt.day, 15)
        
        f_geo = almanac.risings_and_settings(eph, sun, topos, horizon_degrees=0.0)
        times, events = almanac.find_discrete(t_start, t_end, f_geo)
        
        t_sunset = None
        for t, event in zip(times, events):
            if event == 0: t_sunset = t 
        
        if t_sunset is None: return None

        # Posisi saat Sunset
        astrometric_m = (earth + topos).at(t_sunset).observe(moon)
        apparent_m = astrometric_m.apparent()
        apparent_s = (earth + topos).at(t_sunset).observe(sun).apparent()
        
        alt_m, az_m, _ = apparent_m.altaz(pressure_mbar=0)
        alt_s, az_s, _ = apparent_s.altaz(pressure_mbar=0)
        
        # Lag Time & Parameter Lain
        f_moon = almanac.risings_and_settings(eph, moon, topos, horizon_degrees=0.0)
        t_search_next = ts.utc(tgl_dt.year, tgl_dt.month, tgl_dt.day + 1, 6)
        tm, evm = almanac.find_discrete(t_sunset, t_search_next, f_moon)
        lag_menit = 0
        for t, ev in zip(tm, evm):
            if ev == 0: lag_menit = (t - t_sunset) * 24 * 60

        dist_km = 384400 
        rad_deg = math.degrees(math.asin(1737.1 / dist_km))
        fraction = apparent_m.fraction_illuminated(sun)
        w_arcmin = (rad_deg * 2 * fraction) * 60

        return {
            'aD': alt_m.degrees - alt_s.degrees,
            'aL': apparent_m.separation_from(apparent_s).degrees,
            'DAz': az_m.degrees - az_s.degrees,
            'Lag': lag_menit,
            'w': w_arcmin,
            'Illuminasi': fraction * 100,
            'Moon_Alt': alt_m.degrees, 'Moon_Az': az_m.degrees,
            'Sun_Az': az_s.degrees, 'Waktu_Sunset': t_sunset
        }
    except Exception as e:
        print(f"❌ Error Hisab: {e}"); return None

def get_cuaca(use_api, lat, lon, tgl, jam_sunset):
    if use_api:
        print("☁️ Menghubungi Server Cuaca Open-Meteo...")
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,wind_speed_10m",
                "start_date": tgl, "end_date": tgl, "timezone": "UTC"
            }
            r = requests.get(url, params=params).json()
            idx = jam_sunset.utc_datetime().hour
            
            # Konversi km/h -> m/s
            ws_ms = r['hourly']['wind_speed_10m'][idx] * 0.27778
            
            return {
                'Suhu_Atmosfer_C': r['hourly']['temperature_2m'][idx],
                'Kelembapan_Pct': r['hourly']['relative_humidity_2m'][idx],
                'Kecepatan_Angin_ms': ws_ms,
                'Kondisi_Awan_Pct': r['hourly']['cloud_cover'][idx],
                'Transparansi_Index': 1.0 - (r['hourly']['cloud_cover'][idx]/100),
                'Sumber': 'API Real-time'
            }
        except:
            print("⚠️ Gagal koneksi API. Beralih ke Mode Ideal.")
    
    print("ℹ️ Menggunakan Parameter Cuaca IDEAL (Langit Bersih).")
    return {
        'Suhu_Atmosfer_C': 29.0, 'Kelembapan_Pct': 70.0,
        'Kecepatan_Angin_ms': 1.5, 'Kondisi_Awan_Pct': 0.0,
        'Transparansi_Index': 1.0, 'Sumber': 'Simulasi Ideal'
    }

# ==============================================================================
# 🔮 MODUL 3: ANTARMUKA PREDIKSI
# ==============================================================================
def predict_future():
    print("\n[ SISTEM ] MODUL PREDIKSI & VISUALISASI")
    print("-" * 50)
    
    if not os.path.exists(FILENAME_MODEL):
        print("❌ Model AI belum dilatih. Pilih Menu 1 dulu."); return
    
    model = joblib.load(FILENAME_MODEL)
    
    # --- 1. INPUT LOKASI ---
    print(f"Lokasi Saat Ini: {DEF_NAME}")
    print(f"Koordinat      : {DEF_LAT}, {DEF_LON}")
    ganti = input("Ganti lokasi? (y/n) [Default: n]: ").lower()
    
    if ganti == 'y':
        nama_lokasi = input("Nama Lokasi Baru : ")
        try:
            lat = float(input("Latitude         : ")) 
            lon = float(input("Longitude        : ")) 
        except ValueError:
            print("❌ Input angka salah. Kembali ke default."); return
    else:
        lat, lon, nama_lokasi = DEF_LAT, DEF_LON, DEF_NAME

    # --- 2. INPUT TANGGAL ---
    tgl_target = input(f"Tanggal (YYYY-MM-DD) [Default {DEF_DATE}]: ") or DEF_DATE

    # --- 3. INPUT CUACA ---
    print("\n--- OPSI DATA CUACA ---")
    print(" [1] Data Real-time (Untuk prediksi < 14 hari)")
    print(" [2] Simulasi Ideal (Untuk prediksi tahun depan)")
    pilihan_cuaca = input("Pilih (1/2) [Default: 2]: ")
    
    pilih_api = True if pilihan_cuaca == '1' else False

    # --- 4. PROSES HITUNG ---
    astro = hitung_astronomi(lat, lon, tgl_target)
    if not astro: return

    cuaca = get_cuaca(pilih_api, lat, lon, tgl_target, astro['Waktu_Sunset'])

    # Susun Data Input AI
    input_data = pd.DataFrame([{
        'aD': astro['aD'],
        'aL': astro['aL'],
        'DAz': astro['DAz'],
        'Lag': astro['Lag'],
        'w': astro['w'],
        'Illuminasi': astro['Illuminasi'],
        'Suhu_Atmosfer_C': cuaca['Suhu_Atmosfer_C'],
        'Kelembapan_Pct': cuaca['Kelembapan_Pct'],
        'Kecepatan_Angin_ms': cuaca['Kecepatan_Angin_ms'],
        'Kondisi_Awan_Pct': cuaca['Kondisi_Awan_Pct'],
        'Transparansi_Index': cuaca['Transparansi_Index']
    }])

    if hasattr(model, 'feature_names_in_'):
        input_data = input_data[model.feature_names_in_]

    # Prediksi AI
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[:, 1][0] * 100
    status = "TERLIHAT" if pred == 1 else "TIDAK TERLIHAT"

    # Tampilkan Laporan
    print("\n" + "="*45)
    print(f"🌙 HASIL PREDIKSI AI: {status}")
    print("="*45)
    print(f"📍 Lokasi  : {nama_lokasi}")
    print(f"📅 Tanggal : {tgl_target}")
    print(f"🔭 Altitude: {astro['aD']:.2f}°")
    print(f"📐 Elongasi: {astro['aL']:.2f}°")
    print(f"☁️ Cuaca   : {cuaca['Sumber']} (Awan: {cuaca['Kondisi_Awan_Pct']}%)")
    print(f"📈 Tingkat Keyakinan: {prob:.2f}%")
    print("-" * 45)

    clear_output(wait=True)

    # Visualisasi
    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='black', lw=2)
    plt.axhspan(-5, 0, color='#8B4513', alpha=0.3)
    plt.axhspan(0, 25, color='#191970', alpha=0.2)
    
    plt.scatter(astro['Sun_Az'], -0.5, s=500, c='orange', alpha=0.6, label='Matahari')
    col_moon = 'gold' if pred == 1 else 'gray'
    plt.scatter(astro['Moon_Az'], astro['Moon_Alt'], s=250, c=col_moon, edgecolors='white', label=f'Hilal ({status})')
    
    plt.title(f"Visualisasi Hilal: {nama_lokasi}\n{tgl_target} | Probabilitas: {prob:.1f}%")
    plt.xlabel("Azimuth (Arah Mata Angin)"); plt.ylabel("Altitude (Ketinggian)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    mid = (astro['Sun_Az'] + astro['Moon_Az']) / 2
    plt.xlim(mid - 8, mid + 8)
    plt.ylim(-3, max(astro['Moon_Alt']+5, 5))
    
    display(plt.gcf())
    plt.close()
    print("\n" + "="*45)
    print(f"🌙 HASIL PREDIKSI AI: {status}")
    print("="*45)
    print(f"📈 Tingkat Keyakinan: {prob:.2f}%")

# ==============================================================================
# 🚀 MENU UTAMA
# ==============================================================================
if __name__ == "__main__":
    while True:
        print("\n=== 🌙 SISTEM HILAL AI ===")
        print("1. Latih Model AI (Training)")
        print("2. Prediksi Hilal & Visualisasi")
        print("3. Keluar")
        
        p = input("Pilih menu: ")
        if p == '1': train_model()
        elif p == '2': predict_future()
        elif p == '3': 
            print("👋 Wassalamualaikum."); break
        else: print("❌ Pilihan tidak valid.")







