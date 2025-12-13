"""
================================================================================
🌙 SISTEM PREDIKSI VISIBILITAS HILAL MENGGUNAKAN ALGORITMA MACHINE LEARNING
================================================================================
👨‍💻 CREATOR   : Ahmad Tegar Hidayat
🎓 STATUS    : Mahasiswa Ilmu Falak UIN Walisongo Semarang
📧 EMAIL     : ahmadtegar0809@gmail.com
================================================================================
"""

import os
import sys
import time
import warnings
import joblib
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.patches import Circle

# ==============================================================================
# DEPENDENCY CHECKER
# ==============================================================================
def check_library(lib_name, install_cmd):
    """Memastikan library yang dibutuhkan sudah terinstall."""
    try:
        module_name = "sklearn" if lib_name == "scikit-learn" else lib_name.replace("-", "_")
        __import__(module_name)
    except ImportError:
        print(f"❌ Library '{lib_name}' belum terinstall.")
        print(f"   Silakan install manual: pip install {install_cmd}")
        sys.exit()

# Cek library eksternal utama
check_library("hijridate", "hijridate") 
check_library("skyfield", "skyfield")
check_library("scikit-learn", "scikit-learn")

# Import modul setelah pengecekan
from hijridate import Gregorian
from skyfield.api import load, wgs84
from skyfield import almanac
from sklearn.ensemble import GradientBoostingClassifier  # <-- Penting!
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
warnings.filterwarnings('ignore')

# ==============================================================================
# KONFIGURASI SISTEM (CONSTANTS)
# ==============================================================================
FILENAME_CSV    = 'data_set_rukyat_ml.csv'
FILENAME_MODEL  = 'model_hilal_rf.pkl'

# Lokasi Default: Observatorium UIN Walisongo
DEF_NAME        = "Observatorium UIN Walisongo (Semarang)"
DEF_LAT         = -6.995347
DEF_LON         = 110.347949
DEF_DATE        = datetime.now().strftime("%Y-%m-%d")

# Identitas Creator
CREATOR_NAME    = "Ahmad Tegar Hidayat"
CREATOR_EMAIL   = "ahmadtegar0809@gmail.com"
CREATOR_AFILIASI= "Ilmu Falak - UIN Walisongo Semarang"

NAMA_BULAN_HIJRI = [
    "Muharram", "Safar", "Rabiul Awal", "Rabiul Akhir",
    "Jumadil Awal", "Jumadil Akhir", "Rajab", "Sya'ban",
    "Ramadhan", "Syawal", "Dzulqa'dah", "Dzulhijjah"
]

# ==============================================================================
# TAMPILAN & UTILITAS UI
# ==============================================================================
def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def pause_return_menu():
    print("\n" + "-"*50)
    input("👉 Tekan [ENTER] untuk kembali ke Menu Utama...")

def maximize_window():
    try:
        manager = plt.get_current_fig_manager()
        if os.name == 'nt': manager.window.state('zoomed')
        else: manager.window.showMaximized()
    except: pass

def tampilkan_plot_universal(figure):
    maximize_window()
    print("📊 Dashboard Visualisasi Terbuka (Cek Jendela Baru)...")
    plt.tight_layout()
    plt.show()

def tampilkan_menu_utama():
    """Menampilkan desain menu utama dengan Identitas Creator."""
    clear_screen()
    print("\033[1;36m" + "="*75 + "\033[0m")
    print("🌙 SISTEM PREDIKSI VISIBILITAS HILAL MENGGUNAKAN ALGORITMA MACHINE LEARNING")
    print("\033[1;36m" + "="*75 + "\033[0m")
    
    print(f" 👨‍💻 Creator  : {CREATOR_NAME}")
    print(f" 🎓 Afiliasi : {CREATOR_AFILIASI}")
    print(f" 📧 Email    : {CREATOR_EMAIL}")
    print("-" * 75)
    
    print("\n\033[1;33m[ PILIHAN MENU ]\033[0m")
    print(" 1. 🤖 Training Model AI (Wajib dijalankan pertama kali)")
    print(" 2. 🌙 Prediksi & Dashboard Visibilitas Hilal")
    print(" 3. 📂 Generator Data Hisab Massal (CSV)")
    print(" 4. 🚪 Keluar Aplikasi")
    print("\n" + "-"*75)

# ==============================================================================
#    ENGINE ASTRONOMI (SKYFIELD)
# ==============================================================================
def hitung_astronomi(lat, lon, tgl_str, verbose=True):
    if verbose: print("🔭 Menghitung data hisab ephemeris...", end="\r")
    
    try: 
        try:
            eph = load('de421.bsp')
        except: 
            if verbose: print("\n📥 Mendownload file ephemeris (de421.bsp)...")
            eph = load('de421.bsp')
        
        ts = load.timescale()
        topos = wgs84.latlon(lat, lon)
        sun, moon, earth = eph['sun'], eph['moon'], eph['earth']
        
        tgl_dt = datetime.strptime(tgl_str, "%Y-%m-%d")
        t_start_sun = ts.utc(tgl_dt.year, tgl_dt.month, tgl_dt.day, 10) 
        t_end_sun   = ts.utc(tgl_dt.year, tgl_dt.month, tgl_dt.day, 15)
        
        f_geo_sun = almanac.risings_and_settings(eph, sun, topos, horizon_degrees=-0.8333)
        times_s, events_s = almanac.find_discrete(t_start_sun, t_end_sun, f_geo_sun)
        
        if len(times_s) == 0:
            if verbose: print("\n⚠️ Error: Tidak ditemukan waktu sunset.")
            return None
        
        t_sunset = times_s[0]

        observer = (earth + topos).at(t_sunset)
        m = observer.observe(moon).apparent()
        s = observer.observe(sun).apparent()
        alt_m, az_m, _ = m.altaz()
        alt_s, az_s, _ = s.altaz()

        f_geo_moon = almanac.risings_and_settings(eph, moon, topos, horizon_degrees=-0.8333)
        
        # Logika Moonset
        if alt_m.degrees > 0:
            t_search_end = ts.utc(tgl_dt.year, tgl_dt.month, tgl_dt.day, t_sunset.utc_datetime().hour + 6)
            times_m, _ = almanac.find_discrete(t_sunset, t_search_end, f_geo_moon)
            t_moonset = times_m[0] if len(times_m) > 0 else None
        else:
            t_search_start = ts.utc(tgl_dt.year, tgl_dt.month, tgl_dt.day, t_sunset.utc_datetime().hour - 6)
            times_m, _ = almanac.find_discrete(t_search_start, t_sunset, f_geo_moon)
            t_moonset = times_m[-1] if len(times_m) > 0 else None

        if t_moonset is not None:
            lag_minutes = (t_moonset.utc_datetime() - t_sunset.utc_datetime()).total_seconds() / 60
        else:
            lag_minutes = alt_m.degrees * 4 

        illumination_fraction = m.fraction_illuminated(sun)
        w_moa = 31.0 * illumination_fraction 
        local_sunset = t_sunset.astimezone(datetime.now().astimezone().tzinfo)

        # Hitung Ijtimak / Fase Bulan Baru
        t_phase_start = ts.utc(tgl_dt.year, tgl_dt.month, tgl_dt.day - 2)
        t_phase_end = ts.utc(tgl_dt.year, tgl_dt.month, tgl_dt.day + 1)
        phases = almanac.moon_phases(eph)
        t_phases, y_phases = almanac.find_discrete(t_phase_start, t_phase_end, phases)

        ijtimak_str = "Belum terjadi"
        moon_age_str = "-"
        
        for t_p, phase in zip(t_phases, y_phases):
            if phase == 0: 
                if t_p.utc_datetime() < t_sunset.utc_datetime():
                    age_hours = (t_sunset.utc_datetime() - t_p.utc_datetime()).total_seconds() / 3600
                    ijtimak_local = t_p.astimezone(datetime.now().astimezone().tzinfo)
                    ijtimak_str = ijtimak_local.strftime('%H:%M:%S WIB')
                    moon_age_str = f"{age_hours:.2f} Jam"

        return {
            'aD': alt_m.degrees, 'aL': m.separation_from(s).degrees, 'DAz': az_m.degrees - az_s.degrees,
            'Lag': lag_minutes, 'Illuminasi': illumination_fraction * 100, 'w': w_moa,
            'Moon_Alt': alt_m.degrees, 'Moon_Az': az_m.degrees, 'Sun_Az': az_s.degrees,
            'Waktu_Sunset_Lokal': local_sunset.strftime('%H:%M:%S'),
            'Jam_Sunset_Lokal_Int': local_sunset.hour,
            'Ijtimak': ijtimak_str, 'Umur_Bulan': moon_age_str
        }
    except Exception as e: 
        if verbose: print(f"\n❌ Terjadi kesalahan hisab: {e}")
        return None

# ==============================================================================
#   ENGINE CUACA (OPEN-METEO API)
# ==============================================================================
def get_cuaca(use_api, lat, lon, tgl, jam_sunset_lokal=18):
    # Data Default / Simulasi
    data = {
        'Suhu_Atmosfer_C': 28.0, 'Kelembapan_Pct': 75.0, 'Kondisi_Awan_Pct': 10.0, 
        'Kecepatan_Angin_ms': 2.5, 'Transparansi_Index': 0.90,
        'Deskripsi': 'Simulasi Ideal', 'Sumber': 'Simulasi'
    }
    
    if use_api:
        print("☁️ Mengambil data cuaca real-time (API)...")
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat, "longitude": lon, 
                "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,weathercode,wind_speed_10m", 
                "start_date": tgl, "end_date": tgl, "timezone": "auto"
            }
            
            r = requests.get(url, params=params, timeout=5).json()
            if 'error' in r and r['error']: raise Exception(f"API Error: {r['reason']}")

            idx = min(jam_sunset_lokal, 23)
            wind_ms = r['hourly']['wind_speed_10m'][idx] / 3.6
            awan_pct = r['hourly']['cloud_cover'][idx]
            transparansi = 1.0 - (awan_pct / 100.0)
            
            return {
                'Suhu_Atmosfer_C': r['hourly']['temperature_2m'][idx],
                'Kelembapan_Pct': r['hourly']['relative_humidity_2m'][idx],
                'Kondisi_Awan_Pct': awan_pct,
                'Kecepatan_Angin_ms': round(wind_ms, 2),
                'Transparansi_Index': round(transparansi, 2),
                'Deskripsi': f"WMO Code {r['hourly']['weathercode'][idx]}",
                'Sumber': 'API Open-Meteo'
            }
        except Exception as e: 
            print(f"⚠️ Gagal terhubung API ({e}). Menggunakan mode Simulasi Ideal.")
            
    return data

# ==============================================================================
#    MENU 3: GENERATOR DATA HISAB MASSAL
# ==============================================================================
def generate_bulk_hisab():
    print("\n[ MODUL 3 ] GENERATOR DATA HISAB MASSAL")
    print("-" * 50)
    input_file = input("📂 Masukkan nama file CSV input (misal: input.csv): ")
    if not os.path.exists(input_file):
        print("❌ File tidak ditemukan."); pause_return_menu(); return

    try: 
        with open(input_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        sep = ';' if ';' in first_line else ','
        df = pd.read_csv(input_file, sep=sep)
    except Exception as e: 
        print(f"❌ Gagal baca CSV: {e}"); pause_return_menu(); return

    if 'Tanggal' not in df.columns:
        print("❌ Kolom 'Tanggal' tidak ditemukan."); pause_return_menu(); return

    print("\n🚀 Memulai Perhitungan Hisab Massal...")
    results = []
    
    total_data = len(df)
    for i, row in df.iterrows():
        tgl = row['Tanggal']
        lat = float(row['Latitude']) if 'Latitude' in row and pd.notnull(row['Latitude']) else DEF_LAT
        lon = float(row['Longitude']) if 'Longitude' in row and pd.notnull(row['Longitude']) else DEF_LON
        
        astro = hitung_astronomi(lat, lon, tgl, verbose=False)
        
        if astro:
            cuaca = get_cuaca(False, lat, lon, tgl, astro['Jam_Sunset_Lokal_Int'])
            results.append({
                'Tanggal': tgl, 'Latitude': lat, 'Longitude': lon,
                'aD': round(astro['aD'], 4), 'aL': round(astro['aL'], 4),
                'DAz': round(astro['DAz'], 4), 'Lag': round(astro['Lag'], 2),
                'w': round(astro['w'], 4), 'Illuminasi': round(astro['Illuminasi'], 2),
                'Moon_Alt': round(astro['Moon_Alt'], 4),
                'Suhu_Atmosfer_C': cuaca['Suhu_Atmosfer_C'],
                'Kelembapan_Pct': cuaca['Kelembapan_Pct'],
                'Kecepatan_Angin_ms': cuaca['Kecepatan_Angin_ms'],
                'Kondisi_Awan_Pct': cuaca['Kondisi_Awan_Pct'],
                'Transparansi_Index': cuaca['Transparansi_Index'],
                'Waktu_Sunset': astro['Waktu_Sunset_Lokal']
            })
            
        if (i+1) % 10 == 0 or (i+1) == total_data:
            print(f"   ⏳ Proses: {i+1}/{total_data} data selesai...", end="\r")

    output_df = pd.DataFrame(results)
    output_filename = f"hasil_hisab_{int(time.time())}.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"\n\n✅ Selesai! Data disimpan ke: {output_filename}")
    pause_return_menu()

# ==============================================================================
# MENU 1: TRAINING AI (UPDATED: GRADIENT BOOSTING)
# ==============================================================================
def train_model():
    print("\n[ MODUL 1 ] TRAINING & EVALUASI MODEL (GRADIENT BOOSTING)")
    print("-" * 50)
    
    if not os.path.exists(FILENAME_CSV):
        print(f"❌ ERROR: File '{FILENAME_CSV}' tidak ditemukan!"); pause_return_menu(); return

    print("📂 Memuat dataset...")
    try: 
        df = pd.read_csv(FILENAME_CSV, sep=';', encoding='latin1', thousands='.', decimal=',')
    except Exception as e: 
        print(f"❌ Gagal membaca CSV: {e}"); pause_return_menu(); return

    df.columns = df.columns.str.replace('ï»¿', '').str.strip()
    
    # Normalisasi data
    if 'aD' in df.columns and df['aD'].max() > 100: 
        for c in ['aD', 'aL', 'DAz', 'Lag']: 
            if c in df.columns: df[c] = df[c] / 1000

    if 'Visibility' not in df.columns: 
        print("❌ Kolom 'Visibility' tidak ditemukan."); pause_return_menu(); return
        
    df['Target'] = df['Visibility'].apply(lambda x: 1 if x > 0 else 0)
    features = ['aD', 'aL', 'DAz', 'Lag', 'Suhu_Atmosfer_C', 'Kelembapan_Pct', 'Kondisi_Awan_Pct']
    
    for f in features:
        if f not in df.columns: df[f] = 0

    X = df[features].fillna(0)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"🤖 Melatih Gradient Boosting dengan {len(X_train)} data latih...")
    
    #
    pipeline = Pipeline([
        ('model', GradientBoostingClassifier(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=3, 
            random_state=42
        ))
    ])
    # -----------------------------------
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100

    print("\n" + "="*50)
    print("📊 HASIL EVALUASI MODEL")
    print("="*50)
    print(f"  • Akurasi   : {acc:.2f}%")
    print(f"  • F1-Score  : {f1:.2f}%")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=['Tidak Terlihat', 'Terlihat']))
    print("="*50)

    joblib.dump({'pipeline': pipeline, 'accuracy': acc, 'f1': f1}, FILENAME_MODEL)
    print(f"✅ Model Gradient Boosting berhasil disimpan ke: {FILENAME_MODEL}")
    pause_return_menu()

# ==============================================================================
#    MENU 2: PREDIKSI & DASHBOARD
# ==============================================================================
def predict_future():
    print("\n[ MODUL 2 ] PREDIKSI & VISUALISASI")
    print("-" * 50)
    
    if not os.path.exists(FILENAME_MODEL): 
        print("❌ ERROR: Harap Training Model (Menu 1) terlebih dahulu."); pause_return_menu(); return
        
    loaded_data = joblib.load(FILENAME_MODEL)
    model = loaded_data['pipeline'] if isinstance(loaded_data, dict) else loaded_data
    
    ganti = input(f"Gunakan Lokasi Default ({DEF_NAME})? (y/n): ").lower()
    if ganti == 'n':
        nama_lokasi = input("Nama Lokasi : ")
        try: 
            lat = float(input("Latitude    : "))
            lon = float(input("Longitude   : ")) 
        except: print("Input salah!"); return
    else: 
        lat, lon, nama_lokasi = DEF_LAT, DEF_LON, DEF_NAME

    tgl_target = input(f"Tanggal (YYYY-MM-DD) [Default {DEF_DATE}]: ") or DEF_DATE
    
    try:
        y_g, m_g, d_g = map(int, tgl_target.split('-'))
        h_today = Gregorian(y_g, m_g, d_g).to_hijri()
        str_hijri = f"{h_today.day} {NAMA_BULAN_HIJRI[h_today.month - 1]} {h_today.year} H"
    except ValueError:
        print("Format tanggal salah!"); pause_return_menu(); return

    pilih_api = True if input("Mode Cuaca [1] Real-time API  [2] Simulasi Ideal : ") == '1' else False
    
    astro = hitung_astronomi(lat, lon, tgl_target)
    if not astro: pause_return_menu(); return
    
    cuaca = get_cuaca(pilih_api, lat, lon, tgl_target, astro['Jam_Sunset_Lokal_Int'])
    
    # --- PROSES MACHINE LEARNING ---
    input_data = pd.DataFrame([{
        'aD': astro['aD'], 'aL': astro['aL'], 'DAz': astro['DAz'], 'Lag': astro['Lag'],
        'Suhu_Atmosfer_C': cuaca['Suhu_Atmosfer_C'], 'Kelembapan_Pct': cuaca['Kelembapan_Pct'], 'Kondisi_Awan_Pct': cuaca['Kondisi_Awan_Pct']
    }])
    
    if hasattr(model, 'feature_names_in_'):
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # HITUNG PROBABILITAS (PERSENTASE KETERLIHATAN)
    prob_ai = model.predict_proba(input_data)[:, 1][0] * 100
    
    # --- LOGIKA KEPUTUSAN HYBRID (AI + FISIKA) ---
    # 1. Cek Fisika Dasar (Qobla Ghurub)
    if astro['Lag'] < 0:
        prob_ai = 0.0
        status_ai = "TIDAK TERLIHAT (Qobla Ghurub)"
        
    # 2. Cek Kondisi Cuaca Ekstrem (Logika Manusia)
    # Jika awan > 95% (Hampir total), mustahil terlihat meski posisi hilal tinggi
    elif cuaca['Kondisi_Awan_Pct'] >= 95:
        prob_ai = 1.0 # Probabilitas melihat sangat kecil
        status_ai = "TERHALANG CUACA (Awan Tebal)"
        
    # 3. Keputusan AI (Jika cuaca mendukung)
    else:
        # Jika probabilitas AI tinggi, tapi awan lumayan tebal, kurangi confidence
        factor_pengurang = 0
        if cuaca['Kondisi_Awan_Pct'] > 50:
            factor_pengurang = (cuaca['Kondisi_Awan_Pct'] - 50) / 2
            
        final_prob = prob_ai - factor_pengurang
        if final_prob < 0: final_prob = 0

        # Penentuan Status Akhir
        if final_prob >= 50:
            status_ai = "KEMUNGKINAN TERLIHAT"
        else:
            status_ai = "KEMUNGKINAN TIDAK TERLIHAT"

    # ==========================================
    #        PENENTUAN AWAL BULAN
    # ==========================================
    idx_bulan_depan = h_today.month % 12 
    nama_bulan_baru = NAMA_BULAN_HIJRI[idx_bulan_depan]
    tgl_objek = datetime.strptime(tgl_target, "%Y-%m-%d")

    if "TIDAK TERLIHAT" in status_ai or "TERHALANG" in status_ai:
        # Istikmal (LUSA)
        tgl_awal_bulan = tgl_objek + timedelta(days=2)
        ket_awal_bulan = f"1 {nama_bulan_baru} (Istikmal)"
    else:
        # Rukyatul Hilal 
        tgl_awal_bulan = tgl_objek + timedelta(days=1)
        ket_awal_bulan = f"1 {nama_bulan_baru} (Rukyatul Hilal)"

    next_masehi = tgl_awal_bulan.strftime("%d %B %Y")
    # ==========================================

    # --- DASHBOARD VISUALISASI ---
    fig = plt.figure(figsize=(13, 7), facecolor='#f8f9fa')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.2]) 
    
    ax_plot = fig.add_subplot(gs[0])
    ax_plot.set_facecolor('#0f172a')
    
    # 1. Gambar Elemen Dasar
    ax_plot.axhline(0, color='#22c55e', lw=2, alpha=0.8, zorder=20) 
    
    # 2. Plot Matahari (Sun)
    ax_plot.scatter(astro['Sun_Az'], -0.83, s=800, c='#f59e0b', edgecolors='#ef4444', label='Matahari', zorder=5)
    ax_plot.text(astro['Sun_Az'], -0.83 - 1.5, "Matahari", color='#f59e0b', ha='center', fontsize=9)

    # 3. Plot Bulan (Moon)
    col_moon = '#eab308' if "TERLIHAT" in status_ai else '#94a3b8' 
    ax_plot.scatter(astro['Moon_Az'], astro['Moon_Alt'], s=450, c=col_moon, edgecolors='white', label='Bulan', zorder=10)
    
    # Label Text Bulan
    offset_text = 1.2 if astro['Moon_Alt'] > -0.83 else -1.8 
    ax_plot.text(astro['Moon_Az'], astro['Moon_Alt'] + offset_text, f"Bulan: {astro['aD']:.2f}°", 
                 color='#22d3ee', ha='center', fontsize=10, fontweight='bold', zorder=11)

    # 4. Limit & Garis Bantu
    ax_plot.axhline(3.0, color='red', linestyle='--', linewidth=1.5, label='Limit Alt 3°')
    limit_circle = Circle((astro['Sun_Az'], -0.83), 6.4, color='yellow', fill=False, linestyle='--', label='Limit Elongasi 6.4°')
    ax_plot.add_patch(limit_circle)

    # 5. Setting Tampilan & Zoom Otomatis
    ax_plot.set_title(f"VISUALISASI POSISI HILAL\n{nama_lokasi}", color='black', fontsize=13, fontweight='bold', pad=15)
    ax_plot.set_xlabel("Azimuth", fontsize=10); ax_plot.set_ylabel("Altitude", fontsize=10)
    ax_plot.grid(True, linestyle=':', alpha=0.3, color='white')
    
    # Tanah (Ground) - Warnai area di bawah 0 derajat
    y_min_dynamic = min(astro['Moon_Alt'] - 3, -5) 
    ax_plot.axhspan(y_min_dynamic, 0, color='#3f2e18', alpha=0.9, zorder=1) 

    # Atur Zoom Kamera
    ax_plot.set_ylim(y_min_dynamic, max(astro['Moon_Alt'] + 6, 9))
    
    mid_az = (astro['Sun_Az'] + astro['Moon_Az']) / 2
    ax_plot.set_xlim(mid_az - 9, mid_az + 9)
    ax_plot.legend(loc='upper right', fontsize=8, facecolor='white', framealpha=0.9)

    # --- PANEL INFO ---
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')
    
    bg_status = '#16a34a' if "TERLIHAT" in status_ai else '#dc2626'
    bar_fill = int(prob_ai/5)
    bar_conf = "|" + "#" * bar_fill + "." * (20 - bar_fill) + "|"
    
    info_text = f"""
    :: LAPORAN RUKYATUL HILAL AI ::
    
    [ IDENTITAS CREATOR ]
    Nama        : {CREATOR_NAME}
    Afiliasi    : {CREATOR_AFILIASI}
    
    [ WAKTU & LOKASI ]
    Masehi      : {tgl_target}
    Hijriah     : {str_hijri}
    Sunset      : {astro['Waktu_Sunset_Lokal']}
    
    [ DATA HISAB & FISIK ]
    Altitude    : {astro['aD']:.4f}°
    Elongasi    : {astro['aL']:.4f}°
    Fi Illuminasi : {astro['Illuminasi']:.2f}%
    Lebar (w)   : {astro['w']:.4f} moa
    Lag         : {astro['Lag']:.2f} menit
    
    [ KONDISI CUACA ]
    Langit      : {cuaca['Deskripsi']}
    Awan        : {cuaca['Kondisi_Awan_Pct']}%
    Suhu        : {cuaca['Suhu_Atmosfer_C']}°C
    Kelembapan  : {cuaca['Kelembapan_Pct']}%
    Angin       : {cuaca['Kecepatan_Angin_ms']} m/s
    Transparansi: {cuaca['Transparansi_Index']}
    
    [ ANALISIS KEPUTUSAN AI ]
    Peluang Visibilitas : {prob_ai:.2f} %
    {bar_conf}
    
    [ PREDIKSI AWAL BULAN ]
    {ket_awal_bulan}
    Mulai: {next_masehi}
    """
    ax_text.text(0.02, 0.98, info_text, transform=ax_text.transAxes, fontsize=9, 
                 verticalalignment='top', family='monospace', color='#1e293b')

    rect_status = dict(boxstyle="round,pad=0.7", fc=bg_status, ec="black")
    ax_text.text(0.5, 0.05, status_ai, transform=ax_text.transAxes, fontsize=11, 
                 color='white', fontweight='bold', ha='center', va='center', bbox=rect_status)
    
    print("\n" + "="*50)
    simpan = input("📸 Simpan laporan ke Gambar? (y/n): ").strip().lower()
    
    if simpan == 'y':
        try:
            clean_lokasi = nama_lokasi.replace(' ', '_')
            nama_file = f"Laporan_{tgl_target}_{clean_lokasi}.png"
            fig.savefig(nama_file, dpi=200, bbox_inches='tight')
            print(f"✅ SUKSES: Gambar berhasil disimpan sebagai: {nama_file}")
        except Exception as e:
            print(f"\n⚠️ PERINGATAN: Gagal menyimpan gambar. Error: {e}")

    try: tampilkan_plot_universal(fig)
    except Exception as e: print(f"⚠️ Gagal tampil plot ({e})")
    pause_return_menu()

# ==============================================================================
# MAIN LOOP
# ==============================================================================
if __name__ == "__main__":
    while True:
        try:
            tampilkan_menu_utama()
            p = input(" 👉 Masukkan Nomor Menu (1-4): ").strip()
            
            if p == '1': clear_screen(); train_model()
            elif p == '2': clear_screen(); predict_future()
            elif p == '3': clear_screen(); generate_bulk_hisab()
            elif p == '4': print("\n👋 Wassalam!"); break
            else: print("\n❌ Pilihan tidak valid."); time.sleep(1.5)
                
        except KeyboardInterrupt: 
            print("\n👋 Keluar Paksa."); break
        except Exception as e: 
            print(f"\n❌ Error Fatal: {e}")
            input("Enter untuk restart sistem...")



