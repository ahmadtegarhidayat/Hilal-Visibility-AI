"""
================================================================================
🌙 SISTEM CERDAS RUKYATUL HILAL & PENENTUAN AWAL BULAN (AI-BASED)
================================================================================
👨‍💻 CREATOR  : Ahmad Tegar Hidayat
🎓 STATUS   : Mahasiswa Ilmu Falak UIN Walisongo Semarang
📧 EMAIL    : ahmadtegar0809@gmail.com
🏢 INSTANSI : Observatorium UIN Walisongo
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import requests
import os
import warnings
import sys
import time
from datetime import datetime, timedelta

# --- LIBRARY CHECK ---
def check_library(lib_name, install_cmd):
    try:
        __import__(lib_name.replace("-", "_"))
    except ImportError:
        print(f"❌ Library '{lib_name}' belum terinstall.")
        print(f"   Silakan install: pip install {install_cmd}")
        sys.exit()

check_library("hijri_converter", "hijri-converter")
check_library("skyfield", "skyfield")
check_library("sklearn", "scikit-learn")

from hijri_converter import Gregorian
from skyfield.api import load, wgs84
from skyfield import almanac
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from matplotlib.patches import Circle

warnings.filterwarnings('ignore')

# ==============================================================================
# ⚙️ KONFIGURASI SISTEM
# ==============================================================================
FILENAME_CSV    = 'data_set_rukyat_ml.csv'
FILENAME_MODEL  = 'model_hilal_rf.pkl'
DEF_NAME        = "Observatorium UIN Walisongo (Semarang)"
DEF_LAT         = -6.995347
DEF_LON         = 110.347949
DEF_DATE        = datetime.now().strftime("%Y-%m-%d")

# ==============================================================================
# 🖌️ TAMPILAN & UTILITAS
# ==============================================================================
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    clear_screen()
    print("="*70)
    print("🌙  SISTEM PREDIKSI HILAL & GENERATOR DATA HISAB AI")
    print("="*70)
    print(f"👨‍💻  Created by : Ahmad Tegar Hidayat")
    print(f"🎓  Afiliasi   : Ilmu Falak - UIN Walisongo Semarang")
    print(f"📧  Kontak     : ahmadtegar0809@gmail.com")
    print("-" * 70)
    print("\n")

def tampilkan_plot_universal(figure):
    is_notebook = False
    try:
        from IPython import get_ipython
        if get_ipython() is not None: is_notebook = True
    except ImportError: is_notebook = False

    if is_notebook:
        from IPython.display import display, clear_output
        clear_output(wait=True); display(figure)
    else:
        print("📊 Dashboard Visualisasi Telah Terbuka...")
        plt.show()

def pause_return_menu():
    print("\n" + "-"*50)
    input("👉 Tekan [ENTER] untuk kembali ke Menu Utama...")

# ==============================================================================
# 🔭 ENGINE ASTRONOMI (SKYFIELD)
# ==============================================================================
def hitung_astronomi(lat, lon, tgl_str, verbose=True):
    if verbose: print("🔭 Menghitung data hisab ephemeris...", end="\r")
    
    # Auto-download Ephemeris
    try: eph = load('de421.bsp')
    except: 
        if verbose: print("\n📥 Mendownload file ephemeris (de421.bsp)...")
        eph = load('de421.bsp')
        
    ts = load.timescale()
    topos = wgs84.latlon(lat, lon)
    sun, moon, earth = eph['sun'], eph['moon'], eph['earth']
    
    try:
        tgl_dt = datetime.strptime(tgl_str, "%Y-%m-%d")
        
        # 1. Cari Sunset (Range UTC 10:00 - 15:00 = Sore WIB)
        t_start_sun = ts.utc(tgl_dt.year, tgl_dt.month, tgl_dt.day, 10) 
        t_end_sun   = ts.utc(tgl_dt.year, tgl_dt.month, tgl_dt.day, 15)
        
        f_geo_sun = almanac.risings_and_settings(eph, sun, topos, horizon_degrees=-0.8333)
        times_s, events_s = almanac.find_discrete(t_start_sun, t_end_sun, f_geo_sun)
        
        if len(times_s) == 0:
            if verbose: print("\n⚠️ Error: Tidak ditemukan waktu sunset.")
            return None
        t_sunset = times_s[0]

        # 2. Hitung Posisi Bulan & Matahari saat Sunset
        observer = (earth + topos).at(t_sunset)
        m = observer.observe(moon).apparent()
        s = observer.observe(sun).apparent()
        
        alt_m, az_m, _ = m.altaz()
        alt_s, az_s, _ = s.altaz()

        # 3. Cari Moonset (Logika Smart-Lag)
        f_geo_moon = almanac.risings_and_settings(eph, moon, topos, horizon_degrees=-0.8333)
        
        if alt_m.degrees > 0:
            t_search_end = ts.utc(tgl_dt.year, tgl_dt.month, tgl_dt.day, t_sunset.utc_datetime().hour + 6)
            times_m, _ = almanac.find_discrete(t_sunset, t_search_end, f_geo_moon)
            t_moonset = times_m[0] if len(times_m) > 0 else None
        else:
            t_search_start = ts.utc(tgl_dt.year, tgl_dt.month, tgl_dt.day, t_sunset.utc_datetime().hour - 6)
            times_m, _ = almanac.find_discrete(t_search_start, t_sunset, f_geo_moon)
            t_moonset = times_m[-1] if len(times_m) > 0 else None

        # 4. Hitung Lag
        if t_moonset is not None:
            lag_minutes = (t_moonset.utc_datetime() - t_sunset.utc_datetime()).total_seconds() / 60
        else:
            lag_minutes = alt_m.degrees * 4 

        # 5. Parameter Lain
        illumination_fraction = m.fraction_illuminated(sun)
        w_moa = 31.0 * illumination_fraction 
        local_sunset = t_sunset.astimezone(datetime.now().astimezone().tzinfo)

        return {
            'aD': alt_m.degrees,
            'aL': m.separation_from(s).degrees,
            'DAz': az_m.degrees - az_s.degrees,
            'Lag': lag_minutes,
            'Illuminasi': illumination_fraction * 100,
            'w': w_moa,
            'Moon_Alt': alt_m.degrees, 'Moon_Az': az_m.degrees, 'Sun_Az': az_s.degrees,
            'Waktu_Sunset_Lokal': local_sunset.strftime('%H:%M:%S'),
            'Jam_Sunset_Lokal_Int': local_sunset.hour
        }
    except Exception as e: 
        if verbose: print(f"\n❌ Terjadi kesalahan hisab: {e}")
        return None

# ==============================================================================
# ☁️ ENGINE CUACA
# ==============================================================================
def get_cuaca(use_api, lat, lon, tgl, jam_sunset_lokal=18):
    data_def = {'Suhu_Atmosfer_C': 28.0, 'Kelembapan_Pct': 75.0, 'Kondisi_Awan_Pct': 10.0, 'Deskripsi': 'Simulasi Ideal', 'Sumber': 'Simulasi'}
    
    if use_api:
        print("☁️ Mengambil data cuaca real-time (API)...")
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat, "longitude": lon, 
                "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,weathercode", 
                "start_date": tgl, "end_date": tgl, "timezone": "auto"
            }
            r = requests.get(url, params=params, timeout=5).json()
            idx = min(jam_sunset_lokal, 23)
            return {
                'Suhu_Atmosfer_C': r['hourly']['temperature_2m'][idx],
                'Kelembapan_Pct': r['hourly']['relative_humidity_2m'][idx],
                'Kondisi_Awan_Pct': r['hourly']['cloud_cover'][idx],
                'Deskripsi': f"WMO Code {r['hourly']['weathercode'][idx]}",
                'Sumber': 'API Open-Meteo'
            }
        except: print("⚠️ Gagal terhubung API. Menggunakan mode Simulasi Ideal.")
    return data_def

# ==============================================================================
# 🧮 MENU 3: GENERATOR DATA HISAB MASSAL (BARU)
# ==============================================================================
def generate_bulk_hisab():
    print("\n[ MODUL 3 ] GENERATOR DATA HISAB MASSAL (BIG DATA)")
    print("-" * 50)
    print("ℹ️  Modul ini digunakan untuk menghitung parameter astronomi")
    print("    secara otomatis untuk banyak tanggal sekaligus.")
    print("    Format CSV Input wajib memiliki kolom: 'Tanggal' (YYYY-MM-DD)")
    print("    Opsional: 'Latitude', 'Longitude' (Jika kosong pakai default)")
    print("-" * 50)
    
    input_file = input("📂 Masukkan nama file CSV input (misal: input.csv): ")
    if not os.path.exists(input_file):
        print("❌ File tidak ditemukan."); pause_return_menu(); return

    try:
        df = pd.read_csv(input_file, sep=';' if ';' in open(input_file).readline() else ',')
        print(f"✅ Berhasil memuat {len(df)} baris data.")
    except Exception as e:
        print(f"❌ Gagal baca CSV: {e}"); pause_return_menu(); return

    if 'Tanggal' not in df.columns:
        print("❌ Kolom 'Tanggal' tidak ditemukan di CSV."); pause_return_menu(); return

    print("\n🚀 Memulai Perhitungan Hisab Massal...")
    results = []
    
    start_time = time.time()
    
    for i, row in df.iterrows():
        tgl = row['Tanggal']
        
        # Cek apakah ada kolom Latitude/Longitude di CSV, jika tidak pakai Default
        lat = float(row['Latitude']) if 'Latitude' in row and pd.notnull(row['Latitude']) else DEF_LAT
        lon = float(row['Longitude']) if 'Longitude' in row and pd.notnull(row['Longitude']) else DEF_LON
        
        astro = hitung_astronomi(lat, lon, tgl, verbose=False) # Verbose False biar ga spam console
        
        if astro:
            # Format Output sesuai Dataset Training ML
            data_row = {
                'Tanggal': tgl,
                'Latitude': lat,
                'Longitude': lon,
                'aD': round(astro['aD'], 4),
                'aL': round(astro['aL'], 4),
                'DAz': round(astro['DAz'], 4),
                'Lag': round(astro['Lag'], 2),
                'w': round(astro['w'], 4),
                'Illuminasi': round(astro['Illuminasi'], 2),
                'Moon_Alt': round(astro['Moon_Alt'], 4),
                'Waktu_Sunset': astro['Waktu_Sunset_Lokal']
            }
            results.append(data_row)
        
        # Progress Bar Sederhana
        if (i+1) % 10 == 0 or (i+1) == len(df):
            print(f"   ⏳ Proses: {i+1}/{len(df)} data selesai...", end="\r")

    # Simpan Hasil
    output_df = pd.DataFrame(results)
    output_filename = f"hasil_hisab_{int(time.time())}.csv"
    output_df.to_csv(output_filename, index=False)
    
    duration = time.time() - start_time
    print(f"\n\n✅ Selesai! Data berhasil disimpan ke: {output_filename}")
    print(f"⏱️  Waktu proses: {duration:.2f} detik")
    
    pause_return_menu()

# ==============================================================================
# 🧠 MENU 1: TRAINING AI
# ==============================================================================
def train_model():
    print("\n[ MODUL 1 ] TRAINING & EVALUASI MODEL")
    print("-" * 50)
    
    if not os.path.exists(FILENAME_CSV):
        print(f"❌ ERROR KRITIS: File '{FILENAME_CSV}' tidak ditemukan!")
        pause_return_menu(); return

    try: 
        df = pd.read_csv(FILENAME_CSV, sep=';', encoding='latin1', thousands='.', decimal=',')
    except Exception as e: 
        print(f"❌ Gagal membaca CSV: {e}"); pause_return_menu(); return

    # Pembersihan Data
    df.columns = df.columns.str.replace('ï»¿', '').str.strip()
    if 'aD' in df.columns and df['aD'].max() > 100: 
        for c in ['aD', 'aL', 'DAz', 'Lag']: 
            if c in df.columns: df[c] = df[c] / 1000

    if 'Visibility' not in df.columns:
        print("❌ Kolom 'Visibility' tidak ditemukan."); pause_return_menu(); return
        
    df['Target'] = df['Visibility'].apply(lambda x: 1 if x > 0 else 0)
    features = ['aD', 'aL', 'DAz', 'Lag', 'Suhu_Atmosfer_C', 'Kelembapan_Pct', 'Kondisi_Awan_Pct']
    
    for f in features:
        if f not in df.columns: df[f] = 0

    pipeline = Pipeline([('model', RandomForestClassifier(n_estimators=200, random_state=42))])
    
    X = df[features].fillna(0)
    y = df['Target']
    pipeline.fit(X, y)
    
    accuracy = pipeline.score(X, y) * 100
    
    save_data = {
        'pipeline': pipeline,
        'accuracy': accuracy,
        'train_date': datetime.now().strftime("%Y-%m-%d")
    }

    joblib.dump(save_data, FILENAME_MODEL)
    print(f"✅ Model berhasil dilatih!")
    print(f"📊 Akurasi pada data latih: {accuracy:.2f}%")
    print(f"💾 Disimpan ke: {FILENAME_MODEL}")
    
    pause_return_menu()

# ==============================================================================
# 🚀 MENU 2: PREDIKSI & DASHBOARD
# ==============================================================================
def predict_future():
    print("\n[ MODUL 2 ] PREDIKSI & VISUALISASI")
    print("-" * 50)
    
    if not os.path.exists(FILENAME_MODEL): 
        print("❌ Model AI belum dilatih. Jalankan Menu 1 terlebih dahulu."); pause_return_menu(); return
        
    loaded_data = joblib.load(FILENAME_MODEL)
    if isinstance(loaded_data, dict):
        model = loaded_data['pipeline']
        model_acc = loaded_data['accuracy']
    else:
        model = loaded_data; model_acc = 0.0
    
    # Input User
    ganti = input(f"Gunakan Lokasi Default ({DEF_NAME})? (y/n): ").lower()
    if ganti == 'n':
        nama_lokasi = input("Nama Lokasi : ")
        try: lat = float(input("Latitude    : ")); lon = float(input("Longitude   : ")) 
        except: return
    else: lat, lon, nama_lokasi = DEF_LAT, DEF_LON, DEF_NAME

    tgl_target = input(f"Tanggal (YYYY-MM-DD) [Default {DEF_DATE}]: ") or DEF_DATE
    
    # Hijriah
    y_g, m_g, d_g = map(int, tgl_target.split('-'))
    h_today = Gregorian(y_g, m_g, d_g).to_hijri()
    str_hijri = f"{h_today.day} {h_today.month_name()} {h_today.year} H"

    # Proses
    pilih_api = True if input("Mode Cuaca [1] Real-time API  [2] Simulasi Ideal : ") == '1' else False
    astro = hitung_astronomi(lat, lon, tgl_target)
    if not astro: pause_return_menu(); return
    cuaca = get_cuaca(pilih_api, lat, lon, tgl_target, astro['Jam_Sunset_Lokal_Int'])
    
    # Logika Keputusan
    reject_reason = None
    if astro['aD'] < 3.0: reject_reason = "Altitude < 3° (MABIMS)"
    elif astro['aL'] < 6.4: reject_reason = "Elongasi < 6.4° (MABIMS)"
    elif astro['Lag'] < 0: reject_reason = "Moonset sebelum Sunset"
    elif cuaca['Kondisi_Awan_Pct'] > 90: reject_reason = "Awan Tebal (>90%)"
    
    input_data = pd.DataFrame([{
        'aD': astro['aD'], 'aL': astro['aL'], 'DAz': astro['DAz'], 'Lag': astro['Lag'],
        'Suhu_Atmosfer_C': cuaca['Suhu_Atmosfer_C'], 'Kelembapan_Pct': cuaca['Kelembapan_Pct'], 'Kondisi_Awan_Pct': cuaca['Kondisi_Awan_Pct']
    }])
    if hasattr(model, 'feature_names_in_'):
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
        
    pred = model.predict(input_data)[0]
    prob_ai = model.predict_proba(input_data)[:, 1][0] * 100
    
    if reject_reason:
        status_ai = "TIDAK TERLIHAT"
    else:
        status_ai = "TERLIHAT" if pred == 1 else "TIDAK TERLIHAT"

    if status_ai == "TERLIHAT":
        next_masehi = (datetime.strptime(tgl_target, "%Y-%m-%d") + timedelta(days=1)).strftime("%d %B %Y")
        ket_awal_bulan = f"1 {Gregorian(y_g, m_g, d_g + 1).to_hijri().month_name()}"
    else:
        next_masehi = (datetime.strptime(tgl_target, "%Y-%m-%d") + timedelta(days=2)).strftime("%d %B %Y")
        ket_awal_bulan = f"1 {Gregorian(y_g, m_g, d_g + 2).to_hijri().month_name()} (Istikmal)"

    # --- DASHBOARD VISUALISASI ---
    fig = plt.figure(figsize=(13, 7), facecolor='#f4f4f4')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.2]) 
    
    # GRAFIK
    ax_plot = fig.add_subplot(gs[0])
    ax_plot.set_facecolor('#0b0b1a') 
    
    ax_plot.axhline(0, color='#00ff00', lw=2, alpha=0.8) 
    ax_plot.axhspan(-5, 0, color='#2b1d14', alpha=1.0)  
    ax_plot.axhline(3.0, color='red', linestyle='--', linewidth=1, label='Limit Alt 3°')
    limit_circle = Circle((astro['Sun_Az'], -0.83), 6.4, color='yellow', fill=False, linestyle='--', label='Limit Elongasi 6.4°')
    ax_plot.add_patch(limit_circle)
    
    ax_plot.scatter(astro['Sun_Az'], -0.83, s=600, c='orange', edgecolors='red', label='Matahari', zorder=5)
    col_moon = 'gold' if status_ai == "TERLIHAT" else '#555555'
    ax_plot.scatter(astro['Moon_Az'], astro['Moon_Alt'], s=350, c=col_moon, edgecolors='white', label='Bulan', zorder=10)
    ax_plot.text(astro['Moon_Az'], astro['Moon_Alt'] + 0.7, f"Alt: {astro['aD']:.2f}°", color='cyan', ha='center', fontsize=9, fontweight='bold')

    ax_plot.set_title(f"VISUALISASI POSISI HILAL\n{nama_lokasi}", color='black', fontsize=12, fontweight='bold', pad=15)
    ax_plot.set_xlabel("Azimuth (Arah)"); ax_plot.set_ylabel("Altitude (Tinggi)")
    ax_plot.legend(loc='upper right', fontsize=8, facecolor='white', framealpha=0.9)
    ax_plot.grid(True, linestyle=':', alpha=0.3, color='white')
    ax_plot.set_aspect('equal', adjustable='box') 
    
    mid_az = (astro['Sun_Az'] + astro['Moon_Az']) / 2
    ax_plot.set_xlim(mid_az - 8, mid_az + 8)
    ax_plot.set_ylim(-3, max(astro['Moon_Alt'] + 5, 8))

    # TEKS INFO
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')
    
    bg_status = '#28a745' if status_ai == "TERLIHAT" else '#dc3545'
    bar_fill = int(prob_ai/5)
    bar_conf = "|" + "█" * bar_fill + "░" * (20 - bar_fill) + "|"
    
    info_text = f"""
    :: LAPORAN RUKYATUL HILAL AI ::
    
    [ WAKTU & LOKASI ]
    Masehi      : {tgl_target}
    Hijriah     : {str_hijri}
    Sunset      : {astro['Waktu_Sunset_Lokal']}
    
    [ DATA HISAB ]
    Altitude    : {astro['aD']:.4f}°
    Elongasi    : {astro['aL']:.4f}°
    Lebar (w)   : {astro['w']:.4f} moa
    Lag         : {astro['Lag']:.2f} menit
    Illuminasi  : {astro['Illuminasi']:.2f}%
    
    [ KONDISI CUACA ]
    Langit      : {cuaca['Deskripsi']}
    Awan        : {cuaca['Kondisi_Awan_Pct']}%
    Suhu        : {cuaca['Suhu_Atmosfer_C']}°C

    [ KEPUTUSAN AI ]
    Akurasi Model: {model_acc:.2f}%
    Confidence   : {prob_ai:.2f}%
    {bar_conf}
    
    [ PREDIKSI AWAL BULAN ]
    {ket_awal_bulan}
    Mulai: {next_masehi}
    """
    
    ax_text.text(0.05, 0.98, info_text, transform=ax_text.transAxes, fontsize=10, 
                 verticalalignment='top', family='monospace', color='#333333')

    rect_status = dict(boxstyle="round,pad=0.7", fc=bg_status, ec="black")
    ax_text.text(0.5, 0.12, status_ai, transform=ax_text.transAxes, fontsize=14, 
                 color='white', fontweight='bold', ha='center', va='center', bbox=rect_status)
    
    if reject_reason:
        ax_text.text(0.5, 0.05, f"(! {reject_reason} !)", transform=ax_text.transAxes, 
                     fontsize=9, color='red', ha='center', fontweight='bold')

    plt.tight_layout()
    tampilkan_plot_universal(fig)
    
    # NAVIGASI FIX
    pause_return_menu()

# ==============================================================================
# MAIN LOOP PROGRAM
# ==============================================================================
if __name__ == "__main__":
    while True:
        print_header()
        print("=== MENU UTAMA ===")
        print("1. Training Model AI (Gunakan Data Set Asli)")
        print("2. Prediksi & Dashboard Hilal (Single Date)")
        print("3. Generator Data Hisab Massal (Dari CSV)")
        print("4. Keluar")
        print("-" * 30)
        
        try:
            p = input("👉 Pilih Menu (1-4): ")
            if p == '1': train_model()
            elif p == '2': predict_future()
            elif p == '3': generate_bulk_hisab()
            elif p == '4': 
                print("\n👋 Terima kasih. Semoga bermanfaat!"); break
            else:
                input("⚠️ Pilihan tidak valid. Tekan Enter...")
        except KeyboardInterrupt:
            print("\nKeluar paksa..."); break
