import os
import numpy as np
import cv2
import json
import base64
from flask import Flask, request, jsonify, render_template, send_file, flash, redirect, url_for
from flask_cors import CORS
from PIL import Image
import io
from services.histogram import HistogramShiftingWatermark, embed_watermark, extract_watermark, recover_image
from services.rdwt import (
    embed_watermark_rdwt,
    extract_watermark_rdwt,
    recover_image_rdwt,
    calculate_psnr,
    validate_inputs,
    get_text_limits,
    save_overhead_data,
    load_overhead_data
)
from werkzeug.utils import secure_filename
import zipfile
import tempfile
from datetime import datetime
from services.attacks import Attack
from utils.helpers import *


app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for API endpoints

from config.settings import Config
app.config.from_object(Config)
Config.init_app(app)
UPLOAD_FOLDER = Config.UPLOAD_FOLDER
RESULTS_FOLDER = Config.RESULTS_FOLDER
ALLOWED_EXTENSIONS = Config.ALLOWED_EXTENSIONS

# Global session storage
session_store = {}


# Tambahkan fungsi-fungsi ini ke dalam app.py setelah import statements dan sebelum definisi routes













# ===============================
# MAIN WEB ROUTES - DUAL METHOD SUPPORT
# ===============================

@app.route('/')
def index():
    """Serve the main page with method selection."""
    methods_info = {
        'histogram_shifting': {
            'name': 'Histogram Shifting',
            'description': 'Metode tradisional menggunakan modifikasi histogram gambar',
            'advantages': ['Sederhana dan cepat', 'Hasil visual baik', 'Cocok untuk gambar natural'],
            'disadvantages': ['Kurang tahan terhadap serangan', 'Kapasitas terbatas']
        },
        'rdwt': {
            'name': 'True RDWT (Redundant DWT)',
            'description': 'Metode advanced menggunakan Stationary Wavelet Transform',
            'advantages': ['Sangat tahan serangan', 'Parameter otomatis', 'Kapasitas besar', 'Mendukung recovery'],
            'disadvantages': ['Lebih kompleks', 'Membutuhkan overhead data']
        }
    }
    return render_template('index.html', methods=methods_info)

@app.route('/embed/<method>', methods=['GET', 'POST'])
def embed_watermark_method(method):
    """Embed watermark dengan metode yang dipilih."""
    if method not in ['histogram_shifting', 'rdwt']:
        flash('Metode tidak valid')
        return redirect(url_for('index'))
    
    if request.method == 'GET':
        return render_template('embed_method.html', method=method)
    
    try:
        # Ambil data form
        watermark_text = request.form.get('watermark_text', '').strip()
        
        if 'image' not in request.files:
            flash('Tidak ada file gambar yang dipilih')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('Tidak ada file gambar yang dipilih')
            return redirect(request.url)
        
        if not allowed_file(file.filename):
            flash('Tipe file tidak valid. Gunakan PNG, JPG, JPEG, BMP, atau TIFF.')
            return redirect(request.url)
        
        # Baca gambar
        file.seek(0)
        img_data = file.read()
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            flash('Tidak dapat membaca file gambar')
            return redirect(request.url)
        
        # Prepare image for watermarking (Blue Channel preservation for perfect lossless)
        if len(img.shape) == 3 and not (np.array_equal(img[:,:,0], img[:,:,1]) and np.array_equal(img[:,:,1], img[:,:,2])):
            process_img = img[:, :, 0].copy() # Extract Blue channel
            is_color = True
            ycc_img = img # Pass original img to replace channel later
        else:
            process_img = rgb_to_grayscale(img)
            ycc_img = None
            is_color = False
        
        # Convert teks ke bits
        watermark_bits = text_to_bits(watermark_text)
        
        if method == 'histogram_shifting':
            return embed_histogram_shifting(img, process_img, is_color, ycc_img, watermark_text, watermark_bits)
        else:  # rdwt
            return embed_rdwt_method(img, process_img, is_color, ycc_img, watermark_text, watermark_bits)
            
    except Exception as e:
        error_msg = str(e)
        print(f"Embedding error: {error_msg}")
        flash(f'Error saat embedding: {error_msg}')
        return redirect(url_for('embed_watermark_method', method=method))

def embed_histogram_shifting(img, process_img, is_color, ycc_img, watermark_text, watermark_bits):
    """Embed menggunakan Histogram Shifting method."""
    try:
        # Ambil parameter dari form
        block_size = int(request.form.get('block_size', 64))
        strength = int(request.form.get('strength', 3))
        redundancy = int(request.form.get('redundancy', 3))
        
        # Validasi parameter
        if block_size < 16 or block_size > 128:
            flash('Block size harus antara 16-128')
            return redirect(request.url)
        
        if strength < 1 or strength > 10:
            flash('Strength harus antara 1-10')
            return redirect(request.url)
        
        if redundancy < 1 or redundancy > 5:
            flash('Redundancy harus antara 1-5')
            return redirect(request.url)
        
        # Cek kapasitas
        rows, cols = process_img.shape
        blocks_x = cols // block_size
        blocks_y = rows // block_size
        total_blocks = blocks_x * blocks_y
        required_blocks = len(watermark_bits) * redundancy
        
        if required_blocks > total_blocks:
            max_chars = total_blocks // (redundancy * 20)  # Estimasi konservatif
            flash(f'Teks watermark terlalu panjang untuk gambar ini. '
                  f'Diperlukan: {required_blocks} blok, Tersedia: {total_blocks} blok. '
                  f'Coba kurangi teks menjadi maksimal {max_chars} karakter atau '
                  f'perkecil redundancy/block_size.')
            return redirect(request.url)
        
        # Embed watermark
        watermarked_process, overhead_data = embed_watermark(
            process_img, watermark_bits, strength, block_size, redundancy
        )
        
        # Convert kembali ke RGB
        if is_color:
            watermarked_img = ycc_img.copy()
            watermarked_img[:, :, 0] = watermarked_process
        else:
            watermarked_img = grayscale_to_rgb(watermarked_process)
            
        overhead_data['is_color'] = is_color
        
        # Hitung PSNR
        psnr = calculate_psnr(img, watermarked_img)
        
        # Simpan hasil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        watermarked_filename = f"watermarked_hs_{timestamp}.png"
        overhead_filename = f"overhead_hs_{timestamp}.json"
        
        cv2.imwrite(os.path.join(RESULTS_FOLDER, watermarked_filename), watermarked_img)
        
        # Tambahkan info validasi ke overhead data
        overhead_data['method'] = 'histogram_shifting'
        overhead_data['original_text'] = watermark_text
        overhead_data['text_length_chars'] = len(watermark_text)
        overhead_data['bits_used'] = len(watermark_bits)
        overhead_data['encoding'] = 'utf-8'
        overhead_data['timestamp'] = timestamp
        
        save_overhead_data(overhead_data, os.path.join(RESULTS_FOLDER, overhead_filename))

        
        # Siapkan data response
        result = {
            'success': True,
            'method': 'histogram_shifting',
            'method_name': 'Histogram Shifting',
            'original_image': image_to_base64(img),
            'watermarked_image': image_to_base64(watermarked_img),
            'psnr': round(psnr, 2),
            'watermark_text': watermark_text,
            'watermark_length': len(watermark_bits),
            'capacity': total_blocks,
            'capacity_used_blocks': required_blocks,
            'capacity_used_percent': round((required_blocks / total_blocks) * 100, 1),
            'watermarked_filename': watermarked_filename,
            'overhead_filename': overhead_filename,
            'method_info': {
                'description': 'Histogram Shifting dengan Block-based Embedding',
                'parameters': {
                    'block_size': block_size,
                    'strength': strength,
                    'redundancy': redundancy,
                    'blocks_used': required_blocks,
                    'total_blocks': total_blocks
                }
            }
        }
        
        return render_template('embed_result.html', result=result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Histogram Shifting embedding error: {error_msg}")
        flash(f'Error Histogram Shifting: {error_msg}')
        return redirect(request.url)


def embed_rdwt_method(img, process_img, is_color, ycc_img, watermark_text, watermark_bits):
    """Embed menggunakan True RDWT method."""
    try:
        # Validasi input - gunakan fungsi lokal jika tidak ada di rdwt_watermarking
        try:
            errors, warnings = validate_inputs(process_img, watermark_text)
        except (NameError, ImportError):
            # Fallback validation jika fungsi tidak tersedia
            errors = []
            warnings = []
            
            if len(watermark_text) == 0:
                errors.append("Teks watermark tidak boleh kosong")
            if len(watermark_text) > 500:
                warnings.append("Teks watermark sangat panjang, mungkin tidak optimal")
            if process_img.shape[0] < 128 or process_img.shape[1] < 128:
                warnings.append("Ukuran gambar kecil, kapasitas terbatas")
        
        if errors:
            for error in errors:
                flash(error)
            return redirect(request.url)
        
        # Tampilkan peringatan jika ada
        for warning in warnings:
            flash(warning, 'warning')
        
        # Cek kapasitas
        capacity = calculate_capacity(process_img.shape)
        required_bits = len(watermark_bits)
        
        if required_bits > capacity:
            try:
                text_limits = get_text_limits(process_img.shape)
                max_chars = text_limits["recommended_max"]
            except (NameError, ImportError):
                max_chars = capacity // 16  # Fallback estimation
                
            flash(f'Teks watermark terlalu panjang untuk gambar ini. '
                  f'Diperlukan: {required_bits} bits, Tersedia: {capacity} bits. '
                  f'Coba kurangi teks menjadi maksimal {max_chars} karakter.')
            return redirect(request.url)
        
        # Embed watermark dengan parameter otomatis - FIXED SECTION
        try:
            # Coba panggil embed function dan handle different return formats
            result = embed_watermark_rdwt(process_img, watermark_bits)
            
            # Debug print to see what we get
            print(f"embed_watermark_rdwt returned: {type(result)}")
            
            # Handle different return formats
            if isinstance(result, tuple) and len(result) == 2:
                watermarked_process, overhead_data = result
            elif isinstance(result, tuple) and len(result) == 1:
                watermarked_process = result[0]
                overhead_data = create_default_overhead_data()
            else:
                watermarked_process = result
                overhead_data = create_default_overhead_data()
                
        except Exception as embed_error:
            print(f"Error in embed_watermark_rdwt: {embed_error}")
            flash(f'Error dalam proses embedding RDWT: {str(embed_error)}')
            return redirect(request.url)
        
        # Convert kembali ke RGB
        if is_color:
            watermarked_img = ycc_img.copy()
            watermarked_img[:, :, 0] = watermarked_process
        else:
            watermarked_img = grayscale_to_rgb(watermarked_process)
            
        overhead_data['is_color'] = is_color
        
        # Hitung PSNR
        psnr = calculate_psnr(img, watermarked_img)
        
        # Simpan hasil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        watermarked_filename = f"watermarked_rdwt_{timestamp}.png"
        overhead_filename = f"overhead_rdwt_{timestamp}.json"
        
        cv2.imwrite(os.path.join(RESULTS_FOLDER, watermarked_filename), watermarked_img)
        
        # Tambahkan info validasi ke overhead data
        overhead_data['method'] = 'rdwt'
        overhead_data['original_text'] = watermark_text
        overhead_data['text_length_chars'] = len(watermark_text)
        overhead_data['bits_used'] = len(watermark_bits)
        overhead_data['encoding'] = 'utf-8'
        overhead_data['timestamp'] = timestamp
        
        save_overhead_data(overhead_data, os.path.join(RESULTS_FOLDER, overhead_filename))
        
        # Siapkan data response
        result = {
            'success': True,
            'method': 'rdwt',
            'method_name': 'True RDWT (Redundant DWT)',
            'original_image': image_to_base64(img),
            'watermarked_image': image_to_base64(watermarked_img),
            'psnr': round(psnr, 2),
            'watermark_text': watermark_text,
            'watermark_length': len(watermark_bits),
            'capacity': capacity,
            'capacity_used_percent': round((len(watermark_bits) / capacity) * 100, 1),
            'watermarked_filename': watermarked_filename,
            'overhead_filename': overhead_filename,
            'method_info': {
                'description': 'True RDWT (Stationary WT) dengan Parameter Otomatis',
                'parameters': {
                    'wavelet': overhead_data.get('wavelet', 'N/A'),
                    'levels': overhead_data.get('levels', 'N/A'),
                    'subband': overhead_data.get('subband', 'N/A'),
                    'strength': overhead_data.get('strength', 'N/A'),
                    'auto_parameters': True
                },
                'analysis': overhead_data.get('analysis', {})
            }
        }
        
        return render_template('embed_result.html', result=result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"RDWT embedding error: {error_msg}")
        flash(f'Error RDWT: {error_msg}')
        return redirect(request.url)

@app.route('/extract/<method>', methods=['GET', 'POST'])
def extract_watermark_method(method):
    """Extract watermark dengan metode yang sesuai."""
    if method not in ['histogram_shifting', 'rdwt']:
        flash('Metode tidak valid')
        return redirect(url_for('index'))
    
    if request.method == 'GET':
        return render_template('extract_method.html', method=method)
    
    try:
        # Validasi input files
        if 'watermarked_image' not in request.files or 'overhead_data' not in request.files:
            flash('Harap upload gambar watermarked dan file overhead data')
            return redirect(request.url)
        
        img_file = request.files['watermarked_image']
        overhead_file = request.files['overhead_data']
        
        if img_file.filename == '' or overhead_file.filename == '':
            flash('Harap pilih file gambar dan overhead data')
            return redirect(request.url)
        
        if not allowed_file(img_file.filename):
            flash('Format gambar tidak valid')
            return redirect(request.url)
        
        if not overhead_file.filename.endswith('.json'):
            flash('File overhead data harus berformat JSON')
            return redirect(request.url)
        
        # Baca gambar watermarked
        img_file.seek(0)
        img_data = img_file.read()
        img_array = np.frombuffer(img_data, np.uint8)
        watermarked_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if watermarked_img is None:
            flash('Tidak dapat membaca gambar watermarked')
            return redirect(request.url)
        
        # Convert ke grayscale
        watermarked_gray = rgb_to_grayscale(watermarked_img)
        
        # Baca overhead data
        overhead_file.seek(0)
        overhead_json = overhead_file.read().decode('utf-8')
        overhead_data = json.loads(overhead_json)
        
        # Validasi metode
        stored_method = overhead_data.get('method', method)
        if stored_method != method:
            # Auto-switch method if mismatch
            method = stored_method
            flash(f'Metode otomatis disesuaikan ke {method.replace("_", " ").title()} berdasarkan file overhead', 'info')
        
        # Calculate watermarked_process correctly
        if overhead_data.get('is_color', False) and len(watermarked_img.shape) == 3:
            watermarked_process = watermarked_img[:, :, 0]
        else:
            watermarked_process = rgb_to_grayscale(watermarked_img)
            
        if method == 'histogram_shifting':
            return extract_histogram_shifting(watermarked_img, watermarked_process, overhead_data)
        else:  # rdwt
            return extract_rdwt_method(watermarked_img, watermarked_process, overhead_data)
            
    except json.JSONDecodeError:
        flash('Format file overhead data tidak valid')
        return redirect(request.url)
    except Exception as e:
        error_msg = str(e)
        print(f"Extraction error: {error_msg}")
        flash(f'Error saat ekstraksi: {error_msg}')
        return redirect(request.url)

def extract_histogram_shifting(watermarked_img, watermarked_process, overhead_data):
    """Extract menggunakan Histogram Shifting method."""
    try:
        # Validasi format overhead data
        required_keys = ['block_size', 'redundancy', 'strength', 'wm_length', 'blocks_data']
        missing_keys = [key for key in required_keys if key not in overhead_data]
        
        if missing_keys:
            flash(f'Overhead data tidak valid untuk Histogram Shifting. Keys yang hilang: {missing_keys}')
            return redirect(request.url)
        
        # Extract watermark
        extracted_bits = extract_watermark(watermarked_process, overhead_data)
        
        # Convert bits ke teks
        extracted_text = bits_to_text(extracted_bits)
        
        # Hitung akurasi jika teks asli tersedia
        accuracy = calculate_text_accuracy(overhead_data.get('original_text', ''), extracted_text)
        
        # Siapkan hasil
        result = {
            'success': True,
            'method': 'histogram_shifting',
            'method_name': 'Histogram Shifting',
            'watermarked_image': image_to_base64(watermarked_img),
            'extracted_text': extracted_text,
            'original_text': overhead_data.get('original_text', ''),
            'extraction_accuracy': round(accuracy, 2) if accuracy is not None else None,
            'extracted_bits_count': len(extracted_bits),
            'expected_bits_count': overhead_data.get('bits_used', len(extracted_bits)),
            'method_info': {
                'description': 'Histogram Shifting Extraction',
                'parameters': {
                    'block_size': overhead_data.get('block_size', 'N/A'),
                    'strength': overhead_data.get('strength', 'N/A'),
                    'redundancy': overhead_data.get('redundancy', 'N/A'),
                    'blocks_used': len(overhead_data.get('blocks_data', []))
                }
            }
        }
        
        return render_template('extract_result.html', result=result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Histogram Shifting extraction error: {error_msg}")
        flash(f'Error Histogram Shifting extraction: {error_msg}')
        return redirect(request.url)

def extract_rdwt_method(watermarked_img, watermarked_process, overhead_data):
    """Extract menggunakan True RDWT method."""
    try:
        # Validasi format overhead data
        required_keys = ['wavelet', 'levels', 'subband', 'strength', 'watermark_length', 'positions']
        missing_keys = [key for key in required_keys if key not in overhead_data]
        
        if missing_keys:
            flash(f'Overhead data tidak valid untuk RDWT. Keys yang hilang: {missing_keys}')
            return redirect(request.url)
        
        # Extract watermark
        extracted_bits = extract_watermark_rdwt(watermarked_process, overhead_data)
        
        # Convert bits ke teks
        extracted_text = bits_to_text(extracted_bits)
        
        # Hitung akurasi jika teks asli tersedia
        accuracy = calculate_text_accuracy(overhead_data.get('original_text', ''), extracted_text)
        
        # Siapkan hasil
        result = {
            'success': True,
            'method': 'rdwt',
            'method_name': 'True RDWT (Redundant DWT)',
            'watermarked_image': image_to_base64(watermarked_img),
            'extracted_text': extracted_text,
            'original_text': overhead_data.get('original_text', ''),
            'extraction_accuracy': round(accuracy, 2) if accuracy is not None else None,
            'extracted_bits_count': len(extracted_bits),
            'expected_bits_count': overhead_data.get('bits_used', len(extracted_bits)),
            'method_info': {
                'description': 'True RDWT (Stationary WT) Extraction',
                'parameters': {
                    'wavelet': overhead_data.get('wavelet', 'N/A'),
                    'levels': overhead_data.get('levels', 'N/A'),
                    'subband': overhead_data.get('subband', 'N/A'),
                    'strength': overhead_data.get('strength', 'N/A')
                },
                'analysis': overhead_data.get('analysis', {})
            }
        }
        
        return render_template('extract_result.html', result=result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"RDWT extraction error: {error_msg}")
        flash(f'Error RDWT extraction: {error_msg}')
        return redirect(request.url)

def calculate_text_accuracy(original_text, extracted_text):
    """Hitung akurasi ekstraksi teks."""
    if not original_text:
        return None
        
    if extracted_text == original_text:
        return 100.0
    else:
        # Hitung Levenshtein distance
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(original_text, extracted_text)
        max_len = max(len(original_text), len(extracted_text))
        if max_len > 0:
            return ((max_len - distance) / max_len) * 100
        else:
            return 100.0

@app.route('/recover/<method>', methods=['GET', 'POST'])
def recover_image_method(method):
    """Recover gambar asli dengan metode yang sesuai."""
    if method not in ['histogram_shifting', 'rdwt']:
        flash('Metode tidak valid')
        return redirect(url_for('index'))
    
    if request.method == 'GET':
        return render_template('recover_method.html', method=method)
    
    try:
        # Validasi input files
        if 'watermarked_image' not in request.files or 'overhead_data' not in request.files:
            flash('Harap upload gambar watermarked dan file overhead data')
            return redirect(request.url)
        
        img_file = request.files['watermarked_image']
        overhead_file = request.files['overhead_data']
        
        if img_file.filename == '' or overhead_file.filename == '':
            flash('Harap pilih file gambar dan overhead data')
            return redirect(request.url)
        
        # Baca gambar watermarked
        img_file.seek(0)
        img_data = img_file.read()
        img_array = np.frombuffer(img_data, np.uint8)
        watermarked_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Convert ke grayscale
        watermarked_gray = rgb_to_grayscale(watermarked_img)
        
        # Baca overhead data
        overhead_file.seek(0)
        overhead_json = overhead_file.read().decode('utf-8')
        overhead_data = json.loads(overhead_json)
        
        # Validasi metode
        stored_method = overhead_data.get('method', method)
        if stored_method != method:
            # Auto-switch method if mismatch
            method = stored_method
            flash(f'Metode otomatis disesuaikan ke {method.replace("_", " ").title()} berdasarkan file overhead', 'info')
        
        if method == 'histogram_shifting':
            recovered_gray = recover_image(watermarked_gray, overhead_data)
            method_name = 'Histogram Shifting'
        else:  # rdwt
            # Validasi untuk RDWT recovery
            required_keys = ['original_values', 'positions', 'coeffs_shape']
            missing_keys = [key for key in required_keys if key not in overhead_data]
            
            if missing_keys:
                flash(f'Overhead data tidak lengkap untuk recovery RDWT. Keys yang hilang: {missing_keys}')
                return redirect(request.url)
            
            recovered_gray = recover_image_rdwt(watermarked_gray, overhead_data)
            method_name = 'True RDWT (Redundant DWT)'
        
        recovered_img = grayscale_to_rgb(recovered_gray)
        
        # Hitung PSNR untuk evaluasi kualitas recovery
        psnr_recovery = calculate_psnr(watermarked_gray, recovered_gray)
        
        # Simpan hasil recovery
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recovered_filename = f"recovered_{method}_{timestamp}.png"
        cv2.imwrite(os.path.join(RESULTS_FOLDER, recovered_filename), recovered_img)
        # Siapkan hasil recovery
        result = {
            'success': True,
            'method': method,
            'method_name': method_name,
            'watermarked_image': image_to_base64(watermarked_img),
            'recovered_image': image_to_base64(recovered_img),
            'psnr_recovery': round(psnr_recovery, 2),
            'recovered_filename': recovered_filename,
            'method_info': {
                'description': f'{method_name} Image Recovery',
                'parameters': overhead_data
            }
        }
        
        return render_template('recover_result.html', result=result)
        
    except json.JSONDecodeError:
        flash('Format file overhead data tidak valid')
        return redirect(request.url)
    except Exception as e:
        error_msg = str(e)
        print(f"Recovery error: {error_msg}")
        flash(f'Error saat recovery: {error_msg}')
        return redirect(request.url)

@app.route('/attack_test/<method>', methods=['GET', 'POST'])
def attack_test_method(method):
    """Test ketahanan watermark terhadap berbagai serangan."""
    if method not in ['histogram_shifting', 'rdwt']:
        flash('Metode tidak valid')
        return redirect(url_for('index'))
    
    if request.method == 'GET':
        return render_template('attack_test.html', method=method)
    
    try:
        # Validasi input files
        if 'watermarked_image' not in request.files or 'overhead_data' not in request.files:
            flash('Harap upload gambar watermarked dan file overhead data')
            return redirect(request.url)
        
        img_file = request.files['watermarked_image']
        overhead_file = request.files['overhead_data']
        
        # Baca gambar watermarked
        img_file.seek(0)
        img_data = img_file.read()
        img_array = np.frombuffer(img_data, np.uint8)
        watermarked_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Baca overhead data
        overhead_file.seek(0)
        overhead_json = overhead_file.read().decode('utf-8')
        overhead_data = json.loads(overhead_json)
        is_color = overhead_data.get('is_color', False)
        
        # Ambil parameter serangan dari form
        attack_types = request.form.getlist('attack_types')
        if not attack_types:
            flash('Pilih minimal satu jenis serangan')
            return redirect(request.url)
        
        # Initialize attack object
        attack = Attack()
        results = []
        
        # Test setiap jenis serangan
        for attack_type in attack_types:
            try:
                print(f"Testing {attack_type} attack...")
                
                # Apply serangan
                if attack_type == 'gaussian_noise':
                    attacked_img_color = attack.gaussian_noise(watermarked_img, std=10)
                elif attack_type == 'salt_pepper':
                    attacked_img_color = attack.salt_pepper_noise(watermarked_img, prob=0.01)
                elif attack_type == 'median_filter':
                    attacked_img_color = attack.median_filter(watermarked_img, kernel_size=3)
                elif attack_type == 'gaussian_blur':
                    attacked_img_color = attack.gaussian_blur(watermarked_img, kernel_size=5, sigma=1.0)
                elif attack_type == 'compression':
                    attacked_img_color = attack.jpeg_compression(watermarked_img, quality=70)
                elif attack_type == 'rotation':
                    attacked_img_color = attack.rotation(watermarked_img, angle=5)
                elif attack_type == 'scaling':
                    attacked_img_color = attack.scaling(watermarked_img, scale_factor=0.8)
                elif attack_type == 'cropping':
                    attacked_img_color = attack.cropping(watermarked_img, crop_ratio=0.1)
                else:
                    continue
                
                # Extract channel for processing
                if is_color and len(attacked_img_color.shape) == 3:
                    attacked_img_process = attacked_img_color[:, :, 0]
                else:
                    attacked_img_process = rgb_to_grayscale(attacked_img_color)
                
                # Extract watermark dari gambar yang diserang
                if method == 'histogram_shifting':
                    extracted_bits = extract_watermark(attacked_img_process, overhead_data)
                else:  # rdwt
                    extracted_bits = extract_watermark_rdwt(attacked_img_process, overhead_data)
                
                # Convert ke teks
                extracted_text = bits_to_text(extracted_bits)
                original_text = overhead_data.get('original_text', '')
                
                # Hitung akurasi
                accuracy = calculate_text_accuracy(original_text, extracted_text)
                
                # Hitung PSNR
                psnr = calculate_psnr(watermarked_img, attacked_img_color)
                
                results.append({
                    'attack_type': attack_type,
                    'attack_name': attack_type.replace('_', ' ').title(),
                    'attacked_image': image_to_base64(attacked_img_color),
                    'extracted_text': extracted_text,
                    'accuracy': round(accuracy, 2) if accuracy is not None else 0,
                    'psnr': round(psnr, 2),
                    'success': accuracy >= 80 if accuracy is not None else False
                })
                
            except Exception as e:
                print(f"Error in {attack_type} attack: {e}")
                results.append({
                    'attack_type': attack_type,
                    'attack_name': attack_type.replace('_', ' ').title(),
                    'attacked_image': None,
                    'extracted_text': f'Error: {str(e)}',
                    'accuracy': 0,
                    'psnr': 0,
                    'success': False
                })
        
        # Hitung statistik overall
        successful_attacks = sum(1 for r in results if r['success'])
        avg_accuracy = np.mean([r['accuracy'] for r in results if r['accuracy'] > 0])
        
        attack_result = {
            'success': True,
            'method': method,
            'method_name': 'Histogram Shifting' if method == 'histogram_shifting' else 'True RDWT',
            'original_watermarked': image_to_base64(watermarked_img),
            'original_text': overhead_data.get('original_text', ''),
            'results': results,
            'statistics': {
                'total_attacks': len(results),
                'successful_extractions': successful_attacks,
                'success_rate': round((successful_attacks / len(results)) * 100, 1),
                'average_accuracy': round(avg_accuracy, 2) if not np.isnan(avg_accuracy) else 0
            }
        }
        
        return render_template('attack_result.html', result=attack_result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Attack test error: {error_msg}")
        flash(f'Error saat testing serangan: {error_msg}')
        return redirect(request.url)

# ===============================
# API ENDPOINTS
# ===============================

@app.route('/api/embed', methods=['POST'])
def api_embed():
    """API endpoint untuk embedding watermark."""
    try:
        data = request.get_json()
        method = data.get('method', 'histogram_shifting')
        
        if method not in ['histogram_shifting', 'rdwt']:
            return jsonify({'error': 'Invalid method'}), 400
        
        # Process image dari base64
        img_data = base64.b64decode(data['image'].split(',')[1])
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        gray_img = rgb_to_grayscale(img)
        
        # Convert teks ke bits
        watermark_text = data.get('text', '')
        watermark_bits = text_to_bits(watermark_text)
        
        # Embed berdasarkan metode
        if method == 'histogram_shifting':
            params = data.get('parameters', {})
            watermarked_gray, overhead_data = embed_watermark(
                gray_img, watermark_bits,
                strength=params.get('strength', 3),
                block_size=params.get('block_size', 64),
                redundancy=params.get('redundancy', 3)
            )
        else:  # rdwt
            watermarked_gray, overhead_data = embed_watermark_rdwt(process_img, watermark_bits)
        
        watermarked_img = grayscale_to_rgb(watermarked_gray)
        psnr = calculate_psnr(img, watermarked_img)
        
        # Add metadata
        overhead_data['method'] = method
        overhead_data['original_text'] = watermark_text
        
        return jsonify({
            'success': True,
            'watermarked_image': image_to_base64(watermarked_img),
            'overhead_data': overhead_data,
            'psnr': round(psnr, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/extract', methods=['POST'])
def api_extract():
    """API endpoint untuk extracting watermark."""
    try:
        data = request.get_json()
        method = data.get('method', 'histogram_shifting')
        
        # Process image dari base64
        img_data = base64.b64decode(data['image'].split(',')[1])
        img_array = np.frombuffer(img_data, np.uint8)
        watermarked_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        watermarked_gray = rgb_to_grayscale(watermarked_img)
        
        overhead_data = data['overhead_data']
        
        # Extract berdasarkan metode
        if method == 'histogram_shifting':
            extracted_bits = extract_watermark(watermarked_process, overhead_data)
        else:  # rdwt
            extracted_bits = extract_watermark_rdwt(watermarked_process, overhead_data)
        
        extracted_text = bits_to_text(extracted_bits)
        
        return jsonify({
            'success': True,
            'extracted_text': extracted_text,
            'extracted_bits': len(extracted_bits)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===============================
# UTILITY ROUTES
# ===============================

@app.route('/download/<filename>')
def download_file(filename):
    """Download file hasil dari folder results."""
    try:
        return send_file(
            os.path.join(RESULTS_FOLDER, filename),
            as_attachment=True,
            download_name=filename
        )
    except FileNotFoundError:
        flash('File tidak ditemukan')
        return redirect(url_for('index'))

@app.route('/api/capacity', methods=['POST'])
def api_calculate_capacity():
    """API untuk menghitung kapasitas watermark."""
    try:
        data = request.get_json()
        width = data.get('width', 512)
        height = data.get('height', 512)
        method = data.get('method', 'histogram_shifting')
        
        if method == 'histogram_shifting':
            block_size = data.get('block_size', 16)
            redundancy = data.get('redundancy', 1)
            
            blocks_x = width // block_size
            blocks_y = height // block_size
            total_blocks = blocks_x * blocks_y
            capacity_bits = total_blocks // redundancy
            capacity_chars = capacity_bits // 16  # Estimasi konservatif
            
        else:  # rdwt
            capacity_bits = calculate_capacity((height, width))
            capacity_chars = capacity_bits // 16  # Estimasi untuk RDWT
        
        return jsonify({
            'capacity_bits': capacity_bits,
            'capacity_chars': capacity_chars,
            'method': method
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare')
def compare_methods():
    """Halaman perbandingan kedua metode."""
    comparison_data = {
        'histogram_shifting': {
            'name': 'Histogram Shifting',
            'description': 'Metode tradisional berbasis modifikasi histogram',
            'advantages': [
                'Implementasi sederhana dan cepat',
                'Kualitas visual yang baik (PSNR tinggi)',
                'Cocok untuk gambar natural dengan histogram yang jelas',
                'Overhead data minimal'
            ],
            'disadvantages': [
                'Rentan terhadap serangan (noise, filtering, kompresi)',
                'Kapasitas terbatas berdasarkan ukuran blok',
                'Performa buruk pada gambar dengan histogram flat',
                'Tidak mendukung recovery sempurna'
            ],
            'best_for': [
                'Aplikasi dengan kebutuhan kecepatan tinggi',
                'Gambar dengan histogram yang jelas',
                'Watermark teks pendek',
                'Lingkungan dengan risiko serangan rendah'
            ]
        },
        'rdwt': {
            'name': 'True RDWT (Redundant DWT)',
            'description': 'Metode advanced menggunakan Stationary Wavelet Transform',
            'advantages': [
                'Sangat tahan terhadap berbagai jenis serangan',
                'Parameter otomatis berdasarkan analisis gambar',
                'Kapasitas besar dan fleksibel',
                'Mendukung perfect recovery gambar asli',
                'Adaptif terhadap karakteristik gambar'
            ],
            'disadvantages': [
                'Kompleksitas komputasi lebih tinggi',
                'Overhead data lebih besar',
                'Membutuhkan lebih banyak memori',
                'Waktu proses lebih lama'
            ],
            'best_for': [
                'Aplikasi yang membutuhkan keamanan tinggi',
                'Lingkungan dengan risiko serangan tinggi',
                'Watermark teks panjang atau data penting',
                'Sistem yang membutuhkan reversibility'
            ]
        }
    }
    
    return render_template('compare.html', comparison=comparison_data)

@app.route('/help')
def help_page():
    """Halaman bantuan dan dokumentasi."""
    help_data = {
        'embedding': {
            'title': 'Cara Embedding Watermark',
            'steps': [
                'Pilih metode watermarking (Histogram Shifting atau RDWT)',
                'Upload gambar host (PNG, JPG, JPEG, BMP, TIFF)',
                'Masukkan teks watermark yang ingin disembunyikan',
                'Sesuaikan parameter jika menggunakan Histogram Shifting',
                'Klik "Embed Watermark" untuk memproses',
                'Download hasil gambar watermarked dan file overhead data'
            ]
        },
        'extraction': {
            'title': 'Cara Extract Watermark',
            'steps': [
                'Pilih metode yang sama dengan saat embedding',
                'Upload gambar watermarked',
                'Upload file overhead data (.json)',
                'Klik "Extract Watermark" untuk memproses',
                'Lihat hasil ekstraksi teks watermark dan akurasinya'
            ]
        },
        'recovery': {
            'title': 'Cara Recovery Gambar Asli',
            'steps': [
                'Pilih metode yang sama dengan saat embedding',
                'Upload gambar watermarked',
                'Upload file overhead data (.json)',
                'Klik "Recover Image" untuk memproses',
                'Download gambar asli yang telah di-recover'
            ]
        },
        'tips': [
            'Gunakan gambar dengan ukuran minimal 256x256 untuk hasil optimal',
            'Histogram Shifting cocok untuk teks pendek (< 50 karakter)',
            'RDWT dapat menangani teks yang lebih panjang (< 200 karakter)',
            'Simpan file overhead data dengan aman - diperlukan untuk ekstraksi',
            'Test ketahanan watermark dengan fitur Attack Test',
            'Gunakan format PNG untuk menjaga kualitas gambar'
        ]
    }
    
    return render_template('help.html', help_data=help_data)

# ===============================
# ERROR HANDLERS
# ===============================

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', 
                         error_code=404,
                         error_message="Halaman tidak ditemukan"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html',
                         error_code=500, 
                         error_message="Terjadi kesalahan internal server"), 500

@app.errorhandler(413)
def too_large(error):
    flash('File terlalu besar. Maksimal 16MB.')
    return redirect(url_for('index'))

# ===============================
# MAIN APPLICATION
# ===============================

if __name__ == '__main__':
    print("Starting Dual Watermarking System...")
    print("Available methods:")
    print("- Histogram Shifting: Traditional histogram-based method")
    print("- True RDWT: Advanced redundant wavelet transform method")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Results folder: {RESULTS_FOLDER}")
    print("Server running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)