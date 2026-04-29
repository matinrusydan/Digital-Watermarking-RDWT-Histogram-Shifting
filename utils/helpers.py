import os
import cv2
import numpy as np
import base64
from PIL import Image
import io
from config.settings import Config

ALLOWED_EXTENSIONS = Config.ALLOWED_EXTENSIONS

def calculate_capacity(image_shape):
    """
    Hitung kapasitas watermark untuk gambar dengan ukuran tertentu.
    
    Args:
        image_shape: Tuple (height, width) dari gambar
        
    Returns:
        int: Kapasitas dalam bits
    """
    height, width = image_shape
    
    # Estimasi kapasitas berdasarkan area efektif
    # Menggunakan level 3 wavelet, subband detail ~1/64 dari gambar asli
    effective_area = (height // 8) * (width // 8)
    
    # Faktor safety untuk menghindari border dan posisi tidak stabil
    safety_factor = 0.7
    capacity = int(effective_area * safety_factor)
    
    return max(capacity, 100)  # Minimal 100 bits

def estimate_text_capacity(image_shape, encoding='utf-8'):
    """
    Estimasi kapasitas teks berdasarkan ukuran gambar.
    
    Args:
        image_shape: Tuple (height, width) dari gambar
        encoding: Encoding yang digunakan (default: utf-8)
        
    Returns:
        dict: Informasi kapasitas teks
    """
    bit_capacity = calculate_capacity(image_shape)
    
    # Estimasi karakter berdasarkan encoding
    if encoding == 'utf-8':
        # UTF-8: 1-4 bytes per karakter, rata-rata 2 bytes untuk teks campuran
        # Tambahkan overhead untuk header panjang (16 bits)
        available_bits = bit_capacity - 16
        estimated_chars = available_bits // 16  # Konservatif: 2 bytes per char
    else:
        # ASCII atau encoding lain
        available_bits = bit_capacity - 16
        estimated_chars = available_bits // 8
    
    return {
        'bit_capacity': bit_capacity,
        'estimated_chars': max(estimated_chars, 10),
        'recommended_max': max(estimated_chars // 2, 5),  # Lebih konservatif
        'encoding': encoding
    }

def validate_watermark_capacity(image_shape, watermark_text, method='rdwt'):
    """
    Validasi apakah teks watermark bisa diembed dalam gambar.
    
    Args:
        image_shape: Tuple (height, width) dari gambar
        watermark_text: Teks yang akan di-embed
        method: Metode watermarking ('rdwt' atau 'histogram_shifting')
        
    Returns:
        tuple: (is_valid, message, capacity_info)
    """
    if method == 'rdwt':
        capacity_info = estimate_text_capacity(image_shape)
        bit_capacity = capacity_info['bit_capacity']
    else:
        # Untuk histogram shifting, estimasi berdasarkan block
        bit_capacity = (image_shape[0] // 64) * (image_shape[1] // 64) // 3  # Konservatif
        capacity_info = {'bit_capacity': bit_capacity, 'estimated_chars': bit_capacity // 16}
    
    # Hitung bits yang diperlukan
    watermark_bits = text_to_bits(watermark_text)
    required_bits = len(watermark_bits)
    
    if required_bits > bit_capacity:
        return False, f"Teks terlalu panjang. Diperlukan: {required_bits} bits, Tersedia: {bit_capacity} bits", capacity_info
    
    # Warning jika mendekati kapasitas
    usage_percent = (required_bits / bit_capacity) * 100
    if usage_percent > 80:
        return True, f"Peringatan: Menggunakan {usage_percent:.1f}% kapasitas", capacity_info
    
    return True, f"OK: Menggunakan {usage_percent:.1f}% kapasitas", capacity_info

def get_method_recommendations(image_shape, watermark_text):
    """
    Berikan rekomendasi metode berdasarkan gambar dan teks.
    
    Args:
        image_shape: Tuple (height, width) dari gambar
        watermark_text: Teks yang akan di-embed
        
    Returns:
        dict: Rekomendasi untuk setiap metode
    """
    height, width = image_shape
    text_length = len(watermark_text)
    
    recommendations = {
        'histogram_shifting': {
            'suitable': True,
            'score': 0,
            'reasons': []
        },
        'rdwt': {
            'suitable': True,
            'score': 0,
            'reasons': []
        }
    }
    
    # Analisis ukuran gambar
    if width < 256 or height < 256:
        recommendations['histogram_shifting']['score'] -= 2
        recommendations['histogram_shifting']['reasons'].append('Gambar terlalu kecil untuk optimal')
        recommendations['rdwt']['score'] -= 1
        recommendations['rdwt']['reasons'].append('Gambar kecil, kapasitas terbatas')
    
    # Analisis panjang teks
    if text_length > 100:
        recommendations['histogram_shifting']['score'] -= 3
        recommendations['histogram_shifting']['reasons'].append('Teks terlalu panjang untuk metode ini')
        recommendations['rdwt']['score'] += 2
        recommendations['rdwt']['reasons'].append('Cocok untuk teks panjang')
    elif text_length < 30:
        recommendations['histogram_shifting']['score'] += 2
        recommendations['histogram_shifting']['reasons'].append('Optimal untuk teks pendek')
        recommendations['rdwt']['score'] += 1
        recommendations['rdwt']['reasons'].append('Bisa handle teks pendek dengan baik')
    
    # Analisis area gambar
    total_pixels = width * height
    if total_pixels > 512 * 512:
        recommendations['rdwt']['score'] += 2
        recommendations['rdwt']['reasons'].append('Gambar besar, cocok untuk RDWT')
    
    # Validasi final
    for method in ['histogram_shifting', 'rdwt']:
        is_valid, message, _ = validate_watermark_capacity(image_shape, watermark_text, method)
        if not is_valid:
            recommendations[method]['suitable'] = False
            recommendations[method]['score'] = -10
            recommendations[method]['reasons'].append(message)
        else:
            recommendations[method]['reasons'].append(message)
    
    return recommendations

def format_file_size(bytes_size):
    """Format ukuran file ke format yang mudah dibaca."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def get_image_info(image):
    """
    Dapatkan informasi detail tentang gambar.
    
    Args:
        image: numpy array gambar
        
    Returns:
        dict: Informasi gambar
    """
    if len(image.shape) == 3:
        height, width, channels = image.shape
        is_color = True
    else:
        height, width = image.shape
        channels = 1
        is_color = False
    
    # Hitung statistik gambar
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    
    # Analisis histogram untuk grayscale
    if not is_color:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_entropy = -np.sum(hist * np.log2(hist + 1e-10)) / len(hist)
    else:
        # Convert ke grayscale untuk analisis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_entropy = -np.sum(hist * np.log2(hist + 1e-10)) / len(hist)
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'is_color': is_color,
        'total_pixels': width * height,
        'mean_intensity': round(mean_intensity, 2),
        'std_intensity': round(std_intensity, 2),
        'histogram_entropy': round(hist_entropy, 3),
        'file_size_estimate': format_file_size(width * height * channels),
        'aspect_ratio': round(width / height, 2)
    }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(img_array):
    """Convert numpy array to base64 string for web display"""
    if len(img_array.shape) == 3:
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    else:
        img_pil = Image.fromarray(img_array)
    
    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

def rgb_to_grayscale(img):
    """Convert RGB image to grayscale for processing"""
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        if np.array_equal(b, g) and np.array_equal(g, r):
            return b
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def grayscale_to_rgb(gray_img):
    """Convert grayscale image back to RGB"""
    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

def text_to_bits(text):
    """Convert text to binary bits dengan encoding UTF-8 yang aman."""
    if not text:
        return []
    
    try:
        # Gunakan UTF-8 encoding
        text_bytes = text.encode('utf-8')
        text_bits = []
        
        # Tambahkan header panjang (16 bits = max 65535 bytes)
        text_length = len(text_bytes)
        if text_length > 65535:
            raise ValueError("Teks terlalu panjang untuk encoding")
            
        length_bits = [int(b) for b in format(text_length, '016b')]
        text_bits.extend(length_bits)
        
        # Tambahkan bits teks sebenarnya
        for byte in text_bytes:
            byte_bits = [int(b) for b in format(byte, '08b')]
            text_bits.extend(byte_bits)
        
        return text_bits
        
    except Exception as e:
        print(f"Error in text_to_bits: {e}")
        return []

def bits_to_text(bits):
    """Convert binary bits kembali ke teks dengan decoding yang aman."""
    try:
        if len(bits) < 16:
            return "Error: Tidak cukup bits untuk header panjang"
        
        # Extract panjang dari 16 bits pertama
        length_bits = bits[:16]
        text_length = int(''.join(map(str, length_bits)), 2)
        
        # Hitung total bits yang diharapkan
        expected_bits = 16 + (text_length * 8)
        
        if len(bits) < expected_bits:
            # Gunakan apa yang tersedia
            text_length = (len(bits) - 16) // 8
        
        # Extract text bits (skip 16 bits pertama)
        text_bits = bits[16:16 + (text_length * 8)]
        
        if len(text_bits) % 8 != 0:
            # Truncate ke byte lengkap
            text_bits = text_bits[:-(len(text_bits) % 8)]
        
        # Convert bits ke bytes
        text_bytes = []
        for i in range(0, len(text_bits), 8):
            byte_bits = text_bits[i:i+8]
            if len(byte_bits) == 8:
                byte_value = int(''.join(map(str, byte_bits)), 2)
                text_bytes.append(byte_value)
        
        # Convert ke byte array dan decode
        byte_array = bytes(text_bytes)
        
        try:
            decoded_text = byte_array.decode('utf-8')
            return decoded_text
        except UnicodeDecodeError:
            # Coba dengan error handling
            decoded_text = byte_array.decode('utf-8', errors='replace')
            return f"Partial decode: {decoded_text}"
            
    except Exception as e:
        return f"Decoding error: {str(e)}"

def create_default_overhead_data():
    """Create default overhead data when not provided by embed function."""
    return {
        'wavelet': 'db4',
        'levels': 3,
        'subband': 'HH',
        'strength': 0.1,
        'method': 'rdwt',
        'analysis': {
            'coefficients_modified': 0,
            'embedding_strength_used': 0.1,
            'robustness_score': 'medium'
        }
    }

