import numpy as np
import cv2
import pywt
from scipy import ndimage
import json
import warnings
warnings.filterwarnings('ignore')

class RDWTWatermarking:
    """
    Fixed RDWT (Redundant Discrete Wavelet Transform) Watermarking Implementation
    Menggunakan Stationary Wavelet Transform untuk ketahanan yang lebih baik
    """
    
    def __init__(self):
        self.supported_wavelets = ['haar', 'db4', 'db8', 'bior2.2', 'bior4.4', 'coif2']
        self.subbands = ['LL', 'LH', 'HL', 'HH']
        
    def analyze_image_characteristics(self, img):
        """Analisis karakteristik gambar untuk parameter otomatis."""
        analysis = {}
        
        # Analisis tekstur menggunakan gradient
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        analysis['texture_complexity'] = np.std(gradient_magnitude)
        analysis['mean_intensity'] = np.mean(img)
        analysis['std_intensity'] = np.std(img)
        analysis['dynamic_range'] = np.max(img) - np.min(img)
        
        # Analisis histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_norm = hist / np.sum(hist)
        hist_norm = hist_norm[hist_norm > 0]  # Remove zeros to avoid log(0)
        analysis['histogram_entropy'] = -np.sum(hist_norm * np.log2(hist_norm))
        
        # Klasifikasi jenis gambar
        if analysis['texture_complexity'] > 30:
            analysis['image_type'] = 'high_texture'
        elif analysis['texture_complexity'] > 15:
            analysis['image_type'] = 'medium_texture'  
        else:
            analysis['image_type'] = 'smooth'
            
        return analysis
    
    def select_optimal_parameters(self, img, watermark_length):
        """Pilih parameter optimal berdasarkan analisis gambar."""
        analysis = self.analyze_image_characteristics(img)
        
        # Pilih wavelet berdasarkan karakteristik gambar
        if analysis['image_type'] == 'high_texture':
            wavelet = 'db4'  # Lebih stabil daripada db8
            levels = 2       # Kurangi level untuk menghindari masalah
            strength = 0.05
        elif analysis['image_type'] == 'medium_texture':
            wavelet = 'db4'
            levels = 2  
            strength = 0.08
        else:
            wavelet = 'haar'  # Sederhana untuk gambar smooth
            levels = 2
            strength = 0.1
            
        # Pilih subband berdasarkan kebutuhan ketahanan
        if watermark_length > 100:
            subband = 'LH'  # Lebih stabil untuk watermark panjang
        else:
            subband = 'HL'  # Baik untuk watermark pendek
            
        # Adjust strength berdasarkan dynamic range
        if analysis['dynamic_range'] < 100:
            strength *= 1.5  # Tingkatkan untuk gambar low contrast
            
        return {
            'wavelet': wavelet,
            'levels': levels,
            'subband': subband,
            'strength': strength,
            'analysis': analysis
        }
    
    def stationary_wavelet_transform(self, img, wavelet, levels):
        """Implementasi Stationary Wavelet Transform (SWT) dengan error handling yang lebih baik."""
        try:
            # Pastikan level tidak terlalu tinggi
            max_levels = pywt.swt_max_level(min(img.shape))
            if levels > max_levels:
                levels = max_levels
                print(f"Level reduced to {levels} (max possible for image size)")
            
            # Check if SWT can be performed
            if levels < 1:
                raise ValueError("Image too small for SWT")
            
            # PERBAIKAN: Coba SWT dengan lebih hati-hati
            print(f"Attempting SWT with wavelet: {wavelet}, levels: {levels}")
            coeffs = pywt.swt2(img, wavelet, level=levels, trim_approx=False)
            
            # Validasi struktur hasil SWT
            if not isinstance(coeffs, list) or len(coeffs) == 0:
                raise ValueError("SWT returned invalid structure")
            
            # PERBAIKAN UTAMA: Periksa struktur SWT yang sebenarnya
            print(f"SWT returned {len(coeffs)} levels")
            for i, level_coeffs in enumerate(coeffs):
                print(f"Level {i}: type={type(level_coeffs)}")
                
                # PyWavelets SWT2 dapat mengembalikan struktur yang berbeda
                # Kadang mengembalikan array tunggal, kadang tuple
                if isinstance(level_coeffs, np.ndarray):
                    # Jika level_coeffs adalah array tunggal, ini kemungkinan approximation coefficients
                    # Kita perlu mengonversi ke struktur yang konsisten
                    print(f"Level {i} is ndarray with shape {level_coeffs.shape}")
                    # Untuk SWT, kita butuh struktur (LL, (LH, HL, HH))
                    # Jika hanya ada satu array, kita tidak bisa melanjutkan dengan SWT
                    raise ValueError("SWT structure incomplete - missing detail coefficients")
                    
                elif isinstance(level_coeffs, (tuple, list)):
                    if len(level_coeffs) == 2:
                        approx, details = level_coeffs
                        if isinstance(details, (tuple, list)) and len(details) == 3:
                            print(f"Level {i} has correct structure: (LL, (LH, HL, HH))")
                        else:
                            print(f"Level {i} details structure wrong: {type(details)}, len={len(details) if hasattr(details, '__len__') else 'N/A'}")
                            raise ValueError(f"SWT level {i} detail structure invalid")
                    elif len(level_coeffs) == 4:
                        # Kadang SWT mengembalikan (LL, LH, HL, HH) langsung
                        print(f"Level {i} has 4-element structure - converting to nested format")
                        # Convert ke format (LL, (LH, HL, HH))
                        ll, lh, hl, hh = level_coeffs
                        coeffs[i] = (ll, (lh, hl, hh))
                    else:
                        print(f"Level {i} has unexpected length: {len(level_coeffs)}")
                        raise ValueError(f"SWT level {i} structure invalid")
                else:
                    print(f"Level {i} has unexpected type: {type(level_coeffs)}")
                    raise ValueError(f"SWT level {i} structure invalid")
            
            print(f"SWT structure validation successful")
            return coeffs, 'swt'
            
        except Exception as e:
            print(f"SWT failed: {e}, falling back to DWT")
            try:
                # Fallback ke DWT biasa
                coeffs = pywt.wavedec2(img, wavelet, level=levels)
                
                # Validasi struktur DWT
                if not isinstance(coeffs, list) or len(coeffs) < 2:
                    raise ValueError("DWT returned invalid structure")
                
                print(f"DWT fallback successful, got {len(coeffs)} levels")
                return coeffs, 'dwt'
                
            except Exception as e2:
                print(f"DWT also failed: {e2}")
                # Fallback terakhir: DWT dengan level 1
                try:
                    coeffs = pywt.wavedec2(img, 'haar', level=1)
                    print("Final fallback to Haar wavelet, level 1")
                    return coeffs, 'dwt'
                except Exception as e3:
                    print(f"All transforms failed: {e3}")
                    raise e3
    
    def inverse_stationary_wavelet_transform(self, coeffs, wavelet, transform_type='swt'):
        """Implementasi Inverse Transform dengan handling yang lebih baik."""
        try:
            if transform_type == 'swt':
                img_reconstructed = pywt.iswt2(coeffs, wavelet)
            else:  # dwt
                img_reconstructed = pywt.waverec2(coeffs, wavelet)
            return img_reconstructed
        except Exception as e:
            print(f"Error in inverse transform: {e}")
            # Try alternative method
            try:
                if transform_type == 'swt':
                    img_reconstructed = pywt.waverec2(coeffs, wavelet)
                else:
                    img_reconstructed = pywt.iswt2(coeffs, wavelet)
                return img_reconstructed
            except Exception as e2:
                print(f"Both inverse methods failed: {e2}")
                raise e2
    
    def select_embedding_positions(self, subband_shape, watermark_length, method='adaptive'):
        """Pilih posisi embedding yang optimal."""
        rows, cols = subband_shape
        total_coeffs = rows * cols
        
        if watermark_length > total_coeffs:
            raise ValueError(f"Watermark terlalu panjang: {watermark_length} > {total_coeffs}")
        
        # Calculate safe margins
        margin = max(2, min(rows//10, cols//10))
        safe_rows = rows - 2*margin
        safe_cols = cols - 2*margin
        
        if safe_rows <= 0 or safe_cols <= 0:
            margin = 1
            safe_rows = rows - 2
            safe_cols = cols - 2
        
        if method == 'adaptive':
            positions = []
            safe_total = safe_rows * safe_cols
            step = max(1, safe_total // watermark_length)
            
            for i in range(0, safe_total, step):
                if len(positions) >= watermark_length:
                    break
                row = (i // safe_cols) + margin
                col = (i % safe_cols) + margin
                positions.append((row, col))
            
            # Jika tidak cukup, tambahkan posisi tambahan
            while len(positions) < watermark_length:
                row = np.random.randint(margin, rows-margin)
                col = np.random.randint(margin, cols-margin)
                if (row, col) not in positions:
                    positions.append((row, col))
                    
        else:  # random
            positions = []
            while len(positions) < watermark_length:
                row = np.random.randint(margin, rows-margin)
                col = np.random.randint(margin, cols-margin)
                if (row, col) not in positions:
                    positions.append((row, col))
        
        return positions[:watermark_length]
    
    # PERBAIKAN ERROR PADA RDWT WATERMARKING
    # Ganti bagian embed_watermark method di class RDWTWatermarking

    def embed_watermark(self, img, watermark_bits, params=None):
        """Embed watermark menggunakan RDWT dengan error handling yang diperbaiki."""
        if params is None:
            params = self.select_optimal_parameters(img, len(watermark_bits))
        
        wavelet = params['wavelet']
        levels = params['levels']
        subband = params['subband']
        strength = params['strength']
        
        # Wavelet Transform dengan error handling
        try:
            coeffs, transform_type = self.stationary_wavelet_transform(img, wavelet, levels)
        except Exception as e:
            print(f"Transform failed: {e}")
            raise e
        
        # Handle different coefficient structures - PERBAIKAN UTAMA
        try:
            if transform_type == 'swt':
                # SWT returns list of (LL, (LH, HL, HH)) tuples
                if levels > len(coeffs):
                    levels = len(coeffs)
                
                level_coeffs = coeffs[levels-1]  # Use last level
                
                # PERBAIKAN: Struktur sudah diperbaiki di stationary_wavelet_transform
                print(f"Using SWT level {levels-1}, coeffs type: {type(level_coeffs)}")
                
                if isinstance(level_coeffs, (tuple, list)) and len(level_coeffs) == 2:
                    approx_coeffs, detail_coeffs = level_coeffs
                    
                    # Pastikan detail_coeffs adalah tuple/list dengan 3 elemen
                    if not isinstance(detail_coeffs, (tuple, list)) or len(detail_coeffs) != 3:
                        print(f"Detail coeffs structure unexpected after fix: {type(detail_coeffs)}, len: {len(detail_coeffs) if hasattr(detail_coeffs, '__len__') else 'no len'}")
                        # Fallback ke DWT
                        raise ValueError("SWT structure still not as expected, falling back to DWT")
                    
                    if subband == 'LL':
                        target_subband = approx_coeffs
                    elif subband == 'LH':
                        target_subband = detail_coeffs[0]
                    elif subband == 'HL':
                        target_subband = detail_coeffs[1]
                    else:  # HH
                        target_subband = detail_coeffs[2]
                else:
                    print(f"Unexpected level coeffs structure after fix: {type(level_coeffs)}")
                    raise ValueError(f"Unexpected SWT coefficient structure after fix: {type(level_coeffs)}")
                            
            else:  # DWT
                # DWT returns [LL, (LH, HL, HH), (LH, HL, HH), ...]
                if subband == 'LL':
                    target_subband = coeffs[0]
                else:
                    # Use first detail level
                    if len(coeffs) < 2:
                        raise ValueError("DWT coefficients incomplete")
                        
                    detail_coeffs = coeffs[1]
                    if not isinstance(detail_coeffs, (tuple, list)) or len(detail_coeffs) != 3:
                        raise ValueError(f"DWT detail coefficients malformed: {type(detail_coeffs)}")
                        
                    if subband == 'LH':
                        target_subband = detail_coeffs[0]
                    elif subband == 'HL':
                        target_subband = detail_coeffs[1]
                    else:  # HH
                        target_subband = detail_coeffs[2]
                                
        except Exception as e:
            print(f"Error accessing subband: {e}")
            print(f"Coeffs structure: {type(coeffs)}, length: {len(coeffs) if hasattr(coeffs, '__len__') else 'N/A'}")
            if hasattr(coeffs, '__len__') and len(coeffs) > 0:
                print(f"First element type: {type(coeffs[0])}")
            
            # PERBAIKAN: Fallback ke DWT dengan struktur yang lebih sederhana
            print("Attempting fallback to simple DWT...")
            try:
                # Coba DWT sederhana
                coeffs_simple = pywt.wavedec2(img, wavelet, level=1)  # Hanya 1 level
                transform_type = 'dwt'
                
                if subband == 'LL':
                    target_subband = coeffs_simple[0]
                else:
                    detail_coeffs = coeffs_simple[1]
                    if subband == 'LH':
                        target_subband = detail_coeffs[0]
                    elif subband == 'HL':
                        target_subband = detail_coeffs[1]
                    else:  # HH
                        target_subband = detail_coeffs[2]
                
                # Update coeffs untuk proses selanjutnya
                coeffs = coeffs_simple
                levels = 1
                
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                raise e
        
        # Pastikan target_subband adalah numpy array
        target_subband = np.array(target_subband, dtype=np.float64)
        
        # Pilih posisi embedding
        try:
            positions = self.select_embedding_positions(
                target_subband.shape, len(watermark_bits)
            )
        except Exception as e:
            print(f"Error selecting positions: {e}")
            raise e
        
        # Simpan nilai asli untuk recovery
        original_values = {}
        watermarked_subband = target_subband.copy()
        
        # Embed watermark bits
        for i, (row, col) in enumerate(positions):
            if i < len(watermark_bits):
                original_val = watermarked_subband[row, col]
                original_values[f"{row},{col}"] = float(original_val)
                
                # Quantization-based embedding
                if watermark_bits[i] == 1:
                    watermarked_subband[row, col] = original_val + strength * abs(original_val)
                else:
                    watermarked_subband[row, col] = original_val - strength * abs(original_val)
        
        # Update coefficients - PERBAIKAN
        try:
            if transform_type == 'swt':
                # Update SWT coefficients
                level_coeffs = list(coeffs[levels-1])
                
                if subband == 'LL':
                    level_coeffs[0] = watermarked_subband
                else:
                    if len(level_coeffs) >= 2 and isinstance(level_coeffs[1], (tuple, list)):
                        detail_coeffs = list(level_coeffs[1])
                        if subband == 'LH':
                            detail_coeffs[0] = watermarked_subband
                        elif subband == 'HL':
                            detail_coeffs[1] = watermarked_subband
                        else:  # HH
                            detail_coeffs[2] = watermarked_subband
                        level_coeffs[1] = tuple(detail_coeffs)
                    else:
                        raise ValueError("Cannot update SWT coefficients")
                
                coeffs[levels-1] = tuple(level_coeffs)
                
            else:  # DWT
                # Update DWT coefficients
                coeffs = list(coeffs)
                if subband == 'LL':
                    coeffs[0] = watermarked_subband
                else:
                    detail_coeffs = list(coeffs[1])
                    if subband == 'LH':
                        detail_coeffs[0] = watermarked_subband
                    elif subband == 'HL':
                        detail_coeffs[1] = watermarked_subband
                    else:  # HH
                        detail_coeffs[2] = watermarked_subband
                    coeffs[1] = tuple(detail_coeffs)
                        
        except Exception as e:
            print(f"Error updating coefficients: {e}")
            raise e
        
        # Inverse transform
        try:
            watermarked_img = self.inverse_stationary_wavelet_transform(coeffs, wavelet, transform_type)
        except Exception as e:
            print(f"Error in inverse transform: {e}")
            raise e
        
        # Clip values to valid range
        watermarked_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)
        
        # Prepare overhead data
        overhead_data = {
            'wavelet': wavelet,
            'levels': levels,
            'subband': subband,
            'strength': strength,
            'watermark_length': len(watermark_bits),
            'positions': positions,
            'original_values': original_values,
            'coeffs_shape': list(target_subband.shape),
            'image_shape': list(img.shape),
            'transform_type': transform_type,
            'analysis': params.get('analysis', {})
        }
        
        return watermarked_img, overhead_data

    def extract_watermark(self, watermarked_img, overhead_data):
        """Extract watermark dari gambar watermarked."""
        wavelet = overhead_data['wavelet']
        levels = overhead_data['levels']
        subband = overhead_data['subband']
        positions = overhead_data['positions']
        watermark_length = overhead_data['watermark_length']
        transform_type = overhead_data.get('transform_type', 'swt')
        
        # Wavelet transform
        try:
            coeffs, _ = self.stationary_wavelet_transform(watermarked_img, wavelet, levels)
        except Exception as e:
            print(f"Transform failed during extraction: {e}")
            raise e
        
        # Pilih subband yang sama
        try:
            if transform_type == 'swt':
                level_coeffs = coeffs[levels-1]
                approx_coeffs, detail_coeffs = level_coeffs
                
                if subband == 'LL':
                    target_subband = approx_coeffs
                elif subband == 'LH':
                    target_subband = detail_coeffs[0]
                elif subband == 'HL':
                    target_subband = detail_coeffs[1]
                else:  # HH
                    target_subband = detail_coeffs[2]
            else:  # DWT
                if subband == 'LL':
                    target_subband = coeffs[0]
                else:
                    detail_coeffs = coeffs[1]
                    if subband == 'LH':
                        target_subband = detail_coeffs[0]
                    elif subband == 'HL':
                        target_subband = detail_coeffs[1]
                    else:  # HH
                        target_subband = detail_coeffs[2]
        except Exception as e:
            print(f"Error accessing subband during extraction: {e}")
            raise e
        
        # Extract watermark bits
        extracted_bits = []
        original_values = overhead_data.get('original_values', {})
        
        for i, (row, col) in enumerate(positions):
            if i < watermark_length:
                current_val = target_subband[row, col]
                key = f"{row},{col}"
                
                if key in original_values:
                    original_val = original_values[key]
                    
                    # Deteksi bit berdasarkan perubahan
                    if current_val > original_val:
                        extracted_bits.append(1)
                    else:
                        extracted_bits.append(0)
                else:
                    # Fallback jika tidak ada nilai asli
                    # Gunakan threshold sederhana
                    if abs(current_val) > np.mean(np.abs(target_subband)):
                        extracted_bits.append(1)
                    else:
                        extracted_bits.append(0)
        
        return extracted_bits[:watermark_length]
    
    def recover_image(self, watermarked_img, overhead_data):
        """Recover gambar asli dari gambar watermarked."""
        wavelet = overhead_data['wavelet']
        levels = overhead_data['levels']
        subband = overhead_data['subband']
        positions = overhead_data['positions']
        original_values = overhead_data['original_values']
        transform_type = overhead_data.get('transform_type', 'swt')
        
        # Wavelet transform
        coeffs, _ = self.stationary_wavelet_transform(watermarked_img, wavelet, levels)
        
        # Pilih subband yang sama
        if transform_type == 'swt':
            level_coeffs = coeffs[levels-1]
            approx_coeffs, detail_coeffs = level_coeffs
            
            if subband == 'LL':
                target_subband = approx_coeffs.copy()
            elif subband == 'LH':
                target_subband = detail_coeffs[0].copy()
            elif subband == 'HL':
                target_subband = detail_coeffs[1].copy()
            else:  # HH
                target_subband = detail_coeffs[2].copy()
        else:  # DWT
            if subband == 'LL':
                target_subband = coeffs[0].copy()
            else:
                detail_coeffs = coeffs[1]
                if subband == 'LH':
                    target_subband = detail_coeffs[0].copy()
                elif subband == 'HL':
                    target_subband = detail_coeffs[1].copy()
                else:  # HH
                    target_subband = detail_coeffs[2].copy()
        
        # Restore original values
        for row, col in positions:
            key = f"{row},{col}"
            if key in original_values:
                target_subband[row, col] = original_values[key]
        
        # Update coefficients
        if transform_type == 'swt':
            level_coeffs = list(coeffs[levels-1])
            
            if subband == 'LL':
                level_coeffs[0] = target_subband
            else:
                detail_coeffs = list(level_coeffs[1])
                if subband == 'LH':
                    detail_coeffs[0] = target_subband
                elif subband == 'HL':
                    detail_coeffs[1] = target_subband
                else:  # HH
                    detail_coeffs[2] = target_subband
                level_coeffs[1] = tuple(detail_coeffs)
            
            coeffs[levels-1] = tuple(level_coeffs)
        else:  # DWT
            coeffs = list(coeffs)
            if subband == 'LL':
                coeffs[0] = target_subband
            else:
                detail_coeffs = list(coeffs[1])
                if subband == 'LH':
                    detail_coeffs[0] = target_subband
                elif subband == 'HL':
                    detail_coeffs[1] = target_subband
                else:  # HH
                    detail_coeffs[2] = target_subband
                coeffs[1] = tuple(detail_coeffs)
        
        # Inverse transform
        recovered_img = self.inverse_stationary_wavelet_transform(coeffs, wavelet, transform_type)
        
        # Clip values
        recovered_img = np.clip(recovered_img, 0, 255).astype(np.uint8)
        
        return recovered_img

# Global instance
rdwt_watermarking = RDWTWatermarking()

# ===============================
# PUBLIC API FUNCTIONS
# ===============================

def embed_watermark_rdwt(img, watermark_bits, params=None):
    """
    Embed watermark menggunakan RDWT method yang telah diperbaiki.
    
    Args:
        img: Input grayscale image (numpy array)
        watermark_bits: List of watermark bits
        params: Optional parameters dict
        
    Returns:
        tuple: (watermarked_image, overhead_data)
    """
    try:
        return rdwt_watermarking.embed_watermark(img, watermark_bits, params)
    except Exception as e:
        print(f"Error in embed_watermark_rdwt: {e}")
        raise e

def extract_watermark_rdwt(watermarked_img, overhead_data):
    """
    Extract watermark dari gambar menggunakan RDWT.
    
    Args:
        watermarked_img: Watermarked grayscale image
        overhead_data: Overhead data dari proses embedding
        
    Returns:
        list: Extracted watermark bits
    """
    return rdwt_watermarking.extract_watermark(watermarked_img, overhead_data)

def recover_image_rdwt(watermarked_img, overhead_data):
    """
    Recover gambar asli dari watermarked image.
    
    Args:
        watermarked_img: Watermarked grayscale image
        overhead_data: Overhead data dari proses embedding
        
    Returns:
        numpy.ndarray: Recovered original image
    """
    return rdwt_watermarking.recover_image(watermarked_img, overhead_data)

def calculate_psnr(img1, img2):
    """
    Hitung Peak Signal-to-Noise Ratio antara dua gambar.
    
    Args:
        img1, img2: Input images (numpy arrays)
        
    Returns:
        float: PSNR value in dB
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

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
    # Menggunakan level 2 wavelet, subband detail ~1/16 dari gambar asli
    effective_area = (height // 4) * (width // 4)
    
    # Faktor safety untuk menghindari border dan posisi tidak stabil
    safety_factor = 0.6
    capacity = int(effective_area * safety_factor)
    
    return max(capacity, 50)  # Minimal 50 bits

def get_image_limits(image_shape):
    """
    Dapatkan batas-batas gambar untuk watermarking.
    
    Args:
        image_shape: Tuple (height, width)
        
    Returns:
        dict: Dictionary dengan informasi limits
    """
    height, width = image_shape
    capacity = calculate_capacity(image_shape)
    
    return {
        'min_size': (64, 64),
        'current_size': (height, width),
        'capacity_bits': capacity,
        'capacity_chars_conservative': capacity // 16,
        'capacity_chars_optimal': capacity // 12,
        'recommended_max_chars': min(100, capacity // 16)
    }

def get_text_limits(image_shape):
    """
    Dapatkan batas teks yang dapat di-watermark.
    
    Args:
        image_shape: Tuple (height, width)
        
    Returns:
        dict: Dictionary dengan informasi text limits
    """
    limits = get_image_limits(image_shape)
    
    return {
        'max_chars_safe': limits['capacity_chars_conservative'],
        'max_chars_optimal': limits['capacity_chars_optimal'],
        'recommended_max': limits['recommended_max_chars'],
        'capacity_bits': limits['capacity_bits']
    }

def validate_inputs(img, watermark_text):
    """
    Validasi input untuk RDWT watermarking.
    
    Args:
        img: Input image
        watermark_text: Watermark text string
        
    Returns:
        tuple: (errors_list, warnings_list)
    """
    errors = []
    warnings = []
    
    # Validasi gambar
    if img is None:
        errors.append("Gambar tidak valid")
        return errors, warnings
    
    if len(img.shape) != 2:
        errors.append("Gambar harus grayscale")
        return errors, warnings
    
    height, width = img.shape
    
    if height < 64 or width < 64:
        errors.append("Ukuran gambar terlalu kecil (minimum 64x64)")
    
    if height < 128 or width < 128:
        warnings.append("Ukuran gambar kecil, hasil mungkin tidak optimal")
    
    # Validasi teks
    if not watermark_text or not watermark_text.strip():
        errors.append("Teks watermark tidak boleh kosong")
        return errors, warnings
    
    text_limits = get_text_limits(img.shape)
    text_length = len(watermark_text)
    
    if text_length > text_limits['max_chars_optimal']:
        if text_length > text_limits['max_chars_safe']:
            errors.append(f"Teks terlalu panjang. Maksimal: {text_limits['max_chars_safe']} karakter")
        else:
            warnings.append(f"Teks mendekati batas maksimal. Disarankan: {text_limits['recommended_max']} karakter")
    
    # Validasi karakter
    try:
        watermark_text.encode('utf-8')
    except UnicodeEncodeError:
        errors.append("Teks mengandung karakter yang tidak dapat di-encode")
    
    return errors, warnings

def get_optimal_parameters(img, watermark_length):
    """
    Dapatkan parameter optimal untuk gambar dan watermark tertentu.
    
    Args:
        img: Input grayscale image
        watermark_length: Length of watermark in bits
        
    Returns:
        dict: Optimal parameters
    """
    return rdwt_watermarking.select_optimal_parameters(img, watermark_length)

def analyze_image(img):
    """
    Analisis karakteristik gambar.
    
    Args:
        img: Input grayscale image
        
    Returns:
        dict: Analysis results
    """
    return rdwt_watermarking.analyze_image_characteristics(img)

# ===============================
# TESTING FUNCTIONS
# ===============================

def test_rdwt_watermarking(test_img_path=None):
    """
    Test fungsi RDWT watermarking dengan gambar test.
    """
    if test_img_path is None:
        # Buat test image dengan pattern yang lebih realistis
        x = np.linspace(0, 4*np.pi, 256)
        y = np.linspace(0, 4*np.pi, 256)
        X, Y = np.meshgrid(x, y)
        test_img = (128 + 50 * np.sin(X) * np.cos(Y) + 30 * np.random.randn(256, 256)).astype(np.uint8)
        test_img = np.clip(test_img, 0, 255)
        print("Using generated test image (256x256)")
    else:
        test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
        if test_img is None:
            print(f"Cannot read image: {test_img_path}")
            return
        print(f"Using test image: {test_img_path}")
    
    # Test watermark
    test_text = "Hello RDWT!"
    print(f"Test text: '{test_text}'")
    
    # Convert text to bits (simplified)
    test_bits = []
    for char in test_text:
        char_bits = [int(b) for b in format(ord(char), '08b')]
        test_bits.extend(char_bits)
    
    print(f"Watermark length: {len(test_bits)} bits")
    print(f"Image shape: {test_img.shape}")
    
    # Check capacity
    capacity = calculate_capacity(test_img.shape)
    print(f"Estimated capacity: {capacity} bits")
    
    if len(test_bits) > capacity:
        print(f"Warning: Watermark length ({len(test_bits)}) exceeds capacity ({capacity})")
        test_bits = test_bits[:capacity]
        print(f"Truncated to {len(test_bits)} bits")
    
    try:
        # Test embedding
        print("\n=== Testing Embedding ===")
        watermarked_img, overhead_data = embed_watermark_rdwt(test_img, test_bits)
        print("Embedding successful!")
        print(f"Transform type used: {overhead_data.get('transform_type', 'unknown')}")
        
        # Calculate PSNR
        psnr = calculate_psnr(test_img, watermarked_img)
        print(f"PSNR: {psnr:.2f} dB")
        
        # Test extraction
        print("\n=== Testing Extraction ===")
        extracted_bits = extract_watermark_rdwt(watermarked_img, overhead_data)
        print("Extraction successful!")
        
        # Convert extracted bits back to text
        extracted_chars = []
        for i in range(0, len(extracted_bits), 8):
            if i + 8 <= len(extracted_bits):
                byte_bits = extracted_bits[i:i+8]
                char_code = sum(bit << (7-j) for j, bit in enumerate(byte_bits))
                if 32 <= char_code <= 126:  # Printable ASCII range
                    extracted_chars.append(chr(char_code))
                else:
                    extracted_chars.append('?')
        
        extracted_text = ''.join(extracted_chars)
        print(f"Extracted text: '{extracted_text}'")
        
        # Calculate accuracy
        bit_accuracy = sum(1 for i, bit in enumerate(extracted_bits) 
                          if i < len(test_bits) and bit == test_bits[i]) / len(test_bits)
        print(f"Bit accuracy: {bit_accuracy:.3f} ({bit_accuracy*100:.1f}%)")
        
        # Test recovery
        print("\n=== Testing Recovery ===")
        recovered_img = recover_image_rdwt(watermarked_img, overhead_data)
        print("Recovery successful!")
        
        recovery_psnr = calculate_psnr(test_img, recovered_img)
        print(f"Recovery PSNR: {recovery_psnr:.2f} dB")
        
        # Summary
        print("\n=== Test Summary ===")
        print(f"Original image shape: {test_img.shape}")
        print(f"Watermark: '{test_text}' -> '{extracted_text}'")
        print(f"Parameters used: {overhead_data['wavelet']}, {overhead_data['levels']} levels, {overhead_data['subband']} subband")
        print(f"Embedding strength: {overhead_data['strength']}")
        print(f"Watermarking PSNR: {psnr:.2f} dB")
        print(f"Recovery PSNR: {recovery_psnr:.2f} dB")
        print(f"Bit accuracy: {bit_accuracy*100:.1f}%")
        
        # Analysis info
        if 'analysis' in overhead_data:
            analysis = overhead_data['analysis']
            print(f"Image type: {analysis.get('image_type', 'unknown')}")
            print(f"Texture complexity: {analysis.get('texture_complexity', 0):.2f}")
        
        return {
            'success': True,
            'watermarked_img': watermarked_img,
            'recovered_img': recovered_img,
            'overhead_data': overhead_data,
            'psnr': psnr,
            'recovery_psnr': recovery_psnr,
            'bit_accuracy': bit_accuracy,
            'extracted_text': extracted_text
        }
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def benchmark_rdwt(test_sizes=[(128, 128), (256, 256), (512, 512)]):
    """
    Benchmark RDWT watermarking dengan berbagai ukuran gambar.
    """
    print("=== RDWT Watermarking Benchmark ===\n")
    
    results = []
    test_text = "Test watermark for benchmarking"
    
    for height, width in test_sizes:
        print(f"Testing with image size: {height}x{width}")
        
        # Generate test image
        x = np.linspace(0, 4*np.pi, width)
        y = np.linspace(0, 4*np.pi, height)
        X, Y = np.meshgrid(x, y)
        test_img = (128 + 50 * np.sin(X) * np.cos(Y) + 20 * np.random.randn(height, width)).astype(np.uint8)
        test_img = np.clip(test_img, 0, 255)
        
        # Convert text to bits
        test_bits = []
        for char in test_text:
            char_bits = [int(b) for b in format(ord(char), '08b')]
            test_bits.extend(char_bits)
        
        # Check capacity
        capacity = calculate_capacity((height, width))
        if len(test_bits) > capacity:
            test_bits = test_bits[:capacity]
        
        try:
            import time
            
            # Time embedding
            start_time = time.time()
            watermarked_img, overhead_data = embed_watermark_rdwt(test_img, test_bits)
            embed_time = time.time() - start_time
            
            # Time extraction
            start_time = time.time()
            extracted_bits = extract_watermark_rdwt(watermarked_img, overhead_data)
            extract_time = time.time() - start_time
            
            # Calculate metrics
            psnr = calculate_psnr(test_img, watermarked_img)
            bit_accuracy = sum(1 for i, bit in enumerate(extracted_bits) 
                              if i < len(test_bits) and bit == test_bits[i]) / len(test_bits)
            
            result = {
                'size': f"{height}x{width}",
                'capacity': capacity,
                'watermark_bits': len(test_bits),
                'psnr': psnr,
                'bit_accuracy': bit_accuracy,
                'embed_time': embed_time,
                'extract_time': extract_time,
                'wavelet': overhead_data['wavelet'],
                'subband': overhead_data['subband'],
                'strength': overhead_data['strength']
            }
            
            results.append(result)
            
            print(f"  Capacity: {capacity} bits")
            print(f"  PSNR: {psnr:.2f} dB")
            print(f"  Accuracy: {bit_accuracy*100:.1f}%")
            print(f"  Embed time: {embed_time:.3f}s")
            print(f"  Extract time: {extract_time:.3f}s")
            print(f"  Parameters: {overhead_data['wavelet']}, {overhead_data['subband']}, strength={overhead_data['strength']:.3f}")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            print()
    
    # Summary
    if results:
        print("=== Benchmark Summary ===")
        print("Size\t\tCapacity\tPSNR\t\tAccuracy\tEmbed(s)\tExtract(s)")
        print("-" * 80)
        for r in results:
            print(f"{r['size']}\t\t{r['capacity']}\t\t{r['psnr']:.1f}\t\t{r['bit_accuracy']*100:.1f}%\t\t{r['embed_time']:.3f}\t\t{r['extract_time']:.3f}")
    
    return results

def demo_rdwt_with_text(img, text, show_steps=True):
    """
    Demo lengkap RDWT watermarking dengan text.
    
    Args:
        img: Input grayscale image
        text: Text to watermark
        show_steps: Whether to show detailed steps
        
    Returns:
        dict: Demo results
    """
    if show_steps:
        print("=== RDWT Watermarking Demo ===")
        print(f"Input text: '{text}'")
        print(f"Image shape: {img.shape}")
    
    # Validate inputs
    errors, warnings = validate_inputs(img, text)
    
    if errors:
        print("Errors found:")
        for error in errors:
            print(f"  - {error}")
        return {'success': False, 'errors': errors}
    
    if warnings and show_steps:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Convert text to bits
    text_bits = []
    for char in text:
        char_bits = [int(b) for b in format(ord(char), '08b')]
        text_bits.extend(char_bits)
    
    if show_steps:
        print(f"Text converted to {len(text_bits)} bits")
    
    try:
        # Get optimal parameters
        params = get_optimal_parameters(img, len(text_bits))
        
        if show_steps:
            print(f"Optimal parameters selected:")
            print(f"  - Wavelet: {params['wavelet']}")
            print(f"  - Levels: {params['levels']}")
            print(f"  - Subband: {params['subband']}")
            print(f"  - Strength: {params['strength']:.3f}")
            print(f"  - Image type: {params['analysis']['image_type']}")
        
        # Embed watermark
        watermarked_img, overhead_data = embed_watermark_rdwt(img, text_bits, params)
        
        # Calculate quality
        psnr = calculate_psnr(img, watermarked_img)
        
        # Extract watermark
        extracted_bits = extract_watermark_rdwt(watermarked_img, overhead_data)
        
        # Convert back to text
        extracted_chars = []
        for i in range(0, len(extracted_bits), 8):
            if i + 8 <= len(extracted_bits):
                byte_bits = extracted_bits[i:i+8]
                char_code = sum(bit << (7-j) for j, bit in enumerate(byte_bits))
                if 32 <= char_code <= 126:  # Printable ASCII
                    extracted_chars.append(chr(char_code))
                else:
                    extracted_chars.append('?')
        
        extracted_text = ''.join(extracted_chars)
        
        # Calculate accuracy
        bit_accuracy = sum(1 for i, bit in enumerate(extracted_bits) 
                          if i < len(text_bits) and bit == text_bits[i]) / len(text_bits)
        
        # Recover original
        recovered_img = recover_image_rdwt(watermarked_img, overhead_data)
        recovery_psnr = calculate_psnr(img, recovered_img)
        
        if show_steps:
            print(f"\nResults:")
            print(f"  - PSNR: {psnr:.2f} dB")
            print(f"  - Extracted text: '{extracted_text}'")
            print(f"  - Bit accuracy: {bit_accuracy*100:.1f}%")
            print(f"  - Recovery PSNR: {recovery_psnr:.2f} dB")
        
        return {
            'success': True,
            'original_img': img,
            'watermarked_img': watermarked_img,
            'recovered_img': recovered_img,
            'original_text': text,
            'extracted_text': extracted_text,
            'psnr': psnr,
            'recovery_psnr': recovery_psnr,
            'bit_accuracy': bit_accuracy,
            'overhead_data': overhead_data,
            'parameters': params
        }
        
    except Exception as e:
        if show_steps:
            print(f"Demo failed: {e}")
        return {'success': False, 'error': str(e)}

def save_overhead_data(overhead_data, filename):
    """
    Simpan overhead data ke file JSON dengan konversi aman dari semua tipe numpy ke tipe Python native.
    """
    import json
    import numpy as np

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.uint8, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, tuple):
            return [convert(i) for i in obj]
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        else:
            return obj

    # Konversi semua data ke format yang JSON serializable
    serializable_data = convert(overhead_data)

    try:
        # ✅ PERBAIKAN: Gunakan serializable_data, bukan overhead_data
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print(f"[✓] Overhead data saved to {filename}")
    except TypeError as e:
        print(f"[✗] Failed to save overhead data: {e}")
        print("Debug serializable_data contents:")
        for k, v in serializable_data.items():
            print(f"  {k}: {type(v)} = {v}")
        raise e
        
def load_overhead_data(filename):
    """
    Muat overhead data dari file JSON dan ubah 'positions' kembali ke tuple.
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    # Pastikan positions dikonversi kembali ke tuple
    if 'positions' in data:
        data['positions'] = [tuple(pos) for pos in data['positions']]

    print(f"[✓] Overhead data loaded from {filename}")
    return data


# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    print("RDWT Watermarking System - Complete Implementation")
    print("=" * 50)
    
    # Run basic test
    print("\n1. Running basic functionality test...")
    test_result = test_rdwt_watermarking()
    
    if test_result['success']:
        print("✓ Basic test passed!")
    else:
        print("✗ Basic test failed!")
        print(f"Error: {test_result.get('error', 'Unknown error')}")
    
    # Run benchmark
    print("\n2. Running benchmark...")
    benchmark_results = benchmark_rdwt()
    
    # Demo with custom text
    print("\n3. Running text demo...")
    
    # Create a more complex test image
    test_img = np.zeros((256, 256), dtype=np.uint8)
    
    # Add various patterns
    x = np.linspace(0, 2*np.pi, 256)
    y = np.linspace(0, 2*np.pi, 256)
    X, Y = np.meshgrid(x, y)
    
    # Sinusoidal pattern
    pattern1 = 80 * np.sin(3*X) * np.cos(2*Y)
    
    # Circular patterns
    center_x, center_y = 128, 128
    distances = np.sqrt((np.arange(256)[:, None] - center_x)**2 + 
                       (np.arange(256) - center_y)**2)
    pattern2 = 40 * np.sin(distances / 10)
    
    # Combine patterns
    test_img = (128 + pattern1 + pattern2 + 10 * np.random.randn(256, 256)).astype(np.uint8)
    test_img = np.clip(test_img, 0, 255)
    
    demo_result = demo_rdwt_with_text(test_img, "RDWT Watermarking Demo!")
    
    if demo_result['success']:
        print("✓ Text demo passed!")
        print(f"Original: '{demo_result['original_text']}'")
        print(f"Extracted: '{demo_result['extracted_text']}'")
        print(f"Accuracy: {demo_result['bit_accuracy']*100:.1f}%")
    else:
        print("✗ Text demo failed!")
        
    print("\n" + "=" * 50)
    print("RDWT Watermarking System Ready!")
    print("Available functions:")
    print("- embed_watermark_rdwt(img, watermark_bits, params=None)")
    print("- extract_watermark_rdwt(watermarked_img, overhead_data)")
    print("- recover_image_rdwt(watermarked_img, overhead_data)")
    print("- demo_rdwt_with_text(img, text, show_steps=True)")
    print("- calculate_psnr(img1, img2)")
    print("- validate_inputs(img, watermark_text)")
    print("- get_optimal_parameters(img, watermark_length)")
    print("- analyze_image(img)")
        