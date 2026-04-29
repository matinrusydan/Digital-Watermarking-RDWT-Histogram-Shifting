import numpy as np
import cv2

class HistogramShiftingWatermark:
    """Class for histogram shifting watermarking operations."""
    
    def __init__(self, block_size=64, strength=3, redundancy=3):
        """Initialize the watermark object with parameters."""
        self.block_size = block_size
        self.strength = strength
        self.redundancy = redundancy
    
    def find_peak_zero(self, histogram):
        """Find the peak and zero point in the histogram."""
        # Find the peak (maximum value)
        peak_bin = np.argmax(histogram)
        
        # Find the zero or minimum point after the peak
        zero_bin = peak_bin
        min_value = float('inf')
        
        # Search for minimum in range [peak_bin+1, 255]
        for i in range(peak_bin + 1, 256):
            if histogram[i] < min_value:
                min_value = histogram[i]
                zero_bin = i
        
        # If no good zero found, just use peak+1
        if zero_bin == peak_bin:
            zero_bin = min(peak_bin + 1, 255)
        
        return peak_bin, zero_bin

def embed_watermark(image, watermark_bits, strength=3, block_size=64, redundancy=3):
    """Embed watermark bits into image using histogram shifting.
    
    Args:
        image: Grayscale image as numpy array
        watermark_bits: List of binary bits (0s and 1s)
        strength: Embedding strength (1-10)
        block_size: Size of blocks to divide image into
        redundancy: How many blocks to use for each bit
        
    Returns:
        Tuple of (watermarked_image, overhead_data)
    """
    # Make a copy of the image
    watermarked_img = image.copy()
    rows, cols = image.shape
    
    # Calculate how many blocks we can fit
    blocks_x = cols // block_size
    blocks_y = rows // block_size
    total_blocks = blocks_x * blocks_y
    
    # Check if we have enough blocks for the watermark with redundancy
    required_blocks = len(watermark_bits) * redundancy
    if required_blocks > total_blocks:
        raise ValueError(f"Image too small for watermark. Need {required_blocks} blocks, have {total_blocks}")
    
    # Prepare overhead data to store information needed for extraction
    overhead_data = {
        'block_size': block_size,
        'redundancy': redundancy,
        'strength': strength,
        'wm_length': len(watermark_bits),
        'img_shape': image.shape,
        'blocks_data': []
    }
    
    # For each bit in the watermark
    block_index = 0
    for i, bit in enumerate(watermark_bits):
        block_overhead = []
        
        # Embed the same bit multiple times (redundancy)
        for r in range(redundancy):
            if block_index >= total_blocks:
                break
                
            # Calculate block position
            block_y = block_index // blocks_x
            block_x = block_index % blocks_x
            
            # Define block region
            y_start = block_y * block_size
            x_start = block_x * block_size
            y_end = min(y_start + block_size, rows)
            x_end = min(x_start + block_size, cols)
            
            # Extract block
            block = watermarked_img[y_start:y_end, x_start:x_end]
            
            # Calculate histogram
            hist = cv2.calcHist([block], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # Find suitable peak and zero points
            peak, zero = find_peak_zero(hist)
            
            # Store original average pixel value for this block
            original_mean = np.mean(block)
            
            # Store information about this block
            block_overhead.append({
                'idx': block_index,
                'position': [x_start, y_start],
                'peak': int(peak),
                'zero': int(zero),
                'original_mean': float(original_mean),
                'bit': int(bit)  # Store the bit value for debugging
            })
            
            # Modify the block based on the bit value
            if bit == 1:
                # For bit 1: Increase brightness around peak slightly
                mask = (block >= peak - 2) & (block <= peak + 2)
                block[mask] = np.clip(block[mask] + strength, 0, 255)
            else:
                # For bit 0: Decrease brightness around peak slightly
                mask = (block >= peak - 2) & (block <= peak + 2)
                block[mask] = np.clip(block[mask] - strength, 0, 255)
            
            block_index += 1
        
        overhead_data['blocks_data'].append(block_overhead)
    
    return watermarked_img, overhead_data

def extract_watermark(watermarked_img, overhead_data):
    """Extract watermark from watermarked image using overhead data.
    
    Args:
        watermarked_img: Watermarked grayscale image
        overhead_data: Overhead data stored during embedding
        
    Returns:
        List of extracted binary bits
    """
    # Extract parameters from overhead data
    block_size = overhead_data['block_size']
    redundancy = overhead_data['redundancy']
    strength = overhead_data['strength']
    wm_length = overhead_data['wm_length']
    blocks_data = overhead_data['blocks_data']
    
    # Extract watermark bits
    extracted_bits = []
    
    for i in range(wm_length):
        # Get blocks for this bit
        bit_blocks = blocks_data[i]
        bit_values = []
        
        # Process each block for this bit
        for block_info in bit_blocks:
            # Get block position
            x_start, y_start = block_info['position']
            peak = block_info['peak']
            
            # Extract block
            x_end = min(x_start + block_size, watermarked_img.shape[1])
            y_end = min(y_start + block_size, watermarked_img.shape[0])
            block = watermarked_img[y_start:y_end, x_start:x_end]
            
            # Calculate the current mean value of the block
            current_mean = np.mean(block)
            
            # Get the original mean (before embedding)
            original_mean = block_info.get('original_mean')
            
            if original_mean is not None:
                # If we have the original mean, compare with current mean
                # If current mean is higher, bit is 1, otherwise 0
                bit_value = 1 if current_mean > original_mean else 0
            else:
                # Fallback method: analyze pixels around peak area
                # Calculate modified histogram
                hist = cv2.calcHist([block], [0], None, [256], [0, 256])
                hist = hist.flatten()
                
                # Check pixels around peak area
                peak_area = np.sum((block >= peak - 2) & (block <= peak + 2))
                non_peak_area = block.size - peak_area
                
                # Check if peak area has higher average than rest of block
                peak_mask = (block >= peak - 2) & (block <= peak + 2)
                if np.any(peak_mask):
                    peak_avg = np.mean(block[peak_mask])
                    non_peak_mask = ~peak_mask
                    if np.any(non_peak_mask):
                        non_peak_avg = np.mean(block[non_peak_mask])
                        # If peak area is brighter than non-peak area by a threshold,
                        # it's likely a 1 bit was embedded
                        bit_value = 1 if (peak_avg - non_peak_avg) > (strength / 2) else 0
                    else:
                        bit_value = 0
                else:
                    bit_value = 0
            
            bit_values.append(bit_value)
        
        # Use majority voting to determine the bit value
        bit = 1 if sum(bit_values) > len(bit_values) // 2 else 0
        extracted_bits.append(bit)
    
    return extracted_bits

def recover_image(watermarked_img, overhead_data):
    """Recover original image from watermarked image using overhead data.
    
    Args:
        watermarked_img: Watermarked grayscale image
        overhead_data: Overhead data stored during embedding
        
    Returns:
        Recovered original image
    """
    # Make a copy of the watermarked image
    recovered_img = watermarked_img.copy()
    
    # Extract parameters from overhead data
    block_size = overhead_data['block_size']
    strength = overhead_data['strength']
    blocks_data = overhead_data['blocks_data']
    
    # Process each embedded bit
    for bit_blocks in blocks_data:
        # Process each block for this bit
        for block_info in bit_blocks:
            # Get block position
            x_start, y_start = block_info['position']
            peak = block_info['peak']
            bit = block_info.get('bit', 0)  # Get the bit that was embedded
            
            # Extract block
            x_end = min(x_start + block_size, watermarked_img.shape[1])
            y_end = min(y_start + block_size, watermarked_img.shape[0])
            block = recovered_img[y_start:y_end, x_start:x_end]
            
            # Reverse the embedding operation
            peak_mask = (block >= peak - 2) & (block <= peak + 2)
            if bit == 1:
                # Undo bit 1 embedding: decrease brightness
                block[peak_mask] = np.clip(block[peak_mask] - strength, 0, 255)
            else:
                # Undo bit 0 embedding: increase brightness
                block[peak_mask] = np.clip(block[peak_mask] + strength, 0, 255)
    
    return recovered_img

def find_peak_zero(histogram):
    """Find the peak and zero point in the histogram."""
    # Find the peak (maximum value)
    peak_bin = np.argmax(histogram)
    
    # Find the zero or minimum point after the peak
    zero_bin = peak_bin
    min_value = float('inf')
    
    # Search for minimum in range [peak_bin+1, 255]
    for i in range(peak_bin + 1, 256):
        if histogram[i] < min_value:
            min_value = histogram[i]
            zero_bin = i
    
    # If no good zero found, just use peak+1
    if zero_bin == peak_bin:
        zero_bin = min(peak_bin + 1, 255)
    
    return peak_bin, zero_bin