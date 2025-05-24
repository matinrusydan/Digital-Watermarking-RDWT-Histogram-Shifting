    // Global variables
let originalImage = null;
let watermarkedImage = null;
let overheadData = null;
let sessionId = null;

// Set a timeout to hide the loading overlay if OpenCV.js takes too long
const loadingTimeout = setTimeout(() => {
    document.getElementById('loadingOverlay').style.display = 'none';
    console.log('OpenCV.js loading timed out, continuing without it');
}, 10000); // 10 seconds timeout

// Wait for OpenCV.js to be loaded
function onOpenCvReady() {
    clearTimeout(loadingTimeout); // Clear the timeout if OpenCV loaded successfully
    document.getElementById('loadingOverlay').style.display = 'none';
    console.log('OpenCV.js is ready');
}

// Elements
const originalImageInput = document.getElementById('originalImageInput');
const originalImagePreview = document.getElementById('originalImagePreview');
const watermarkedImagePreview = document.getElementById('watermarkedImagePreview');
const binaryInput = document.getElementById('binaryInput');
const textInput = document.getElementById('textInput');
const strengthSlider = document.getElementById('strengthSlider');
const strengthValue = document.getElementById('strengthValue');
const blockSizeSelect = document.getElementById('blockSizeSelect');
const redundancySlider = document.getElementById('redundancySlider');
const redundancyValue = document.getElementById('redundancyValue');
const includeOverheadSwitch = document.getElementById('includeOverheadSwitch');
const embedButton = document.getElementById('embedButton');
const downloadButton = document.getElementById('downloadButton');
const statusMessage = document.getElementById('statusMessage');
const loadingOverlay = document.getElementById('loadingOverlay');
const metricsContainer = document.getElementById('metricsContainer');
const psnrValue = document.getElementById('psnrValue');
const wmSizeValue = document.getElementById('wmSizeValue');
// Tambahkan setelah deklarasi elemen yang sudah ada
const methodSelect = document.getElementById('methodSelect');
const rdwtParameters = document.getElementById('rdwtParameters');
const waveletSelect = document.getElementById('waveletSelect');
const levelsSlider = document.getElementById('levelsSlider');
const levelsValue = document.getElementById('levelsValue');
const subbandSelect = document.getElementById('subbandSelect');
const checkCapacityButton = document.getElementById('checkCapacityButton');
const capacityInfo = document.getElementById('capacityInfo');

// Tabs
const tabs = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');

// Event listeners
originalImageInput.addEventListener('change', handleImageUpload);
strengthSlider.addEventListener('input', updateStrengthValue);
redundancySlider.addEventListener('input', updateRedundancyValue);
embedButton.addEventListener('click', embedWatermark);
downloadButton.addEventListener('click', downloadWatermarkedImage);
textInput.addEventListener('input', updateBinaryFromText);
// Tambahkan setelah event listeners yang sudah ada
methodSelect.addEventListener('change', toggleMethodParameters);
levelsSlider.addEventListener('input', updateLevelsValue);
checkCapacityButton.addEventListener('click', checkRDWTCapacity);

// Tab switching
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const tabId = tab.getAttribute('data-tab');
        
        // Update active tab
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        // Update active content
        tabContents.forEach(content => content.classList.remove('active'));
        document.getElementById(`${tabId}-tab`).classList.add('active');
        
        // If switching to text tab, update binary from text
        if (tabId === 'text' && textInput.value) {
            updateBinaryFromText();
        }
    });
});

// Functions

function onOpenCvReady() {
   document.getElementById('loadingOverlay').style.display = 'none';
    console.log('OpenCV.js is ready');
}
function updateStrengthValue() {
    strengthValue.textContent = strengthSlider.value;
}

function updateRedundancyValue() {
    redundancyValue.textContent = redundancySlider.value;
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    
    // Create FormData to send the file
    const formData = new FormData();
    formData.append('image', file);
    
    // Send the image to the server API
    fetch('/api/upload_image', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || 'Failed to upload image');
            });
        }
        return response.json();
    })
    .then(data => {
        // Store session ID for future requests
        sessionId = data.session_id;
        
        // Display preview
        const img = document.createElement('img');
        img.src = data.image_data;
        originalImagePreview.innerHTML = '';
        originalImagePreview.appendChild(img);
        
        // Enable embed button
        embedButton.disabled = false;
        updateStatus('Image uploaded successfully!', 'success');
    })
    .catch(error => {
        updateStatus('Error: ' + error.message, 'error');
    })
    .finally(() => {
        loadingOverlay.style.display = 'none';
    });
}

function textToBinary(text) {
    let binary = '';
    for (let i = 0; i < text.length; i++) {
        const charBinary = text.charCodeAt(i).toString(2).padStart(8, '0');
        binary += charBinary + ' ';
    }
    return binary.trim();
}

function updateBinaryFromText() {
    const text = textInput.value;
    if (text) {
        binaryInput.value = textToBinary(text);
    }
}

function processBinaryInput(input) {
    // Remove all spaces and validate
    const cleanBinary = input.replace(/\s+/g, '');
    
    // Check if it's valid binary
    if (!/^[01]+$/.test(cleanBinary)) {
        throw new Error('Invalid binary input. Use only 0s and 1s.');
    }
    
    return cleanBinary;
}

function embedWatermark() {
    if (!sessionId) {
        updateStatus('Please upload an image first', 'error');
        return;
    }

    try {
        // Get watermark binary data
        const binaryData = processBinaryInput(binaryInput.value);
        if (!binaryData) {
            updateStatus('Please enter binary watermark data', 'error');
            return;
        }

        // Get parameters
        const strength = parseInt(strengthSlider.value);
        const blockSize = parseInt(blockSizeSelect.value);
        const redundancy = parseInt(redundancySlider.value);

        // Determine watermark type
        const isText = document.querySelector('.tab.active').getAttribute('data-tab') === 'text';
        const watermarkSource = isText ? textInput.value : binaryData;

        // Show loading overlay
        loadingOverlay.style.display = 'flex';

        // 🔧 Persiapkan data request termasuk opsi RDWT
        const requestData = {
            session_id: sessionId,
            watermark: watermarkSource,
            is_text: isText,
            strength: strength,
            block_size: blockSize,
            redundancy: redundancy,
            use_rdwt: methodSelect.value === 'rdwt'  // ✅ tambahkan flag RDWT
        };

        // ✅ Jika metode RDWT dipilih, tambahkan parameter spesifik RDWT
        if (methodSelect.value === 'rdwt') {
            requestData.wavelet = waveletSelect.value;
            requestData.levels = parseInt(levelsSlider.value);
            requestData.subband = subbandSelect.value;
        }

        // Kirim ke server
        fetch('/api/embed_watermark', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)  // ✅ gunakan variabel ini
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'Failed to embed watermark');
                });
            }
            return response.json();
        })
        .then(data => {
            // Tampilkan gambar hasil watermarking
            const img = document.createElement('img');
            img.src = data.image_data;
            img.style.maxWidth = '100%';
            img.style.maxHeight = '300px';
            watermarkedImagePreview.innerHTML = '';
            watermarkedImagePreview.appendChild(img);

            // Tampilkan metrik
            psnrValue.textContent = data.psnr.toFixed(2);
            wmSizeValue.textContent = data.wm_size;
            metricsContainer.style.display = 'flex';

            // Aktifkan tombol download
            downloadButton.disabled = false;

            updateStatus('Watermark embedded successfully!', 'success');
        })
        .catch(error => {
            updateStatus('Error: ' + error.message, 'error');
        })
        .finally(() => {
            loadingOverlay.style.display = 'none';
        });

    } catch (error) {
        updateStatus('Error: ' + error.message, 'error');
    }
}

function downloadWatermarkedImage() {
    if (!sessionId) {
        updateStatus('No watermarked image to download', 'error');
        return;
    }
    
    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    
    // Download watermarked image
    const imgLink = document.createElement('a');
    imgLink.href = `/api/download_watermarked_image?session_id=${sessionId}`;
    imgLink.download = 'watermarked_image.png';
    document.body.appendChild(imgLink);
    imgLink.click();
    document.body.removeChild(imgLink);
    
    // Download overhead data if requested
    if (includeOverheadSwitch.checked) {
        setTimeout(() => {
            const dataLink = document.createElement('a');
            dataLink.href = `/api/download_overhead_data?session_id=${sessionId}`;
            dataLink.download = 'watermark_overhead.json';
            document.body.appendChild(dataLink);
            dataLink.click();
            document.body.removeChild(dataLink);
        }, 500);
    }
    
    updateStatus('Files downloaded', 'success');
    loadingOverlay.style.display = 'none';
}

function updateStatus(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = 'status ' + type;
    
    // Hide the message after 5 seconds
    setTimeout(() => {
        statusMessage.className = 'status';
        statusMessage.textContent = '';
    }, 5000);
}

// Function to validate input and update UI state
function validateInputs() {
    // Enable embed button only if both session and watermark data are present
    if (sessionId && (binaryInput.value.trim() !== '' || textInput.value.trim() !== '')) {
        embedButton.disabled = false;
    } else {
        embedButton.disabled = true;
    }
}

// Function baru setelah RDWT
function toggleMethodParameters() {
    const isRDWT = methodSelect.value === 'rdwt';
    rdwtParameters.style.display = isRDWT ? 'block' : 'none';
    
    // Reset capacity info when switching methods
    capacityInfo.style.display = 'none';
    capacityInfo.innerHTML = '';
}

function updateLevelsValue() {
    levelsValue.textContent = levelsSlider.value;
}

function checkRDWTCapacity() {
    if (!sessionId) {
        updateStatus('Please upload an image first', 'error');
        return;
    }
    
    loadingOverlay.style.display = 'flex';
    
    fetch('/api/check_capacity', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: sessionId,
            wavelet: waveletSelect.value,
            levels: parseInt(levelsSlider.value)
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        capacityInfo.innerHTML = `
            <div class="capacity-result">
                <strong>Maximum Capacity:</strong> ${data.capacity} bits<br>
                <strong>Image Size:</strong> ${data.image_shape[1]} x ${data.image_shape[0]} pixels
            </div>
        `;
        capacityInfo.style.display = 'block';
    })
    .catch(error => {
        updateStatus('Error: ' + error.message, 'error');
    })
    .finally(() => {
        loadingOverlay.style.display = 'none';
    });
}
// JavaScript for Extract Watermark functionality
// Add this to your existing JavaScript or include as a separate file

// Elements for extraction functionality
const watermarkedImageExtractInput = document.getElementById('watermarkedImageExtractInput');
const watermarkedImageForExtraction = document.getElementById('watermarkedImageForExtraction');
const overheadDataInput = document.getElementById('overheadDataInput');
const extractButton = document.getElementById('extractButton');
const extractedBinary = document.getElementById('extractedBinary');
const extractedText = document.getElementById('extractedText');
const recoverButton = document.getElementById('recoverButton');
const recoveredImagePreview = document.getElementById('recoveredImagePreview');
const recoveredImagePanel = document.getElementById('recoveredImagePanel');
const downloadRecoveredButton = document.getElementById('downloadRecoveredButton');
const recoveredStatusMessage = document.getElementById('recoveredStatusMessage');

// Extract session ID and loaded flags
let extractSessionId = null;
let watermarkedImageLoaded = false;
let overheadDataLoaded = false;

// Event listeners for extraction functionality
watermarkedImageExtractInput.addEventListener('change', handleWatermarkedImageUpload);
overheadDataInput.addEventListener('change', handleOverheadDataUpload);
extractButton.addEventListener('click', extractWatermark);
recoverButton.addEventListener('click', recoverOriginalImage);
downloadRecoveredButton.addEventListener('click', downloadRecoveredImage);

// Tab functionality for extraction panel (reuse the existing tab code pattern)
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const tabId = tab.getAttribute('data-tab');
        
        // Find the parent panel
        const panel = tab.closest('.panel');
        
        // Update active tab in this panel only
        panel.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        // Update active content in this panel only
        panel.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        panel.querySelector(`#${tabId}-tab`).classList.add('active');
    });
});

// Functions for extraction functionality
function handleWatermarkedImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    
    // Create FormData to send the file
    const formData = new FormData();
    formData.append('image', file);
    
    // Send the image to the server API
    fetch('/api/upload_watermarked_image', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || 'Failed to upload image');
            });
        }
        return response.json();
    })
    .then(data => {
        // Store session ID for future requests
        extractSessionId = data.session_id;
        
        // Display preview
        const img = document.createElement('img');
        img.src = data.image_data;
        watermarkedImageForExtraction.innerHTML = '';
        watermarkedImageForExtraction.appendChild(img);
        
        // Update state
        watermarkedImageLoaded = true;
        updateExtractButtonState();
        
        updateExtractStatus('Watermarked image uploaded successfully!', 'success');
    })
    .catch(error => {
        updateExtractStatus('Error: ' + error.message, 'error');
    })
    .finally(() => {
        loadingOverlay.style.display = 'none';
    });
}

function handleOverheadDataUpload(event) {
    const file = event.target.files[0];
    if (!file || !extractSessionId) return;
    
    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    
    // Create FormData to send the file
    const formData = new FormData();
    formData.append('overhead', file);
    formData.append('session_id', extractSessionId);
    
    // Send the overhead data to the server API
    fetch('/api/upload_overhead', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || 'Failed to upload overhead data');
            });
        }
        return response.json();
    })
    .then(data => {
        // Update state
        overheadDataLoaded = true;
        updateExtractButtonState();
        
        updateExtractStatus('Overhead data uploaded successfully!', 'success');
    })
    .catch(error => {
        updateExtractStatus('Error: ' + error.message, 'error');
    })
    .finally(() => {
        loadingOverlay.style.display = 'none';
    });
}

function updateExtractButtonState() {
    // Enable extract button only if both watermarked image and overhead data are loaded
    extractButton.disabled = !(watermarkedImageLoaded && overheadDataLoaded);
}

function extractWatermark() {
    if (!extractSessionId || !watermarkedImageLoaded || !overheadDataLoaded) {
        updateExtractStatus('Please upload both watermarked image and overhead data', 'warning');
        return;
    }
    
    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    
    // Send request to extract watermark
    fetch('/api/extract_watermark', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: extractSessionId
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || 'Failed to extract watermark');
            });
        }
        return response.json();
    })
    .then(data => {
        // Display extracted binary data
        extractedBinary.value = data.binary || '';
        
        // Display extracted text if available
        if (data.text) {
            extractedText.value = data.text;
        } else {
            extractedText.value = 'No valid text data extracted';
        }
        
        // Enable recover button
        recoverButton.disabled = false;
        
        // Switch to extraction result tab
        const extractionResultTab = document.querySelector('[data-tab="extraction-result"]');
        extractionResultTab.click();
        
        updateExtractStatus('Watermark extracted successfully!', 'success');
    })
    .catch(error => {
        updateExtractStatus('Error: ' + error.message, 'error');
    })
    .finally(() => {
        loadingOverlay.style.display = 'none';
    });
}

function recoverOriginalImage() {
    if (!extractSessionId) {
        updateExtractStatus('Session expired or invalid', 'error');
        return;
    }
    
    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    
    // Send request to recover original image
    fetch('/api/recover_image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: extractSessionId
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || 'Failed to recover image');
            });
        }
        return response.json();
    })
    .then(data => {
        // Display recovered image
        const img = document.createElement('img');
        img.src = data.image_data;
        recoveredImagePreview.innerHTML = '';
        recoveredImagePreview.appendChild(img);
        
        // Show recovered image panel
        recoveredImagePanel.style.display = 'block';
        
        // Enable download button
        downloadRecoveredButton.disabled = false;
        
        updateRecoveredStatus('Image recovered successfully!', 'success');
    })
    .catch(error => {
        updateRecoveredStatus('Error: ' + error.message, 'error');
    })
    .finally(() => {
        loadingOverlay.style.display = 'none';
    });
}

function downloadRecoveredImage() {
    if (!extractSessionId) {
        updateRecoveredStatus('No recovered image to download', 'error');
        return;
    }
    
    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    
    // Download recovered image
    const imgLink = document.createElement('a');
    imgLink.href = `/api/download_recovered_image?session_id=${extractSessionId}`;
    imgLink.download = 'recovered_image.png';
    document.body.appendChild(imgLink);
    imgLink.click();
    document.body.removeChild(imgLink);
    
    updateRecoveredStatus('Recovered image downloaded', 'success');
    loadingOverlay.style.display = 'none';
}

function updateExtractStatus(message, type) {
    // Create or update status element
    let statusEl = document.querySelector('.extraction-status');
    if (!statusEl) {
        statusEl = document.createElement('div');
        statusEl.className = 'extraction-status';
        extractButton.parentNode.appendChild(statusEl);
    }
    
    statusEl.textContent = message;
    statusEl.className = 'extraction-status ' + type;
    
    // Hide the message after 5 seconds
    setTimeout(() => {
        statusEl.className = 'extraction-status';
        statusEl.textContent = '';
    }, 5000);
}

function updateRecoveredStatus(message, type) {
    recoveredStatusMessage.textContent = message;
    recoveredStatusMessage.className = 'status ' + type;
    
    // Hide the message after 5 seconds
    setTimeout(() => {
        recoveredStatusMessage.className = 'status';
        recoveredStatusMessage.textContent = '';
    }, 5000);
}


// Elements for watermark verification
const verifyWatermarkButton = document.getElementById('verifyWatermarkButton');
const verificationResultDiv = document.getElementById('verificationResult');
const verificationStatusMessage = document.getElementById('verificationStatusMessage');

// Add event listener for verification button
if (verifyWatermarkButton) {
    verifyWatermarkButton.addEventListener('click', verifyWatermark);
}

// Function to verify extracted watermark against original
function verifyWatermark() {
    if (!extractSessionId) {
        updateVerificationStatus('No active extraction session', 'error');
        return;
    }
    
    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    
    // Send request to verify watermark
    fetch('/api/verify_watermark', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: extractSessionId
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || 'Failed to verify watermark');
            });
        }
        return response.json();
    })
    .then(data => {
        // Create verification result HTML
        let resultHTML = `
            <div class="verification-container">
                <h3>Watermark Verification Result</h3>
                <div class="verification-status ${data.matched ? 'success' : 'error'}">
                    <strong>Match Status:</strong> ${data.matched ? 'MATCHED' : 'NOT MATCHED'}
                </div>
                <div class="verification-metric">
                    <strong>Match Rate:</strong> ${(data.match_rate * 100).toFixed(2)}%
                </div>
        `;
        
        // Add text comparison if available
        if (data.original_text && data.extracted_text) {
            resultHTML += `
                <div class="verification-details">
                    <div class="comparison">
                        <div class="comparison-item">
                            <strong>Original Text:</strong>
                            <div class="code-display">${escapeHTML(data.original_text)}</div>
                        </div>
                        <div class="comparison-item">
                            <strong>Extracted Text:</strong>
                            <div class="code-display">${escapeHTML(data.extracted_text)}</div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Add binary comparison if available
        if (data.original_bits || data.original_binary) {
            resultHTML += `
                <div class="verification-details">
                    <div class="comparison">
                        <div class="comparison-item">
                            <strong>Original Binary:</strong>
                            <div class="code-display binary">${data.original_bits || data.original_binary}</div>
                        </div>
                        <div class="comparison-item">
                            <strong>Extracted Binary:</strong>
                            <div class="code-display binary">${data.extracted_bits || data.extracted_binary}</div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        resultHTML += '</div>';
        
        // Display verification result
        verificationResultDiv.innerHTML = resultHTML;
        verificationResultDiv.style.display = 'block';
        
        updateVerificationStatus(data.matched ? 
            'Watermark verified successfully!' : 
            'Watermark verification failed - mismatch detected', 
            data.matched ? 'success' : 'warning');
    })
    .catch(error => {
        updateVerificationStatus('Error: ' + error.message, 'error');
    })
    .finally(() => {
        loadingOverlay.style.display = 'none';
    });
}

function updateVerificationStatus(message, type) {
    if (!verificationStatusMessage) return;
    
    verificationStatusMessage.textContent = message;
    verificationStatusMessage.className = 'status ' + type;
    
    // Hide the message after 5 seconds
    setTimeout(() => {
        verificationStatusMessage.className = 'status';
        verificationStatusMessage.textContent = '';
    }, 5000);
}

// Helper function to escape HTML
function escapeHTML(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// Add clear session functionality when switching operations
document.querySelectorAll('.panel-select').forEach(button => {
    button.addEventListener('click', function() {
        const selectedPanel = this.getAttribute('data-panel');
        
        // Clear active sessions if switching between operations
        if ((selectedPanel === 'embed-panel' && extractSessionId) || 
            (selectedPanel === 'extract-panel' && sessionId)) {
            clearCurrentSession();
        }
    });
});

// Function to clear current session data
function clearCurrentSession() {
    if (sessionId) {
        fetch('/api/clear_session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId
            })
        })
        .then(() => {
            sessionId = null;
            originalImagePreview.innerHTML = '';
            watermarkedImagePreview.innerHTML = '';
            embedButton.disabled = true;
            downloadButton.disabled = true;
            metricsContainer.style.display = 'none';
        })
        .catch(error => console.error('Error clearing session:', error));
    }
    
    if (extractSessionId) {
        fetch('/api/clear_session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: extractSessionId
            })
        })
        .then(() => {
            extractSessionId = null;
            watermarkedImageLoaded = false;
            overheadDataLoaded = false;
            watermarkedImageForExtraction.innerHTML = '';
            extractedBinary.value = '';
            extractedText.value = '';
            recoveredImagePreview.innerHTML = '';
            recoveredImagePanel.style.display = 'none';
            extractButton.disabled = true;
            recoverButton.disabled = true;
            downloadRecoveredButton.disabled = true;
            if (verificationResultDiv) {
                verificationResultDiv.innerHTML = '';
                verificationResultDiv.style.display = 'none';
            }
        })
        .catch(error => console.error('Error clearing extraction session:', error));
    }
}
// Add event listeners for input validation
binaryInput.addEventListener('input', validateInputs);
textInput.addEventListener('input', validateInputs);

// Initialize UI elements once page is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Update initial values
    updateStrengthValue();
    updateRedundancyValue();
    toggleMethodParameters();
    updateLevelsValue();
    // Check OpenCV status
    if (typeof cv !== 'undefined') {
        clearTimeout(loadingTimeout);
        document.getElementById('loadingOverlay').style.display = 'none';
        console.log('OpenCV.js loaded successfully');
    } else {
        console.log('Waiting for OpenCV.js to load...');
        // The overlay will be hidden by onOpenCvReady callback or the timeout
    }
});