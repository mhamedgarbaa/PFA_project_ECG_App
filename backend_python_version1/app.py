from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt, detrend
from joblib import load
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Constants
MODEL_PATH = 'random_forest_model.pkl'  # Path to the pre-trained Random Forest model
SCALER_PATH = 'scaler.pkl'              # Path to the scaler used during training
TARGET_FS = 360                         # Sampling frequency of the ECG signal
WINDOW_SIZE = 180                       # Window size for feature extraction
OVERLAP = 90                            # Overlap between windows for feature extraction

# Label mapping for classification results
LABEL_MAPPING = {
    0: 'Normal beat',
    1: 'Left bundle branch block beat',
    2: 'Right bundle branch block beat',
    3: 'Atrial premature beat',
    4: 'Premature ventricular contraction'
}

# Load pre-trained model and scaler
rf_model = load(MODEL_PATH)
scaler = load(SCALER_PATH)

def preprocess_signal(signal, fs=360):
    """Apply bandpass filter and detrend the signal."""
    def bandpass_filter(signal, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)
    filtered_signal = bandpass_filter(signal, lowcut=0.5, highcut=40, fs=fs)
    detrended_signal = detrend(filtered_signal)
    return detrended_signal

def extract_wavelet_features(segment, level=4):
    """Extract wavelet features from a segment."""
    coeffs = pywt.wavedec(segment, 'db4', level=level)
    std_coeffs = [np.std(c) for c in coeffs]
    energy_coeffs = [np.sum(c ** 2) for c in coeffs]
    
    # Calculate entropy safely
    entropy_coeffs = []
    for c in coeffs:
        # Normalize coefficients
        norm_c = c / (np.sum(np.abs(c)) + 1e-10)
        # Clip values to avoid invalid log2 computations
        clipped_norm_c = np.clip(norm_c, 1e-10, None)
        # Compute entropy
        entropy = -np.sum(clipped_norm_c * np.log2(clipped_norm_c))
        entropy_coeffs.append(entropy)
    
    return np.concatenate([std_coeffs, energy_coeffs, entropy_coeffs])

def extract_multilead_features(leads, window_size=180, overlap=90, fs=360):
    """Extract features from multiple leads."""
    step_size = window_size - overlap
    features = []
    for i in range(0, len(leads[0]) - window_size + 1, step_size):
        segment_features = []
        for lead in leads:
            segment = lead[i:i + window_size]
            wavelet_features = extract_wavelet_features(segment)
            time_features = [np.mean(segment), np.std(segment)]
            freq_features = [np.mean(np.abs(np.fft.fft(segment)))]
            combined_features = np.concatenate([wavelet_features, time_features, freq_features])
            segment_features.extend(combined_features)
        features.append(segment_features)
    return np.array(features)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty file submitted'}), 400

        # Read the CSV file
        df = pd.read_csv(file)
        if 'Lead_I' not in df.columns or 'Lead_II' not in df.columns:
            return jsonify({'error': 'CSV must contain Lead_I and Lead_II columns'}), 400

        # Preprocess the signals
        lead_i = preprocess_signal(df['Lead_I'].values, fs=TARGET_FS)
        lead_ii = preprocess_signal(df['Lead_II'].values, fs=TARGET_FS)

        # Extract features
        features = extract_multilead_features([lead_i, lead_ii], window_size=WINDOW_SIZE, overlap=OVERLAP, fs=TARGET_FS)

        # Scale features
        scaled_features = scaler.transform(features)

        # Predict using the model
        predictions = rf_model.predict(scaled_features)
        prediction_counts = np.bincount(predictions)
        most_common_prediction = np.argmax(prediction_counts)

        # Prepare feature names and values
        feature_names = [
            *[f"Wavelet Std {i}" for i in range(5)],
            *[f"Wavelet Energy {i}" for i in range(5)],
            *[f"Wavelet Entropy {i}" for i in range(5)],
            "Mean", "Std", "FFT Magnitude"
        ] * 2  # Repeat for both leads
        feature_values = scaled_features[0].tolist()  # Example: first segment's features

        # Return the result
        result = {
    'prediction': LABEL_MAPPING.get(most_common_prediction, 'Unknown'),
    'features': dict(zip(feature_names, feature_values)),
    'lead_i_signal': lead_i.tolist(),  # Add preprocessed Lead I signal
    'lead_ii_signal': lead_ii.tolist(),  # Add preprocessed Lead II signal
    'sampling_rate': TARGET_FS
}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)