from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt, detrend
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
import io
import neurokit2 as nk
import warnings

app = Flask(__name__)

# Constants
MODEL_PATH = 'enhanced_ecg_classifier02.pkl'
SCALER_PATH = 'ecg_scaler02.pkl'              
TARGET_FS = 360                             
WINDOW_SIZE = 360                           
OVERLAP = 180                               

# Label mapping
LABEL_MAPPING = {
    0: 'Normal beat',
    1: 'Left bundle branch block beat',
    2: 'Right bundle branch block beat',
    3: 'Atrial premature beat',
    4: 'Premature ventricular contraction',
    5: 'Paced beat',
    6: 'Fusion of ventricular and normal beat',
    7: 'Fusion beat',
    8: 'Nodal (junctional) escape beat',
    9: 'Aberrated atrial premature beat',
    10: 'Nodal (junctional) premature beat',
    11: 'Ventricular escape beat'
}

# Feature names
FEATURE_NAMES = [
    # Time-domain features
    'Mean', 'Std', 'Median', 'Range', 'Q1', 'Q3',
    # Frequency-domain features
    'VLF_mean', 'VLF_std', 'LF_mean', 'LF_std', 'HF_mean', 'HF_std',
    # Wavelet features
    *[f'Wavelet_{i}_mean' for i in range(5)],
    *[f'Wavelet_{i}_std' for i in range(5)],
    *[f'Wavelet_{i}_median' for i in range(5)],
    # HRV features
    'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_MedianNN',
    'HRV_HTI', 'HRV_LF', 'HRV_HF', 'HRV_LFHF', 'HRV_TotalPower'
]

# Selected feature indices
SELECTED_FEATURE_INDICES = [0,1,2,3,5,6,7,13,16,17,20,22,23,24,25,26]

# Load model and scaler
try:
    rf_model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    # Load evaluation metrics if available
    try:
        with open('model_metrics.json', 'r') as f:
            model_metrics = json.load(f)
    except FileNotFoundError:
        model_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'class_metrics': {str(i): {'precision': 0.0} for i in range(len(LABEL_MAPPING))}
        }
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

# Configure warning suppression
warnings.filterwarnings("ignore", category=RuntimeWarning, module="neurokit2")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

def preprocess_signal(signal, fs=360):
    """Apply advanced preprocessing"""
    try:
        cleaned = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit")
    except:
        # Fallback to bandpass filter
        nyquist = 0.5 * fs
        low = 0.5 / nyquist
        high = 40 / nyquist
        b, a = butter(4, [low, high], btype='band')
        cleaned = filtfilt(b, a, signal)
    
    # Remove baseline wander
    baseline = nk.signal_filter(cleaned, sampling_rate=fs, lowcut=0.5, highcut=None, method='butterworth')
    return cleaned - baseline

def extract_all_features(segment, fs=360):
    """Extract all possible features"""
    features = []
    
    def safe_stats(data):
        if len(data) == 0:
            return 0, 0, 0, 0, 0, 0
        return (
            np.nanmean(data),
            np.nanstd(data),
            np.nanmedian(data),
            np.nanmax(data) - np.nanmin(data),
            np.nanpercentile(data, 25),
            np.nanpercentile(data, 75)
        )
    
    # Time-domain features
    mean, std, median, rng, q25, q75 = safe_stats(segment)
    features.extend([mean, std, median, rng, q25, q75])
    
    # Frequency-domain features
    try:
        fft = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), 1/fs)
        for (low, high) in [(0.5, 5), (5, 15), (15, 40)]:
            band = fft[(freqs >= low) & (freqs <= high)]
            b_mean, b_std, _, _, _, _ = safe_stats(band)
            features.extend([b_mean, b_std])
    except:
        features.extend([0]*6)
    
    # Wavelet features
    try:
        coeffs = pywt.wavedec(segment, 'db4', level=4)
        for coeff in coeffs:
            c_mean, c_std, c_med, _, _, _ = safe_stats(coeff)
            features.extend([c_mean, c_std, c_med])
    except:
        features.extend([0]*(3*5))
    
    # HRV features
    hrv_features = {
        'HRV_MeanNN': 0,
        'HRV_SDNN': 0,
        'HRV_RMSSD': 0,
        'HRV_pNN50': 0,
        'HRV_MedianNN': 0
    }
    
    try:
        cleaned = nk.ecg_clean(segment, sampling_rate=fs)
        rpeaks = nk.ecg_findpeaks(cleaned, sampling_rate=fs, method="kalidas2017")
        
        if len(rpeaks["ECG_R_Peaks"]) >= 4:
            hrv = nk.hrv_time(rpeaks, sampling_rate=fs)
            for key in hrv_features:
                val = hrv[key].iloc[0] if not np.isnan(hrv[key].iloc[0]) else 0
                hrv_features[key] = val
    except:
        pass
    
    features.extend(hrv_features.values())
    return np.array(features)

def extract_selected_features(segment, fs=360):
    """Extract only selected features"""
    all_features = extract_all_features(segment, fs)
    return all_features[SELECTED_FEATURE_INDICES]

@app.route('/')
def index():
    return render_template('index.html', 
                         model_metrics=model_metrics,
                         label_mapping=LABEL_MAPPING)

@app.route('/feature_analysis')
def feature_analysis():
    try:
        # 1. Load and validate metrics data
        try:
            with open('model_metrics.json', 'r') as f:
                metrics_data = json.load(f)
        except FileNotFoundError:
            return jsonify({'error': 'Metrics file not found'}), 404
        except json.JSONDecodeError as e:
            app.logger.error(f"Invalid JSON: {str(e)}")
            return jsonify({'error': 'Invalid metrics file format'}), 400

        # 2. Process feature importances
        features = []
        if 'feature_importances' in metrics_data:
            features = [
                {
                    'name': f"Feature {i}",
                    'importance': float(imp),
                    'type': (
                        "Time-Domain" if i < 6 else
                        "Frequency-Domain" if i < 12 else
                        "Wavelet" if i < 27 else
                        "HRV"
                    )
                }
                for i, (name, imp) in enumerate(metrics_data['feature_importances'].items())
                if isinstance(imp, (int, float))  # Only include numeric values
            ]
            features.sort(key=lambda x: x['importance'], reverse=True)

        # 3. Validate and clean class metrics
        class_metrics = {}
        if 'class_metrics' in metrics_data:
            for class_id, metrics in metrics_data['class_metrics'].items():
                try:
                    # Convert string keys to integers and validate structure
                    class_metrics[int(class_id)] = {
                        'f1': float(metrics.get('f1', 0)),
                        'precision': float(metrics.get('precision', 0)),
                        'recall': float(metrics.get('recall', 0)),
                        'roc_auc': float(metrics.get('roc_auc', 0))
                    }
                except (ValueError, TypeError):
                    app.logger.warning(f"Invalid metrics for class {class_id}")

        # 4. Verify required overall metrics
        required_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        overall_metrics = metrics_data.get('overall_metrics', {})
        if not all(m in overall_metrics for m in required_metrics):
            missing = [m for m in required_metrics if m not in overall_metrics]
            app.logger.error(f"Missing required metrics: {missing}")
            return jsonify({'error': f'Missing metrics: {", ".join(missing)}'}), 400

        # 5. Render template with cleaned data
        return render_template('feature_analysis.html',
                            features=features,
                            overall_metrics=overall_metrics,
                            class_metrics=class_metrics,
                            metadata=metrics_data.get('metadata', {}),
                            label_mapping=LABEL_MAPPING)
        
    except Exception as e:
        app.logger.error(f"Feature analysis error: {str(e)}")
        return jsonify({'error': f'Feature analysis error: {str(e)}'}), 500
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files accepted'}), 400

        content = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))
        
        required_cols = ['Lead_I', 'Lead_II']
        if not all(col in df.columns for col in required_cols):
            return jsonify({'error': f'Required columns: {required_cols}'}), 400

        # Preprocess signals and keep original for display
        leads = [preprocess_signal(df[col].values, TARGET_FS) for col in required_cols]
        original_signals = [df[col].values[:1000] for col in required_cols]  # Take first 1000 samples for display
        
        # Feature extraction
        features = []
        for i in range(0, len(leads[0]) - WINDOW_SIZE + 1, WINDOW_SIZE - OVERLAP):
            segment_features = []
            for lead in leads:
                segment = lead[i:i + WINDOW_SIZE]
                segment_features.extend(extract_selected_features(segment, TARGET_FS))
            features.append(segment_features)
        
        features = np.array(features)
        scaled_features = scaler.transform(features)
        
        predictions = rf_model.predict(scaled_features)
        probas = rf_model.predict_proba(scaled_features)
        
        REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
        precision_lookup = {
            int(class_id): model_metrics['class_metrics'][str(class_id)]['precision'] 
            for class_id in range(len(LABEL_MAPPING))
        }

        segment_results = []
        for i, (pred, proba) in enumerate(zip(predictions, probas)):
            try:
                class_label = LABEL_MAPPING[pred]
            except KeyError:
                class_label = f"Unknown({pred})"
    
            prob_dist = {}
            for cls in range(len(LABEL_MAPPING)):
                try:
                    label = LABEL_MAPPING[cls]
                    prob_dist[label] = float(proba[cls])
                except KeyError:
                    prob_dist[f"Unknown({cls})"] = float(proba[cls])
    
            segment_results.append({
                'segment': i,
                'prediction': class_label,
                'precision': precision_lookup.get(pred, 0.0),
                'probability_distribution': prob_dist
             })

        diagnosis_summary = {}
        for class_id, count in pd.Series(predictions).value_counts().items():
            try:
                label = LABEL_MAPPING[class_id]
            except KeyError:
                label = f"Unknown({class_id})"
    
            diagnosis_summary[label] = {
                'count': count,
                'percentage': round(count/len(predictions)*100, 1),
                'precision': precision_lookup.get(class_id, 0.0)
           }
        
        return jsonify({
            'diagnosis_summary': diagnosis_summary,
            'segment_details': segment_results,
            'model_metrics': model_metrics,
            'signal_quality': assess_signal_quality(leads[0]),
            'ecg_signals': {  # Include original signals for display
                'Lead_I': original_signals[0].tolist(),
                'Lead_II': original_signals[1].tolist()
            }
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

def assess_signal_quality(signal):
    """Basic signal quality assessment"""
    noise = signal - np.convolve(signal, np.ones(10)/10, mode='same')
    snr = 10 * np.log10(np.var(signal)/np.var(noise)) if np.var(noise) > 0 else 0
    
    return {
        'snr_db': float(snr),
        'quality': 'Excellent' if snr > 20 else 'Good' if snr > 10 else 'Poor'
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)