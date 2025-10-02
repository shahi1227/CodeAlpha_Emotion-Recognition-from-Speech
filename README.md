# CodeAlpha_ProjectName
import librosa
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
RAVDESS_DIR = '/path/to/your/RAVDESS/Audio_Speech_Actors_01-24/' # <<<< UPDATE THIS PATH
N_MFCC = 40      # Number of MFCC coefficients to extract
MAX_PAD_LENGTH = 150 # Fixed sequence length for the model (adjust based on your dataset analysis)
TARGET_SR = 22050  # Target sampling rate

# (Include the get_emotion_label_from_ravdess_filename function here)

def extract_padded_mfcc(file_path, n_mfcc=N_MFCC, max_len=MAX_PAD_LENGTH, sr=TARGET_SR):
    """Loads audio, extracts MFCCs, and pads/truncates to a fixed length."""
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Transpose MFCCs to be (Time, Features) for RNN/LSTM input
        mfccs = mfccs.T

        # Pad or Truncate the sequence to max_len
        if mfccs.shape[0] < max_len:
            # Pad with zeros at the end
            pad_width = max_len - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        else:
            # Truncate to max_len
            mfccs = mfccs[:max_len, :]

        return mfccs

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# --- MAIN FEATURE EXTRACTION LOOP ---
features_list = []
labels_list = []

# RAVDESS files are organized into Actor subdirectories (Actor_01, Actor_02, etc.)
for dirname, _, filenames in os.walk(RAVDESS_DIR):
    for filename in filenames:
        if filename.endswith('.wav'):
            file_path = os.path.join(dirname, filename)
            
            # 1. Extract Label
            emotion = get_emotion_label_from_ravdess_filename(filename)
            
            # 2. Extract Padded Features
            mfcc_sequence = extract_padded_mfcc(file_path)
            
            if mfcc_sequence is not None:
                features_list.append(mfcc_sequence)
                labels_list.append(emotion)

# 3. Convert to NumPy Arrays
X = np.array(features_list)
y = np.array(labels_list)

print(f"Total samples extracted: {X.shape[0]}")
print(f"Feature matrix shape (Samples, Time, Features): {X.shape}")
print(f"Number of unique emotions: {np.unique(y).shape[0]}")

# --- DATA PREPARATION FOR MODEL ---

# 4. Standardization (Scaling)
# Reshape X from 3D (samples, time, features) to 2D (samples * time, features) for the scaler
X_2D = X.reshape(-1, N_MFCC) 
scaler = StandardScaler()
X_2D_scaled = scaler.fit_transform(X_2D)

# Reshape back to 3D
X_scaled = X_2D_scaled.reshape(X.shape)

# 5. One-Hot Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 6. Final Split and Reshape for CNN-LSTM
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y
)

# Add the 'channel' dimension (1) for the CNN
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print("-" * 30)
print(f"Final Training Data Shape: {X_train.shape}")
print(f"Final Testing Data Shape: {X_test.shape}")
print(f"Output Label Shape: {y_train.shape}")
