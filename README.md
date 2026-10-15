<div align="center">

# EMOTION RECOGNITION FROM SPEECH

**teaching a model to hear how you feel**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](#source-code)
[![Librosa](https://img.shields.io/badge/Audio-librosa-orange.svg)](https://librosa.org)
[![Keras](https://img.shields.io/badge/Deep_Learning-Keras-D00000?logo=keras&logoColor=white)](https://keras.io)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-RAVDESS-blueviolet.svg)](https://zenodo.org/record/1188976)

*Turns raw speech recordings into MFCC features that a **CNN-LSTM** model can learn
from, so it can tell whether a speaker sounds happy, sad, angry and more.
Built on the RAVDESS dataset as part of the **CodeAlpha** internship.*

*Project by [shahi1227](https://github.com/shahi1227). README by [Abdul Latheef J](https://github.com/Abdullatheef01).*

[How it works](#how-it-works) · [Run it](#run-it) · [Configuration](#configuration) · [Source code](#source-code) · [Limitations](#limitations-honestly) · [Roadmap](#roadmap)

</div>

---

## How it works

| # | Step | What happens |
|---|------|--------------|
| 1 | Load | Each `.wav` file is loaded with `librosa` at a fixed sample rate of 22,050 Hz. |
| 2 | Label | The emotion is read from the RAVDESS file name (the third number in the name). |
| 3 | Extract | 40 MFCC coefficients are computed per audio frame and transposed to `(Time, Features)`. |
| 4 | Pad / truncate | Every clip is brought to exactly 150 frames: short clips are zero-padded, long clips are cut. |
| 5 | Scale | Features are standardised with `StandardScaler` (zero mean, unit variance). |
| 6 | Encode | Emotion names become integers with `LabelEncoder`, then one-hot vectors. |
| 7 | Split | 80% train, 20% test, stratified so every emotion appears in both sets. |
| 8 | Reshape | A channel dimension is added so the data fits a CNN-LSTM input. |

> MFCCs (Mel-Frequency Cepstral Coefficients) describe the shape of the sound the way
> human hearing perceives it. They are a standard feature for speech and emotion tasks.

---

## Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song), speech
portion: 24 actors, 1,440 `.wav` files, each actor in its own `Actor_XX` folder.

| Code | Emotion |
|------|---------|
| 01 | Neutral |
| 02 | Calm |
| 03 | Happy |
| 04 | Sad |
| 05 | Angry |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

Download it from [Zenodo](https://zenodo.org/record/1188976) and extract
`Audio_Speech_Actors_01-24`.

---

## Repo layout

| Path | Contents |
|------|----------|
| `README.md` | This file, including the full script in the [Source code](#source-code) section. |

---

## Run it

### 1. Install the libraries

Needs Python 3.9 or newer.

```bash
pip install librosa numpy pandas scikit-learn keras tensorflow
```

### 2. Save the script

Copy the code from the [Source code](#source-code) section into a file named
`emotion_recognition.py`.

### 3. Get the dataset

Download RAVDESS speech audio and extract it somewhere on your machine.

### 4. Point the script at your data

Open `emotion_recognition.py` and update this line:

```python
RAVDESS_DIR = '/path/to/your/RAVDESS/Audio_Speech_Actors_01-24/'
```

On Windows, use a path like `C:/Users/you/RAVDESS/Audio_Speech_Actors_01-24/`.

### 5. Run it

```bash
python emotion_recognition.py
```

You should see output like this:

```
Total samples extracted: 1440
Feature matrix shape (Samples, Time, Features): (1440, 150, 40)
Number of unique emotions: 8
------------------------------
Final Training Data Shape: (1152, 150, 40, 1)
Final Testing Data Shape: (288, 150, 40, 1)
Output Label Shape: (1152, 8)
```

---

## Configuration

Set at the top of `emotion_recognition.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `RAVDESS_DIR` | none | Folder that contains the `Actor_XX` subfolders. Must be updated. |
| `N_MFCC` | `40` | Number of MFCC coefficients extracted per frame. |
| `MAX_PAD_LENGTH` | `150` | Fixed number of frames per clip (about 3.5 seconds at the default settings). |
| `TARGET_SR` | `22050` | Sample rate every file is resampled to. |

---

## Source code

<details>
<summary><b>emotion_recognition.py</b> (click to expand)</summary>

```python
# CodeAlpha_emotion_recognition_from_speech
import librosa
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
RAVDESS_DIR = '/path/to/your/RAVDESS/Audio_Speech_Actors_01-24/' # <<<< UPDATE THIS PATH
N_MFCC = 40      # Number of MFCC coefficients to extract
MAX_PAD_LENGTH = 150 # Fixed sequence length for the model (adjust based on your dataset analysis)
TARGET_SR = 22050  # Target sampling rate

# RAVDESS file names look like 03-01-06-01-02-01-12.wav
# The third number is the emotion code.
RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised',
}


def get_emotion_label_from_ravdess_filename(filename):
    """Returns the emotion name encoded in a RAVDESS file name."""
    emotion_code = filename.split('-')[2]
    return RAVDESS_EMOTIONS[emotion_code]


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
```

</details>

---

## Limitations, honestly

- **Preprocessing only, for now.** The script prepares the data. The CNN-LSTM model
  itself is not trained in this file yet.
- **Scaler sees the test data.** `StandardScaler` is fitted on the whole dataset before
  the split, which leaks a little test information into training. Fitting it on the
  training set only is the cleaner way.
- **Fixed length.** Clips longer than 150 frames lose their ending, and shorter clips
  carry zero padding.
- **Acted speech.** RAVDESS has professional actors reading two sentences in English.
  A model trained on it may not work well on natural, spontaneous speech.
- **Slow first run.** Loading and converting 1,440 audio files takes a while. Saving
  the features to disk (for example with `np.save`) avoids repeating it.

---

## Roadmap

- [ ] Move the script into its own `.py` file
- [ ] Build and train the CNN-LSTM model
- [ ] Report accuracy, a confusion matrix and per-emotion results
- [ ] Fit the scaler on the training set only
- [ ] Cache extracted features to disk
- [ ] Try data augmentation (noise, pitch shift, time stretch)
- [ ] Predict the emotion of a single new audio file

---

## Credits

Dataset: Livingstone & Russo, *The Ryerson Audio-Visual Database of Emotional Speech
and Song (RAVDESS)*. Project by [shahi1227](https://github.com/shahi1227) for the
**CodeAlpha** internship. README written by
[Abdul Latheef J](https://github.com/Abdullatheef01).
