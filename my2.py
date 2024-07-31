from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
import json
from flask_cors import CORS

# Initialize the Flask application
app = Flask(__name__)

# Allow CORS for a specific origin
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

# Configuration
UPLOAD_FOLDER = '/Users/kolanianupreddy/Desktop/music'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
import sklearn.preprocessing as skp
import random
seed = 12
np.random.seed(seed)
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

lookup_genre_name = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock"
}

df = pd.read_csv('/Users/kolanianupreddy/Desktop/Book1.csv')
columns_to_drop = ['filename','length']
df.drop(columns=columns_to_drop, inplace=True)

label_encoder = LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])

column_names = df.columns.tolist()
column_names.remove('Class')


# Normalize numerical features
scaler = StandardScaler()
df[column_names] = scaler.fit_transform(df[column_names])


x=df.drop('Class',axis=1)    
y=df['Class']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)
accuracies={}

from imblearn.over_sampling import BorderlineSMOTE

# Initialize BorderlineSMOTE with the best parameters
borderline_smote = BorderlineSMOTE(
    k_neighbors=3,
    kind='borderline-1',
    m_neighbors=5,
    random_state=42,
    sampling_strategy='auto'
)

# Perform resampling
x_resampled, y_resampled = borderline_smote.fit_resample(x_train, y_train)

x_train=x_resampled
y_train=y_resampled


# KNearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_neighbors': [5],  # Different values of k
    'p': [1, 2],  # Different values of p for Minkowski distance (1: Manhattan, 2: Euclidean)
    'weights': ['uniform', 'distance'],  # Weighting schemes
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithms for computing nearest neighbors
    'leaf_size': [20, 30, 40, 50],  # Different values for leaf size in trees
}

# Initialize KNN classifier
knn = KNeighborsClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Perform GridSearchCV
grid_search.fit(x_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_


# Get the best estimator
best_knn = grid_search.best_estimator_


def predict_genre(audio_features):
    # Extract the numerical values from the dictionary
    feature_values = np.array(list(audio_features.values()))
    
    # Transform the data using the scaler
    data1 = scaler.transform([feature_values])
    
    # Predict the genre using the KNN model
    genre_prediction = best_knn.predict(data1)
    
    # Lookup the genre name using the predicted genre index
    return lookup_genre_name[genre_prediction[0]]

# Function to extract audio features
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path)
    
    # Compute features
    features = {
        "chroma_stft_mean": np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        "chroma_stft_var": np.var(librosa.feature.chroma_stft(y=y, sr=sr)),
        "rms_mean": np.mean(librosa.feature.rms(y=y)),
        "rms_var": np.var(librosa.feature.rms(y=y)),
        "spectral_centroid_mean": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_centroid_var": np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_bandwidth_mean": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "spectral_bandwidth_var": np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "rolloff_mean": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "rolloff_var": np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "zero_crossing_rate_mean": np.mean(librosa.feature.zero_crossing_rate(y=y)),
        "zero_crossing_rate_var": np.var(librosa.feature.zero_crossing_rate(y=y)),
        "harmony_mean": np.mean(librosa.effects.harmonic(y)),
        "harmony_var": np.var(librosa.effects.harmonic(y)),
        "perceptr_mean": np.mean(librosa.effects.percussive(y)),
        "perceptr_var": np.var(librosa.effects.percussive(y)),
        "tempo": librosa.beat.tempo(y=y, sr=sr)[0]
    }
    
    # Compute MFCCs and their mean and variance
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])
        features[f"mfcc{i+1}_var"] = np.var(mfccs[i])

    return features

@app.route('/predict', methods=['POST'])
def predict():
    if 'wavFile' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['wavFile']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract audio features
        audio_features = extract_audio_features(file_path)

        # Simulated prediction logic (replace with actual logic)
        #predicted_genre = "Simulated Genre Prediction"  # Replace this with real prediction logic

        # Clean up the uploaded file if needed
        os.remove(file_path)

        # Include extracted features in the response if needed

        print(audio_features)
        predicted_genre = predict_genre(audio_features)

        # Predict the genre
    try:
        return jsonify({'predicted_genre': predicted_genre})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
