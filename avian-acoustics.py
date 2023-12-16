import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.utils import shuffle

# Function definitions
def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

def remove_silence(audio, sr, top_db=20):
    _, idx = librosa.effects.trim(audio, top_db=top_db)
    return audio[idx[0]:idx[1]]

def standardize_length(audio, sr, duration=5):
    required_length = int(duration * sr)
    if len(audio) > required_length:
        return audio[:required_length]
    return np.pad(audio, (0, max(0, required_length - len(audio))), "constant")

def extract_additional_features(signal, sr):
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)
    return np.mean(spectral_rolloff), np.mean(zero_crossing_rate)

def extract_audio_features(audio_path, sr=44100, duration=5):
    signal, sr = librosa.load(audio_path, sr=sr, duration=duration)
    if len(signal) == 0:
        return None
    signal = normalize_audio(signal)
    signal = remove_silence(signal, sr)
    signal = standardize_length(signal, sr, duration)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr)
    mfcc_mean = np.mean(mfcc, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    mel_mean = np.mean(mel, axis=1)
    contrast_mean = np.mean(contrast, axis=1)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    spectral_rolloff, zero_crossing_rate = extract_additional_features(signal, sr)
    features = np.concatenate((mfcc_mean, chroma_mean, mel_mean, contrast_mean, tonnetz_mean, [spectral_rolloff, zero_crossing_rate]))
    return features

# Data Preparation
features = []
labels = []
data_dir = "data"  # Update this path as per your data directory

for species_folder in os.listdir(data_dir):
    species_path = os.path.join(data_dir, species_folder)
    if not os.path.isdir(species_path):
        continue
    for audio_file in os.listdir(species_path):
        if not audio_file.endswith('.mp3'):
            continue
        audio_path = os.path.join(species_path, audio_file)
        audio_features = extract_audio_features(audio_path)
        if audio_features is None:
            continue
        features.append(audio_features)
        labels.append(species_folder)

if not features:
    raise ValueError("No features extracted. Please check the audio files and paths.")

features = np.array(features)
labels = np.array(labels)
features, labels = shuffle(features, labels, random_state=42)
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Classifier Definition and Training
svm_classifier = SVC(random_state=42)
param_grid_svm = {
    'C': [0.1, 1, 10, 50, 100, 200],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}
grid_search_svm = GridSearchCV(svm_classifier, param_grid_svm, cv=5, n_jobs=-1, scoring='accuracy')
grid_search_svm.fit(X_train, y_train)
best_svm = grid_search_svm.best_estimator_

gb_classifier = GradientBoostingClassifier(random_state=42)
param_grid_gb = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.5, 1],
    'max_depth': [3, 5, 7]
}
grid_search_gb = GridSearchCV(gb_classifier, param_grid_gb, cv=5, n_jobs=-1, scoring='accuracy')
grid_search_gb.fit(X_train, y_train)
best_gb = grid_search_gb.best_estimator_

stacked_classifier = StackingClassifier(
    estimators=[('svm', best_svm), ('gb', best_gb)],
    final_estimator=GradientBoostingClassifier(random_state=42),
    cv=5,
    n_jobs=-1
)
stacked_classifier.fit(X_train, y_train)

# Model Evaluation
y_pred = stacked_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Stacked Model Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Visualizations

# PCA Visualization
pca = PCA(n_components=2)
features_reduced = pca.fit_transform(features)
plt.figure(figsize=(10, 8))
unique_labels = np.unique(labels)
for label in unique_labels:
    indices = labels == label
    plt.scatter(features_reduced[indices, 0], features_reduced[indices, 1], label=label, alpha=0.5)
plt.title('2D PCA of Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Confusion Matrix Visualization
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(labels))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Feature Distribution Visualization
plt.figure(figsize=(10, 6))
sns.histplot(features[:, 0], kde=True, color='green')
plt.title('Distribution of First Feature')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.show()

# Save the model
joblib.dump(stacked_classifier, "bird_song_model_stacked.pkl")
joblib.dump(scaler, "scaler.pkl")
