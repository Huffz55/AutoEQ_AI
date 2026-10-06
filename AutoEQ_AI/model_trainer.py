import os
import numpy as np
import librosa
import tensorflow as tf
import tf2onnx
from sklearn.model_selection import train_test_split

# --- GPU MEMORY GROWTH AYARI (RTX 4060 8GB VRAM İÇİN) ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # GPU belleğinin tamamını bloke etmek yerine sadece gerektiği kadar kullan
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} adet GPU bulundu ve Memory Growth (Esnek Bellek) aktif edildi.")
    except RuntimeError as e:
        print(f"GPU bellek ayarı sırasında hata: {e}")
else:
    print("Sistemde native GPU bulunamadı (Windows üzerinde TF 2.11+ GPU için WSL2 veya DirectML gerektirebilir). CPU ile devam ediliyor...")


DATASET_PATH = "dataset"
GENRES = ["Trap", "Progressive Metal", "Classical"] 
NUM_CLASSES = len(GENRES)
SAMPLE_RATE = 48000
CHUNK_LENGTH = 2 * SAMPLE_RATE

# Ses resmi (Mel-Spectrogram) boyutları: (128 Mel Bandı, 128 Zaman Adımı, 1 Kanal Siyah/Beyaz)
MEL_BANDS = 128
TIME_STEPS = 128
CHANNELS = 1

# --- GÜVENLİ VE TÖRE ÖZEL DATA AUGMENTATION ---

def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def pitch_shift(data, sr, n_steps=2):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def volume_scaling(data, factor):
    return np.clip(data * factor, -1.0, 1.0)

def extract_mel_spectrogram(chunk, sr):
    """
    Sesi bir resim (Mel-Spectrogram) matrisine çevirir.
    Logaritmik dB ölçeği kullanılarak görseller belirginleştirilir.
    Dönüş Çıktısı: (128, 128, 1)
    """
    # 96000 sample / 750 hop_length ~ 129 frame. 
    # n_mels=128 ile (128, 129) matris elde ederiz.
    S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_fft=2048, hop_length=750, n_mels=MEL_BANDS)
    
    # Sesi Power'dan Decibel'e çeviriyoruz (CNN resimleri okurken dB'e ihtiyaç duyar)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Tam olarak (128, 128) olmasını garantiliyoruz
    if S_db.shape[1] < TIME_STEPS:
        pad_width = TIME_STEPS - S_db.shape[1]
        S_db = np.pad(S_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif S_db.shape[1] > TIME_STEPS:
        S_db = S_db[:, :TIME_STEPS]
        
    # (128, 128) -> (128, 128, 1) resim kanalı (Grayscale)
    S_db = np.expand_dims(S_db, axis=-1)
    return S_db

def extract_features_from_file(file_path, genre):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        features = []
        
        for start in range(0, len(audio) - CHUNK_LENGTH, CHUNK_LENGTH):
            chunk = audio[start:start + CHUNK_LENGTH]
            if len(chunk) < CHUNK_LENGTH: continue
                
            features.append(extract_mel_spectrogram(chunk, sr))
            
            if genre == "Classical":
                features.append(extract_mel_spectrogram(add_noise(chunk, noise_factor=0.001), sr))
                features.append(extract_mel_spectrogram(volume_scaling(chunk, 0.6), sr))
                features.append(extract_mel_spectrogram(volume_scaling(chunk, 1.3), sr))
            else:
                features.append(extract_mel_spectrogram(add_noise(chunk, noise_factor=0.003), sr))
                features.append(extract_mel_spectrogram(pitch_shift(chunk, sr, n_steps=1.5), sr))
                features.append(extract_mel_spectrogram(pitch_shift(chunk, sr, n_steps=-1.5), sr))
            
        return features
    except Exception as e:
        print(f"Hata ({file_path}): {e}")
        return []

print("Veriler işleniyor (Görüntü İşleme: Mel-Spectrogram 128x128x1 çıkarılıyor)...")
X, y = [], []
for index, genre in enumerate(GENRES):
    genre_dir = os.path.join(DATASET_PATH, genre)
    if not os.path.exists(genre_dir): continue
    for filename in os.listdir(genre_dir):
        if filename.endswith(('.wav', '.mp3')):
            extracted = extract_features_from_file(os.path.join(genre_dir, filename), genre)
            for feature in extracted:
                X.append(feature)
                y.append(index)

X = np.array(X, dtype=np.float32) 
y = np.array(y)

print(f"Eğitim Verisi Boyutu: {X.shape}") # Örn: (N, 128, 128, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2D CNN (GÖRÜNTÜ İŞLEME) AĞ MİMARİSİ ---
input_layer = tf.keras.layers.Input(shape=(MEL_BANDS, TIME_STEPS, CHANNELS), name="input_features")

# Resmi ağın içinde otomatik normalize ediyoruz
x = tf.keras.layers.BatchNormalization(name="auto_zscore_normalizer")(input_layer)

x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.BatchNormalization()(x)

# 2D resmi tek boyutlu vektöre düzleştirme (Flatten)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)

output_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name="output_0")(x)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), batch_size=32, callbacks=[early_stopping], verbose=1)

# Keras 3 formatında kayıt
keras_model_path = "temp_tf_model.keras"
model.save(keras_model_path)

print("\nONNX opset 13 formatina donusturuluyor...")
onnx_model_path = "AutoEQ_Model.onnx"
spec = (tf.TensorSpec((None, MEL_BANDS, TIME_STEPS, CHANNELS), tf.float32, name="input_features"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=onnx_model_path)

print(f"\nBasarili! Yeni nesil 2D Gorsel (CNN) model '{onnx_model_path}' dosyasina kaydedildi.")