import librosa
import numpy as np

class AudioFeatureExtractor:
    def __init__(self, sample_rate=48000, n_mfcc=13, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        # n_mfcc: Sesin karakteristiğini özetleyecek katsayı sayısı (Genelde 13 veya 20 kullanılır)
        self.n_mfcc = n_mfcc
        # n_fft: Frekans analizi için kullanılacak pencere boyutu
        self.n_fft = n_fft
        # hop_length: Pencereler arası kaydırma miktarı
        self.hop_length = hop_length

    def extract_features(self, audio_data):
        """
        Ham ses verisini alır ve yapay zeka modeli için MFCC özelliklerini çıkarır.
        """
        # C#'tan gelecek olan veriyi numpy float32 formatına garantiliyoruz
        audio_array = np.array(audio_data, dtype=np.float32)
        
        # Sesi MFCC matrisine dönüştür
        mfccs = librosa.feature.mfcc(
            y=audio_array, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # Zamana bağlı matrisin ortalamasını alarak modeli eğiteceğimiz 
        # sabit boyutlu (1D) bir özellik vektörü (Feature Vector) elde ediyoruz.
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        return mfccs_mean

# Modülü doğrudan çalıştırdığımızda test etmesi için bir blok ekliyoruz
if __name__ == "__main__":
    print("Özellik Çıkarıcı Testi Başlıyor...")
    extractor = AudioFeatureExtractor()
    
    # C# servisimizin saniyede gönderdiği formata benzer, 
    # 48000 uzunluğunda rastgele sayılardan oluşan sahte (dummy) bir ses verisi üretiyoruz
    dummy_audio = np.random.uniform(low=-1.0, high=1.0, size=48000)
    
    # Özellik çıkarımını başlat
    features = extractor.extract_features(dummy_audio)
    
    print(f"Çıkarılan Özellik Vektörü Boyutu: {features.shape}")
    print(f"Örnek Değerler: {features[:5]}")
    print("Test Başarılı! Librosa ham sesi başarıyla işledi ve AI için özetledi.")