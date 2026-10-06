using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using AutoEQ.Audio.Contracts;

namespace AutoEQ.Audio.Services
{
    public class OnnxAudioClassifierService : IAudioClassifierService, IDisposable
    {
        private readonly InferenceSession _session;
        public readonly string[] Genres = { "Trap", "Progressive Metal", "Classical" };

        public OnnxAudioClassifierService(string modelPath = "AutoEQ_Model.onnx")
        {
            _session = new InferenceSession(modelPath);
        }

        public float[] PredictProbabilities(float[,] melSpectrogram)
        {
            if (melSpectrogram.GetLength(0) != 128 || melSpectrogram.GetLength(1) != 128)
                throw new ArgumentException("Mel-Spektrogram tam olarak 128x128 boyutunda bir matris olmalıdır.");

            // 1. Librosa power_to_db(ref=np.max) Taklidi: 
            // Python'daki model_trainer.py dosyasında 'ref=np.max' kullanılmış. 
            // Bu, her 2 saniyelik parçanın KENDİ maksimumunu 0 dB'ye çeker (Auto-Gain).
            // Windows ses seviyesinden (WASAPI volume) etkilenmemek ve Python ile birebir eşleşmek için bu Lokal Max zorunludur.
            float chunkMax = float.MinValue;
            for (int h = 0; h < 128; h++)
            {
                for (int w = 0; w < 128; w++)
                {
                    if (melSpectrogram[h, w] > chunkMax) chunkMax = melSpectrogram[h, w];
                }
            }

            // 2. Referansa göre oranla (Max = 0) ve -80 dB ile sınırla (top_db=80)
            float newMin = float.MaxValue;
            float newMax = float.MinValue;
            double sumVal = 0;

            for (int h = 0; h < 128; h++)
            {
                for (int w = 0; w < 128; w++)
                {
                    float val = melSpectrogram[h, w] - chunkMax;
                    
                    if (val < -80f) val = -80f; // Sınırlandırma (Clipping)
                    
                    melSpectrogram[h, w] = val;

                    // Yeni durumun loglanması için ölçüm:
                    if (val < newMin) newMin = val;
                    if (val > newMax) newMax = val;
                    sumVal += val;
                }
            }
            
            float avgVal = (float)(sumVal / (128 * 128));
            Console.WriteLine($"\n[DEBUG] Global Referanslı Spektrogram -> Min: {newMin:F2} | Max: {newMax:F2} | Ortalama: {avgVal:F2}");

            // Keras/TensorFlow modelleri genellikle Channels-Last (NHWC) formatında çalışır.
            // Bu yüzden ONNX modeline (Batch=1, Height=128, Width=128, Channels=1) boyutunda bir tensor besliyoruz.
            var inputTensor = new DenseTensor<float>(new[] { 1, 128, 128, 1 });

            for (int h = 0; h < 128; h++)
            {
                for (int w = 0; w < 128; w++)
                {
                    inputTensor[0, h, w, 0] = melSpectrogram[h, w];
                }
            }

            // Modelin giriş adını (input name) dinamik olarak bul
            var inputName = _session.InputMetadata.Keys.First();
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            // Modeli çalıştır
            using var results = _session.Run(inputs);
            
            // Çıktıyı al (Örn: output_0)
            var outputTensor = results.First().AsTensor<float>();

            // Çıktı genelde (Batch=1, Sınıflar=3) şeklinde olur
            float[] probabilities = new float[3];
            for (int i = 0; i < 3; i++)
            {
                // Tensor'den olasılıkları çek
                probabilities[i] = outputTensor.GetValue(i); 
            }

            return probabilities;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}