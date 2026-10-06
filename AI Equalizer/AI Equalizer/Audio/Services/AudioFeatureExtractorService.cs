using System;
using System.Collections.Generic;
using NWaves.FeatureExtractors;
using NWaves.FeatureExtractors.Options;

namespace AutoEQ.Audio.Services
{
    public class AudioFeatureExtractorService
    {
        private readonly FilterbankExtractor _melExtractor;
        private readonly int _targetTimeSteps = 128; // CNN Genişliği (Width)
        private readonly int _targetMelBands = 128;  // CNN Yüksekliği (Height)

        public AudioFeatureExtractorService(int sampleRate)
        {
            var options = new FilterbankOptions
            {
                SamplingRate = sampleRate,
                FilterBankSize = _targetMelBands, // 128 Mel frekans bandı
                FrameDuration = 2048.0 / sampleRate,
                HopDuration = 750.0 / sampleRate, 
                LowFrequency = 0,
                HighFrequency = sampleRate / 2,
                NonLinearity = NonLinearityType.ToDecibel, 
                SpectrumType = SpectrumType.Power        
            };

            // KRİTİK DÜZELTME: Frekans Kaymasının (Shift) Gerçek Sebebi
            // NWaves varsayılan olarak "HTK" Mel skalası kullanırken, Librosa varsayılan olarak "Slaney" Mel skalası kullanır.
            // Piksellerin yanlış frekanslara kaymasını önlemek için filtre bankasını Slaney olarak zorluyoruz.
            options.FilterBank = NWaves.Filters.Fda.FilterBanks.MelBankSlaney(_targetMelBands, 2048, sampleRate, 0, sampleRate / 2);

            _melExtractor = new FilterbankExtractor(options);
        }

        public float[,] ExtractFeatures(float[] samples)
        {
            var melFrames = _melExtractor.ComputeFrom(samples);
            int actualTimeSteps = melFrames.Count;
            
            // 128x128 boyutunda Mel-Spektrogram Görüntü Matrisi
            // Genelde resimler (Height, Width) olarak temsil edilir: Height = MelBands, Width = TimeSteps
            float[,] melSpectrogram = new float[_targetMelBands, _targetTimeSteps];

            int stepsToCopy = Math.Min(actualTimeSteps, _targetTimeSteps);

            // Mevcut kareleri matrise kopyala
            for (int t = 0; t < stepsToCopy; t++)
            {
                for (int f = 0; f < _targetMelBands; f++)
                {
                    // PyTorch/Keras formatına uyum için: [Frekans, Zaman]
                    melSpectrogram[f, t] = melFrames[t][f];
                }
            }

            // EDGE-PADDING: Ses beklenen saniyeden kısa ise / kare sayısı eksikse, resmi sağa doğru uzat (Son kareyi kopyala)
            // CNN modellerinde siyah/sıfır boşluklar bırakmak yerine son sesi uzatmak çok daha iyi sonuç verir.
            if (actualTimeSteps < _targetTimeSteps && actualTimeSteps > 0)
            {
                for (int t = actualTimeSteps; t < _targetTimeSteps; t++)
                {
                    for (int f = 0; f < _targetMelBands; f++)
                    {
                        melSpectrogram[f, t] = melFrames[actualTimeSteps - 1][f];
                    }
                }
            }

            return melSpectrogram;
        }
    }
}