using System;
using System.Collections.Generic;
using System.Runtime.Versioning;
using System.Threading;
using System.Threading.Tasks;
using AutoEQ.Audio.Services;

namespace AutoEQ
{
    class Program
    {
        [SupportedOSPlatform("windows")]
        static async Task Main(string[] args)
        {
            Console.WriteLine("AI Equalizer - Uçtan Uca Ses ve EQ Kontrolü Başlıyor...\n");

            // Servislerimizi ayaklandırıyoruz
            using var captureService = new WasapiLoopbackCaptureService();
            using var classifier = new OnnxAudioClassifierService();
            var mfccExtractor = new AudioFeatureExtractorService(captureService.SampleRate);

            // EQ Profil servisini başlatıyoruz
            var fallbackProfile = new AutoEQ.Audio.Models.EqProfile("Flat", 0, new List<AutoEQ.Audio.Models.EqFilter>());
            // profiles.json'ı uygulamanın çalıştığı dizinden okur
            var profileManager = new ProfileManager("profiles.json", fallbackProfile);
            var apoWriter = new ApoConfigWriter(@"C:\Program Files\EqualizerAPO\config\config.txt");

            // Kararlılık filtresi (EMA - Exponential Moving Average)
            var smoother = new AutoEQ.Audio.Algorithms.ProbabilitySmoother(classifier.Genres, alpha: 0.2f, switchThreshold: 0.60f);

            Console.WriteLine("Sistem sesi dinleniyor, Yapay Zeka analiz ediyor ve EQ dinamik olarak ayarlanıyor...");
            Console.WriteLine("Çıkış yapmak için CTRL + C tuşlarına basın.\n");

            captureService.StartCapture();
            var reader = captureService.GetAudioStreamReader();
            var cts = new CancellationTokenSource();

            Console.CancelKeyPress += (s, e) =>
            {
                e.Cancel = true;
                cts.Cancel();
            };

            try
            {
                List<float> audioBuffer = new List<float>();
                int targetSampleCount = 48000 * 2; // 2 Saniyelik karar mekanizması
                
                string activeGenre = "Flat";

                while (await reader.WaitToReadAsync(cts.Token))
                {
                    if (reader.TryRead(out var audioChunk))
                    {
                        float[] monoChunk = new float[audioChunk.Length / 2];
                        for (int i = 0; i < monoChunk.Length; i++)
                        {
                            monoChunk[i] = (audioChunk[i * 2] + audioChunk[i * 2 + 1]) / 2f;
                        }

                        audioBuffer.AddRange(monoChunk);

                        if (audioBuffer.Count >= targetSampleCount)
                        {
                            // HİÇBİR SES YÜKSELTME YAPMADAN DOĞRUDAN ANALİZ
                            var features = mfccExtractor.ExtractFeatures(audioBuffer.ToArray());
                            
                            // EMA YUMUŞATMA ENTEGRASYONU (Senaryo B)
                            float[] probabilities = classifier.PredictProbabilities(features);
                            string stableGenre = smoother.ProcessProbabilities(probabilities);

                            // Tür değiştiyse profili getir ve APO'ya yaz
                            if (stableGenre != activeGenre)
                            {
                                activeGenre = stableGenre;
                                var profileToApply = profileManager.GetProfile(activeGenre);
                                apoWriter.ApplyProfile(profileToApply);
                            }

                            float maxAmplitude = 0f;
                            foreach (var sample in audioBuffer)
                                if (Math.Abs(sample) > maxAmplitude) maxAmplitude = Math.Abs(sample);

                            // O anlık tahmin edilen en yüksek olasılıklı türü loglamak isterseniz:
                            int maxIndex = Array.IndexOf(probabilities, System.Linq.Enumerable.Max(probabilities));
                            string rawPrediction = classifier.Genres[maxIndex];

                            Console.Write($"\rAnlık Tahmin: [{rawPrediction.PadRight(18)}] | Kararlı EQ: [{activeGenre.PadRight(18)}] | Max Genlik: {maxAmplitude:F4}");
                            audioBuffer.Clear();
                        }
                    }
                }
            }
            catch (OperationCanceledException) { }
            finally
            {
                captureService.StopCapture();

                // Çıkarken EQ'yu sıfırlıyoruz (Flat profil)
                apoWriter.ApplyProfile(fallbackProfile);

                Console.WriteLine("\nServis güvenli bir şekilde kapatıldı ve EQ sıfırlandı.");
            }
        }
    }
}