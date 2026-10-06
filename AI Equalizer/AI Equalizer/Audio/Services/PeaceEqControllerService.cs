using System;
using System.IO;
using AutoEQ.Audio.Contracts;

namespace AutoEQ.Audio.Services
{
    public class PeaceEqControllerService : IEqControllerService
    {
        private readonly string _configPath;
        private string _currentProfile = string.Empty;

        // Equalizer APO'nun varsayılan kurulum dizini
        public PeaceEqControllerService(string configPath = @"C:\Program Files\EqualizerAPO\config\config.txt")
        {
            _configPath = configPath;
        }

        public void ApplyProfile(string genre)
        {
            // SSD'yi yormamak ve performansı korumak için, aynı tür çalıyorsa dosyayı tekrar yazmıyoruz
            if (_currentProfile == genre) return;

            string eqSettings = genre switch
            {
                // Sub-bass frekanslarını (40Hz) derinleştirip tizleri parlatır
                "Trap" => "Preamp: -4 dB\nFilter 1: ON PK Fc 40 Hz Gain 6 dB Q 1.41\nFilter 2: ON PK Fc 8000 Hz Gain 3 dB Q 1.41",

                // Davul netliğini ve double kick (80Hz) hassasiyetini artırır, çamurlanmayı (-250Hz) alır
                "Progressive Metal" => "Preamp: -3 dB\nFilter 1: ON PK Fc 80 Hz Gain 4.5 dB Q 2.0\nFilter 2: ON PK Fc 250 Hz Gain -2.5 dB Q 1.0",

                // 19. yüzyıl Osmanlı Batı müziği kompozisyonları (piyano/yaylılar) için orta-üst frekansları açar
                "Classical" => "Preamp: -2 dB\nFilter 1: ON PK Fc 2000 Hz Gain 2 dB Q 1.0",

                // İnsan sesi veya algılanamayan türler için temiz/flat profil
                _ => "Preamp: 0 dB"
            };

            try
            {
                File.WriteAllText(_configPath, eqSettings);
                _currentProfile = genre;
            }
            catch (Exception)
            {
                // Klasör yetkisi yoksa programın çökmesini engelliyoruz
            }
        }
    }
}