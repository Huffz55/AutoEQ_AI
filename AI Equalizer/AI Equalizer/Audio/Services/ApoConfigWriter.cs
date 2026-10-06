using System.IO;
using System.Globalization;
using AutoEQ.Audio.Models;

namespace AutoEQ.Audio.Services
{
    public class ApoConfigWriter
    {
        private readonly string _configPath;
        private string _lastWrittenGenre = string.Empty;

        public ApoConfigWriter(string configPath = @"C:\Program Files\EqualizerAPO\config\config.txt")
        {
            _configPath = configPath;
        }

        public void ApplyProfile(EqProfile profile)
        {
            if (profile.Genre == _lastWrittenGenre) return;

            try
            {
                using var writer = new StreamWriter(_configPath, append: false);
                
                writer.WriteLine($"# AutoEQ Dynamic Profile: {profile.Genre}");
                writer.WriteLine($"Preamp: {profile.Preamp.ToString(CultureInfo.InvariantCulture)} dB");
                
                foreach (var f in profile.Filters)
                {
                    writer.WriteLine($"Filter {f.Band}: {f.Type} Fc {f.Frequency.ToString(CultureInfo.InvariantCulture)} Hz Gain {f.Gain.ToString(CultureInfo.InvariantCulture)} dB Q {f.Q.ToString(CultureInfo.InvariantCulture)}");
                }

                _lastWrittenGenre = profile.Genre;
            }
            catch
            {
                // Klasör yetkisi yoksa uygulamanın çökmesini engelliyoruz
            }
        }
    }
}
