using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using AutoEQ.Audio.Models;

namespace AutoEQ.Audio.Services
{
    public class ProfileManager
    {
        private readonly ConcurrentDictionary<string, EqProfile> _profiles = new(StringComparer.OrdinalIgnoreCase);
        private readonly EqProfile _fallbackProfile;

        public ProfileManager(string configPath, EqProfile fallbackProfile)
        {
            _fallbackProfile = fallbackProfile;
            LoadProfiles(configPath);
        }

        private void LoadProfiles(string path)
        {
            if (!File.Exists(path))
            {
                Console.WriteLine($"[UYARI] Profil dosyası bulunamadı: {path}");
                return;
            }
            
            var json = File.ReadAllText(path);
            var options = new JsonSerializerOptions { Converters = { new JsonStringEnumConverter() } };
            var loaded = JsonSerializer.Deserialize<List<EqProfile>>(json, options);
            
            if (loaded != null)
            {
                foreach (var profile in loaded)
                {
                    _profiles[profile.Genre] = profile;
                }
            }
        }

        public EqProfile GetProfile(string genre)
        {
            return _profiles.TryGetValue(genre, out var profile) ? profile : _fallbackProfile;
        }
    }
}
