using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace AutoEQ.Audio.Models
{
    public enum FilterType 
    { 
        [JsonStringEnumMemberName("ON")] Peaking, 
        [JsonStringEnumMemberName("LS")] LowShelf, 
        [JsonStringEnumMemberName("HS")] HighShelf, 
        [JsonStringEnumMemberName("LP")] LowPass, 
        [JsonStringEnumMemberName("HP")] HighPass 
    }

    public record EqFilter(int Band, FilterType Type, double Frequency, double Gain, double Q);

    public record EqProfile(string Genre, double Preamp, List<EqFilter> Filters);
}
