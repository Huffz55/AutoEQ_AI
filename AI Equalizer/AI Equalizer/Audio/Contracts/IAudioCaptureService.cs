using System;
using System.Threading.Channels;

namespace AutoEQ.Audio.Contracts
{
    public interface IAudioCaptureService : IDisposable
    {
        void StartCapture();
        void StopCapture();
        ChannelReader<float[]> GetAudioStreamReader();
        int SampleRate { get; }
    }
}