using NAudio.Wave;
using System;
using System.Runtime.InteropServices;
using System.Runtime.Versioning; 
using System.Threading.Channels;
using AutoEQ.Audio.Contracts;

#pragma warning disable CS0618 

namespace AutoEQ.Audio.Services
{   
    [SupportedOSPlatform("windows")]
    public class WasapiLoopbackCaptureService : IAudioCaptureService
    {
        private WasapiLoopbackCapture? _captureDevice;
        private readonly Channel<float[]> _audioChannel;
        private bool _isDisposed;

        public int SampleRate => _captureDevice?.WaveFormat.SampleRate ?? 0;

        public WasapiLoopbackCaptureService()
        {
            var channelOptions = new BoundedChannelOptions(100)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleWriter = true,
                SingleReader = true
            };

            _audioChannel = Channel.CreateBounded<float[]>(channelOptions);
            InitializeDevice();
        }

        private void InitializeDevice()
        {
            _captureDevice = new WasapiLoopbackCapture();
            _captureDevice.DataAvailable += OnDataAvailable;
            _captureDevice.RecordingStopped += OnRecordingStopped;
        }

        public void StartCapture()
        {
            if (_captureDevice?.CaptureState != CaptureState.Capturing)
            {
                _captureDevice?.StartRecording();
            }
        }

        public void StopCapture()
        {
            if (_captureDevice?.CaptureState == CaptureState.Capturing)
            {
                _captureDevice?.StopRecording();
            }
        }

        public ChannelReader<float[]> GetAudioStreamReader()
        {
            return _audioChannel.Reader;
        }

        private void OnDataAvailable(object? sender, WaveInEventArgs e)
        {
            if (e.BytesRecorded == 0) return;

            Span<float> floatSpan = MemoryMarshal.Cast<byte, float>(e.Buffer.AsSpan(0, e.BytesRecorded));
            float[] audioChunk = floatSpan.ToArray();

            _audioChannel.Writer.TryWrite(audioChunk);
        }

        private void OnRecordingStopped(object? sender, StoppedEventArgs e)
        {
            if (e.Exception != null)
            {
                Console.WriteLine($"Ses yakalama hatası: {e.Exception.Message}");
            }
        }

        public void Dispose()
        {
            if (!_isDisposed)
            {
                StopCapture();
                if (_captureDevice != null)
                {
                    _captureDevice.DataAvailable -= OnDataAvailable;
                    _captureDevice.RecordingStopped -= OnRecordingStopped;
                    _captureDevice.Dispose();
                }
                _audioChannel.Writer.Complete();
                _isDisposed = true;
            }
            GC.SuppressFinalize(this);
        }
    }
}