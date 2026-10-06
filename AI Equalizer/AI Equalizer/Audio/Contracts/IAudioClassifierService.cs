namespace AutoEQ.Audio.Contracts
{
    public interface IAudioClassifierService
    {
        /// <summary>
        /// 128x128 boyutlarındaki Mel-Spektrogram görüntüsünü (CNN formatında) alıp müziğin türüne ait olasılıkları döner.
        /// </summary>
        /// <param name="melSpectrogram">128x128 boyutunda 2D float dizisi</param>
        /// <returns>Sınıfların (Trap, Progressive Metal, Classical) olasılık dizisi</returns>
        float[] PredictProbabilities(float[,] melSpectrogram);
    }
}