namespace AutoEQ.Audio.Algorithms
{
    public class ProbabilitySmoother
    {
        private readonly float _alpha;
        private readonly float[] _smoothedProbs;
        private readonly string[] _classes;
        private int _currentStableClassIndex = -1;
        private readonly float _switchThreshold;

        public ProbabilitySmoother(string[] classes, float alpha = 0.2f, float switchThreshold = 0.65f)
        {
            _classes = classes;
            _alpha = alpha;
            _switchThreshold = switchThreshold;
            _smoothedProbs = new float[classes.Length];
        }

        public string ProcessProbabilities(float[] currentProbabilities)
        {
            float maxProb = 0f;
            int maxIndex = -1;

            for (int i = 0; i < currentProbabilities.Length; i++)
            {
                _smoothedProbs[i] = (_alpha * currentProbabilities[i]) + ((1f - _alpha) * _smoothedProbs[i]);
                
                if (_smoothedProbs[i] > maxProb)
                {
                    maxProb = _smoothedProbs[i];
                    maxIndex = i;
                }
            }

            if (maxProb >= _switchThreshold)
            {
                _currentStableClassIndex = maxIndex;
            }

            return _currentStableClassIndex != -1 ? _classes[_currentStableClassIndex] : "Neutral";
        }
    }
}
