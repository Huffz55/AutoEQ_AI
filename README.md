# AutoEQ AI: Real-Time Audio Classification & Dynamic Equalizer

AutoEQ AI is a dual-platform machine learning pipeline that captures real-time system audio, classifies the musical genre using a Convolutional Neural Network (CNN), and dynamically switches Equalizer APO / Peace EQ profiles without user intervention.

## System Architecture

The project is split into two synchronized environments to handle both deep learning and zero-latency real-time processing:

### 1. AI Training Pipeline (Python)
*   **Feature Extraction:** Converts raw `.wav` datasets into 128x128 Mel-Spectrogram matrices using `librosa`.
*   **Model:** 2D Convolutional Neural Network (CNN) built with TensorFlow/Keras, trained to visually recognize complex frequency and rhythm patterns (e.g., Classical vs. Progressive Metal).
*   **Export:** Converts the trained `.keras` model to a highly optimized `.onnx` format via `tf2onnx` for deployment.

### 2. Real-Time Inference Engine (C# / .NET)
*   **Audio Capture:** Zero-latency system audio capture using Windows WASAPI Loopback.
*   **DSP Processing:** Utilizes `NWaves` for real-time Mel-Spectrogram extraction.
*   **Inference:** Evaluates 2-second audio chunks in real-time using Microsoft `ONNX Runtime`.
*   **EQ Control:** Automatically maps model predictions to system-wide EQ profiles and triggers seamless transitions via Peace EQ.

## Technical Highlights & Engineering Solutions

Building a bridge between offline Python training and real-time C# inference required solving critical Digital Signal Processing (DSP) discrepancies:

*   **Cross-Platform DSP Synchronization (Slaney vs. HTK):** Aligned Python's `librosa` and C#'s `NWaves` by enforcing the Slaney Mel Scale across both environments. This eliminated the frequency shift anomalies caused by default HTK scale implementations, ensuring real-time audio maps to the exact tensor coordinates the CNN expects.
*   **Streaming Normalization (Auto-Gain Mitigation):** Replicated Librosa's `ref=np.max` normalization precisely within the C# streaming loop. This prevents local Auto-Gain artifacts during silent passages (e.g., acoustic classical solos), maintaining absolute volume contrast and preventing false positives.
*   **Time-Axis Alignment:** Calibrated WASAPI buffer sizes and hop lengths (e.g., 750 frames) to match the CNN's exact temporal expectations, preventing "slow-motion" processing distortion in real-time.
*   **Prediction Buffering:** Implemented a stable EQ caching logic to prevent erratic profile switching during complex musical transitions or intro samples.
    
## Setup & Installation

### Prerequisites
*   [.NET 8.0 SDK](https://dotnet.microsoft.com/download)
*   [Python 3.10+](https://www.python.org/downloads/)
*   [Equalizer APO](https://sourceforge.net/projects/equalizerapo/) & [Peace EQ](https://sourceforge.net/projects/peace-equalizer-apo-extension/)

### Running the Real-Time Client (C#)
1. Open `AI Equalizer.slnx` in Visual Studio.
2. Ensure Equalizer APO and Peace are running in the background.
3. Build and Run the project. The system will immediately begin analyzing desktop audio and switching profiles.

### Re-training the Model (Python)
1. Navigate to the `AutoEQ_AI` Python directory.
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare your `.wav` dataset in the appropriate folders.
4. Run the training script: `python model_trainer.py`
5. Copy the newly generated `AutoEQ_Model.onnx` to the C# execution directory.
