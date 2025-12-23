import librosa
import numpy as np

def load_wav(path, sr=16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)
