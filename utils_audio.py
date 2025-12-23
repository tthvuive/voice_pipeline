import librosa
import numpy as np

def load_wav(path, sr=16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)

def sec_to_mmss(sec):
    sec = int(sec)
    m = sec // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"

def merge_segments(segments):
    merged = []

    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]

        if seg["speaker"] == last["speaker"]:
            last["end"] = seg["end"]

            if seg["text"].strip():
                last["text"] += " " + seg["text"]
        else:
            merged.append(seg)

    return merged


