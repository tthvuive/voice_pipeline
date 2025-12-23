def segment_audio(wav, sr=16000, segment_len=1.5):
    seg_samples = int(sr * segment_len)
    segments = []

    for i in range(0, len(wav), seg_samples):
        seg = wav[i:i + seg_samples]
        if len(seg) > 0.5 * seg_samples:
            segments.append((i / sr, seg))

    return segments
