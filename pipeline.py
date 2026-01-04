
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from utils_audio import load_wav, merge_segments, sec_to_mmss
from segmentation import segment_audio
from embedding import extract_embedding
from predict_speaker import load_model
from asr import transcribe


@dataclass
class SegmentResult:
    start: float
    end: float
    speaker: str
    text: str
    confidence: float | None = None


def run_speaker_diarization_asr(
    wav_path: str,
    model_path: str = "models/speaker_model.npz",
    segment_len: float = 1.5,
    sr: int = 16000,
) -> Tuple[List[Dict], str]:
    """
    Run: segmentation -> embedding -> speaker classification -> ASR -> merge adjacent same-speaker segments.

    Returns:
      - merged_segments: list[dict] with keys: start, end, speaker, text, (optional) confidence
      - pretty_text: formatted multiline text for UI display / saving.
    """
    wav = load_wav(wav_path, sr=sr)
    segments = segment_audio(wav, sr=sr, segment_len=segment_len)

    clf, label_map = load_model(model_path)

    results: List[Dict] = []
    for start_sec, seg_wav in segments:
        emb = extract_embedding(seg_wav).reshape(1, -1)
        probs = clf.predict_proba(emb)[0]
        pred_id = int(probs.argmax())
        speaker = str(label_map[pred_id])
        conf = float(probs[pred_id])

        text = transcribe(seg_wav, sr_rate=sr)

        results.append(
            {
                "start": float(start_sec),
                "end": float(start_sec + (len(seg_wav) / sr)),
                "speaker": speaker,
                "text": text,
                "confidence": conf,
            }
        )

    merged = merge_segments(results)

    lines: List[str] = []
    lines.append("===== FINAL TIMELINE =====")
    for seg in merged:
        if not str(seg.get("text", "")).strip():
            continue
        start = sec_to_mmss(seg["start"])
        end = sec_to_mmss(seg["end"])
        speaker = seg["speaker"]
        conf = seg.get("confidence", None)
        if conf is None:
            lines.append(f"{start} – {end} | {speaker}: {seg['text']}")
        else:
            lines.append(f"{start} – {end} | {speaker} ({conf:.2f}): {seg['text']}")
    pretty = "\n".join(lines) + "\n"
    return merged, pretty
