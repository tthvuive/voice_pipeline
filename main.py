from utils_audio import load_wav
from utils_audio import sec_to_mmss, merge_segments
from segmentation import segment_audio
from embedding import extract_embedding
from predict_speaker import load_model
from asr import transcribe
SEGMENT_LEN = 1   # seconds


wav = load_wav("data/test/conversation.wav")

segments = segment_audio(wav)

clf, label_map = load_model("models/speaker_model.npz")

results = []

for start, seg in segments:
    end = start + SEGMENT_LEN

    emb = extract_embedding(seg).reshape(1, -1)
    spk_id = clf.predict(emb)[0]
    text = transcribe(seg)

    results.append({
        "start": start,
        "end": end,
        "speaker": chr(ord("A") + spk_id),
        "text": text
    })

# 5. MERGE timeline
merged = merge_segments(results)

# 6. PRINT FINAL OUTPUT
print("\n===== FINAL TIMELINE =====\n")

for seg in merged:
    if not seg["text"].strip():
        continue

    start = sec_to_mmss(seg["start"])
    end = sec_to_mmss(seg["end"])

    print(f"{start} – {end} | Speaker {seg['speaker']}: {seg['text']}")