from utils_audio import load_wav
from segmentation import segment_audio
from embedding import extract_embedding
from predict_speaker import load_model
from asr import transcribe

wav = load_wav("data/test/conversation.wav")
segments = segment_audio(wav)

clf, label_map = load_model("models/speaker_model.npz")

for start, seg in segments:
    emb = extract_embedding(seg).reshape(1, -1)
    spk_id = clf.predict(emb)[0]
    text = transcribe(seg)

    print(f"[{start:.2f}s] Speaker {label_map[spk_id]}: {text}")
