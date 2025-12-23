import os
import numpy as np
from utils_audio import load_wav
from embedding import extract_embedding
from classifier import SoftmaxClassifier

DATA_DIR = "data/train"

X, y = [], []
label_map = {}
label_id = 0

for speaker in sorted(os.listdir(DATA_DIR)):
    label_map[label_id] = speaker
    spk_dir = os.path.join(DATA_DIR, speaker)

    for f in os.listdir(spk_dir):
        if f.endswith(".wav"):
            wav = load_wav(os.path.join(spk_dir, f))
            emb = extract_embedding(wav)
            X.append(emb)
            y.append(label_id)

    label_id += 1

X = np.vstack(X)
y = np.array(y)

clf = SoftmaxClassifier(input_dim=X.shape[1], num_classes=len(label_map))
clf.train(X, y, epochs=120)

os.makedirs("models", exist_ok=True)
np.savez(
    "models/speaker_model.npz",
    W=clf.W,
    b=clf.b,
    label_map=label_map
)

print("✅ Train xong speaker classifier")
