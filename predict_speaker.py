import numpy as np
from classifier import SoftmaxClassifier

def load_model(path):
    data = np.load(path, allow_pickle=True)
    clf = SoftmaxClassifier(
        input_dim=data["W"].shape[0],
        num_classes=data["b"].shape[0]
    )
    clf.W = data["W"]
    clf.b = data["b"]
    return clf, data["label_map"].item()
