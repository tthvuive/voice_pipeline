import numpy as np

class SoftmaxClassifier:
    def __init__(self, input_dim, num_classes, lr=0.05):
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros(num_classes)
        self.lr = lr

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict_proba(self, X):
        return self.softmax(X @ self.W + self.b)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def train(self, X, y, epochs=100):
        N = X.shape[0]
        Y = np.eye(self.b.shape[0])[y]

        for ep in range(epochs):
            probs = self.predict_proba(X)
            loss = -np.mean(np.sum(Y * np.log(probs + 1e-9), axis=1))

            dW = X.T @ (probs - Y) / N
            db = (probs - Y).mean(axis=0)

            self.W -= self.lr * dW
            self.b -= self.lr * db

            if ep % 10 == 0:
                print(f"Epoch {ep}, Loss={loss:.4f}")
