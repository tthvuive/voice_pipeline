from resemblyzer import VoiceEncoder

encoder = VoiceEncoder()

def extract_embedding(wav):
    return encoder.embed_utterance(wav)
