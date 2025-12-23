import speech_recognition as sr
import tempfile
import soundfile as sf

recognizer = sr.Recognizer()

def transcribe(wav, sr_rate=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, wav, sr_rate)
        with sr.AudioFile(f.name) as source:
            audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return ""
