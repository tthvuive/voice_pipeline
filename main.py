
import argparse
from pipeline import run_speaker_diarization_asr

def main():
    parser = argparse.ArgumentParser(description="Speaker diarization + ASR pipeline")
    parser.add_argument("--test", required=True, help="Path to test wav file, e.g. data/test/conversation.wav")
    parser.add_argument("--model", default="models/speaker_model.npz", help="Path to model .npz")
    parser.add_argument("--segment-len", type=float, default=1.5, help="Segment length in seconds")
    args = parser.parse_args()

    _, pretty = run_speaker_diarization_asr(
        wav_path=args.test,
        model_path=args.model,
        segment_len=args.segment_len,
    )
    print(pretty)

if __name__ == "__main__":
    main()
